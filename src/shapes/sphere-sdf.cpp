#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>

#if defined(MTS_ENABLE_OPTIX)
    #include <mitsuba/render/optix_api.h>
    #include "optix/sphere-sdf.cuh"
#endif

NAMESPACE_BEGIN(mitsuba)


template <typename Float, typename Spectrum>
class SphereSdf final : public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape, m_to_world, m_to_object, set_children,
                    get_children_string, parameters_grad_enabled)
    MTS_IMPORT_TYPES()

    using typename Base::ScalarSize;

    SphereSdf(const Properties &props) : Base(props) {
        /// Are the SphereSdf normals pointing inwards? default: no
        m_flip_normals = props.bool_("flip_normals", false);

        // Update the to_world transform if radius and center are also provided
        m_to_world = m_to_world * ScalarTransform4f::translate(props.point3f("center", 0.f));
        m_to_world = m_to_world * ScalarTransform4f::scale(props.float_("radius", 1.f));

        update();
        set_children();
    }

    void update() {
        // Extract center and radius from to_world matrix (25 iterations for numerical accuracy)
        auto [S, Q, T] = transform_decompose(m_to_world.matrix, 25);

        if (abs(S[0][1]) > 1e-6f || abs(S[0][2]) > 1e-6f || abs(S[1][0]) > 1e-6f ||
            abs(S[1][2]) > 1e-6f || abs(S[2][0]) > 1e-6f || abs(S[2][1]) > 1e-6f)
            Log(Warn, "'to_world' transform shouldn't contain any shearing!");

        if (!(abs(S[0][0] - S[1][1]) < 1e-6f && abs(S[0][0] - S[2][2]) < 1e-6f))
            Log(Warn, "'to_world' transform shouldn't contain non-uniform scaling!");

        m_center = T;
        m_radius = S[0][0];

        if (m_radius <= 0.f) {
            m_radius = std::abs(m_radius);
            m_flip_normals = !m_flip_normals;
        }

        // Reconstruct the to_world transform with uniform scaling and no shear
        m_to_world = transform_compose(ScalarMatrix3f(m_radius), Q, T);
        m_to_object = m_to_world.inverse();

        m_inv_surface_area = rcp(surface_area());
    }

    ScalarBoundingBox3f bbox() const override {
        ScalarBoundingBox3f bbox;
        bbox.min = m_center - m_radius;
        bbox.max = m_center + m_radius;
        return bbox;
    }

    ScalarFloat surface_area() const override {
        return 4.f * math::Pi<ScalarFloat> * m_radius * m_radius;
    }

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f &ray,
                                                        Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        using Double = std::conditional_t<is_cuda_array_v<Float>, Float, Float64>;
        using Double3 = Vector<Double, 3>;

        Double mint = Double(ray.mint);
        Double maxt = Double(ray.maxt);

        Double3 o = Double3(ray.o) - Double3(m_center);
        Double3 d(ray.d);

        Double A = squared_norm(d);
        Double B = scalar_t<Double>(2.f) * dot(o, d);
        Double C = squared_norm(o) - sqr((scalar_t<Double>) m_radius);

        auto [solution_found, near_t, far_t] = math::solve_quadratic(A, B, C);

        // Sphere doesn't intersect with the segment on the ray
        Mask out_bounds = !(near_t <= maxt && far_t >= mint); // NaN-aware conditionals

        // Sphere fully contains the segment of the ray
        Mask in_bounds = near_t < mint && far_t > maxt;

        active &= solution_found && !out_bounds && !in_bounds;

        PreliminaryIntersection3f pi = zero<PreliminaryIntersection3f>();
        pi.t = select(active,
                      select(near_t < mint, Float(far_t), Float(near_t)),
                      math::Infinity<Float>);
        pi.shape = this;

        return pi;
    }

    Mask ray_test(const Ray3f &ray, Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        using Double = std::conditional_t<is_cuda_array_v<Float>, Float, Float64>;
        using Double3 = Vector<Double, 3>;

        Double mint = Double(ray.mint);
        Double maxt = Double(ray.maxt);

        Double3 o = Double3(ray.o) - Double3(m_center);
        Double3 d(ray.d);

        Double A = squared_norm(d);
        Double B = scalar_t<Double>(2.f) * dot(o, d);
        Double C = squared_norm(o) - sqr((scalar_t<Double>) m_radius);

        auto [solution_found, near_t, far_t] = math::solve_quadratic(A, B, C);

        // Sphere doesn't intersect with the segment on the ray
        Mask out_bounds = !(near_t <= maxt && far_t >= mint); // NaN-aware conditionals

        // Sphere fully contains the segment of the ray
        Mask in_bounds  = near_t < mint && far_t > maxt;

        return solution_found && !out_bounds && !in_bounds && active;
    }

    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     PreliminaryIntersection3f pi,
                                                     HitComputeFlags flags,
                                                     Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        bool differentiable = false;
        if constexpr (is_diff_array_v<Float>)
            differentiable = requires_gradient(ray.o) ||
                             requires_gradient(ray.d) ||
                             parameters_grad_enabled();

        // Recompute ray intersection to get differentiable prim_uv and t
        if (differentiable && !has_flag(flags, HitComputeFlags::NonDifferentiable))
            pi = ray_intersect_preliminary(ray, active);

        active &= pi.is_valid();

        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        si.t = select(active, pi.t, math::Infinity<Float>);

        si.sh_frame.n = normalize(ray(pi.t) - m_center);

        // Re-project onto the sphere to improve accuracy
        si.p = fmadd(si.sh_frame.n, m_radius, m_center);

        if (likely(has_flag(flags, HitComputeFlags::UV))) {
            Vector3f local = m_to_object.transform_affine(si.p);

            Float rd_2  = sqr(local.x()) + sqr(local.y()),
                  theta = unit_angle_z(local),
                  phi   = atan2(local.y(), local.x());

            masked(phi, phi < 0.f) += 2.f * math::Pi<Float>;

            si.uv = Point2f(phi * math::InvTwoPi<Float>, theta * math::InvPi<Float>);
            if (likely(has_flag(flags, HitComputeFlags::dPdUV))) {
                si.dp_du = Vector3f(-local.y(), local.x(), 0.f);

                Float rd      = sqrt(rd_2),
                      inv_rd  = rcp(rd),
                      cos_phi = local.x() * inv_rd,
                      sin_phi = local.y() * inv_rd;

                si.dp_dv = Vector3f(local.z() * cos_phi,
                                    local.z() * sin_phi,
                                    -rd);

                Mask singularity_mask = active && eq(rd, 0.f);
                if (unlikely(any(singularity_mask)))
                    si.dp_dv[singularity_mask] = Vector3f(1.f, 0.f, 0.f);

                si.dp_du = m_to_world * si.dp_du * (2.f * math::Pi<Float>);
                si.dp_dv = m_to_world * si.dp_dv * math::Pi<Float>;
            }
        }

        if (m_flip_normals)
            si.sh_frame.n = -si.sh_frame.n;

        si.n = si.sh_frame.n;

        if (has_flag(flags, HitComputeFlags::dNSdUV)) {
            ScalarFloat inv_radius = (m_flip_normals ? -1.f : 1.f) / m_radius;
            si.dn_du = si.dp_du * inv_radius;
            si.dn_dv = si.dp_dv * inv_radius;
        }

        return si;
    }

    // std::pair<Vector3f, Vector3f> normal_derivative(const SurfaceInteraction3f &si,
    //                                                 bool /*shading_frame*/,
    //                                                 Mask active) const override {
    //     MTS_MASK_ARGUMENT(active);
    //     ScalarFloat inv_radius = (m_flip_normals ? -1.f : 1.f) / m_radius;
    //     return { si.dp_du * inv_radius, si.dp_dv * inv_radius };
    // }

    //! @}
    // =============================================================

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/) override {
        update();
        Base::parameters_changed();
#if defined(MTS_ENABLE_OPTIX)
        optix_prepare_geometry();
#endif
    }

#if defined(MTS_ENABLE_OPTIX)
    using Base::m_optix_data_ptr;

    void optix_prepare_geometry() override {
        if constexpr (is_cuda_array_v<Float>) {
            if (!m_optix_data_ptr)
                m_optix_data_ptr = cuda_malloc(sizeof(OptixSphereSdfData));

            OptixSphereSdfData data = { bbox(), m_to_world, m_to_object,
                                     m_center, m_radius, m_flip_normals };

            cuda_memcpy_to_device(m_optix_data_ptr, &data, sizeof(OptixSphereSdfData));
        }
    }
#endif

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SphereSdf[" << std::endl
            << "  to_world = " << string::indent(m_to_world) << "," << std::endl
            << "  center = "  << m_center << "," << std::endl
            << "  radius = "  << m_radius << "," << std::endl
            << "  surface_area = " << surface_area() << "," << std::endl
            << "  " << string::indent(get_children_string()) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    /// Center in world-space
    ScalarPoint3f m_center;
    /// Radius in world-space
    ScalarFloat m_radius;

    ScalarFloat m_inv_surface_area;

    bool m_flip_normals;
};

MTS_IMPLEMENT_CLASS_VARIANT(SphereSdf, Shape)
MTS_EXPORT_PLUGIN(SphereSdf, "SphereSdf intersection primitive");
NAMESPACE_END(mitsuba)
