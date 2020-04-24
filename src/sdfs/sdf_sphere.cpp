#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/sdf.h>

#if defined(MTS_ENABLE_EMBREE)
    #include <embree3/rtcore.h>
#endif

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class SphereSDF final : public SDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(SDF)
    MTS_IMPORT_TYPES()

    SphereSDF(const Properties &props) : Base(props) {
        m_object_to_world =
            ScalarTransform4f::translate(ScalarVector3f(props.point3f("center", ScalarPoint3f(0.f))));
        m_radius = props.float_("radius", 1.f);

        if (props.has_property("to_world")) {
            ScalarTransform4f object_to_world = props.transform("to_world");
            ScalarFloat radius = norm(object_to_world * ScalarVector3f(1, 0, 0));
            // Remove the scale from the object-to-world transform
            m_object_to_world =
                object_to_world
                * ScalarTransform4f::scale(ScalarVector3f(1.f / radius))
                * m_object_to_world;
            m_radius *= radius;
        }

        /// Are the sphere normals pointing inwards? default: no
        m_flip_normals = props.bool_("flip_normals", false);
        m_center = m_object_to_world * ScalarPoint3f(0, 0, 0);
        m_world_to_object = m_object_to_world.inverse();

        if (m_radius <= 0.f) {
            m_radius = std::abs(m_radius);
            m_flip_normals = !m_flip_normals;
        }
    }

    ScalarBoundingBox3f bbox() const override {
        ScalarBoundingBox3f bbox;
        bbox.min = m_center - m_radius - 2*math::RayEpsilon<Float>;
        bbox.max = m_center + m_radius + 2*math::RayEpsilon<Float>;
        return bbox;
    }

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    Float distance(const Interaction3f &it, Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        return select(active, norm(it.p - m_center) - m_radius, math::Infinity<Float>);
    }

    void fill_surface_interaction(const Ray3f &ray, const Float * /*cache*/,
                                  SurfaceInteraction3f &si_out, Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        SurfaceInteraction3f si(si_out);

        si.p = ray(si.t);
        si.p = fmadd(ray.d, distance(si, active), si.p);
        si.sh_frame.n = normalize(si.p - m_center);
        auto [dp_du, dp_dv] = coordinate_system(si.sh_frame.n);
        si.dp_du = dp_du;
        si.dp_dv = dp_dv;

        if (m_flip_normals)
            si.sh_frame.n = -si.sh_frame.n;

        si.n = si.sh_frame.n;
        si.time = ray.time;

        si.wi = select(active, si.to_local(-ray.d), -ray.d);

        si_out[active] = si;
    }

    //! @}
    // =============================================================

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("center", m_center);
        callback->put_parameter("radius", m_radius);
        Base::traverse(callback);
    }

    void parameters_changed() override {
        Base::parameters_changed();
        m_object_to_world = ScalarTransform4f::translate(m_center);
        m_world_to_object = m_object_to_world.inverse();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SphereSDF[" << std::endl
            << "  radius = "  << m_radius << "," << std::endl
            << "  center = "  << m_center << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ScalarTransform4f m_object_to_world;
    ScalarTransform4f m_world_to_object;
    ScalarPoint3f m_center;
    ScalarFloat m_radius;
    bool m_flip_normals;
};

MTS_IMPLEMENT_CLASS_VARIANT(SphereSDF, SDF)
MTS_EXPORT_PLUGIN(SphereSDF, "Sphere SDF intersection primitive");
NAMESPACE_END(mitsuba)
