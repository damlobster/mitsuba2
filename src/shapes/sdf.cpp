#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/volume_texture.h>


#if defined(MTS_ENABLE_OPTIX)
    #include <cuda.h>

    #include <cuda_runtime.h>
    #include <texture_fetch_functions.h>
    #include <cuda_texture_types.h>
    #include <texture_indirect_functions.h>
    #include "device_launch_parameters.h"

    #include <mitsuba/render/optix_api.h>
    #include "optix/sdf.cuh"
#endif

NAMESPACE_BEGIN(mitsuba)


template <typename Float, typename Spectrum>
class Sdf final : public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape, m_to_world, m_to_object, set_children, get_children_string)

    MTS_IMPORT_TYPES(Texture, Volume)

    using typename Base::ScalarSize;

    Sdf(const Properties &props) : Base(props) {
        // Find SDF texture parameter
        for (auto &kv : props.objects()) {
            Volume *volume = dynamic_cast<Volume *>(kv.second.get());
            if (volume) {
                m_sdf = volume;
            }
        }
        update();
        set_children();
    }

    void update() {
        m_to_object = m_to_world.inverse();
    }

    ScalarBoundingBox3f bbox() const override {
        if (m_sdf) {
            return m_sdf->bbox();
        } else {
            ScalarBoundingBox3f aabb(ScalarVector3f(0.f, 0.f, 0.f), ScalarVector3f(1.f, 1.f, 1.f));
            Log(Info, "Using default AABB: %s", aabb);
            return aabb;
        }
    }

    ScalarFloat surface_area() const override {
        return 1.f;
    }


    Vector3f bspline_offsets(const Float &coord, ScalarUInt32 res) const {
        // TODO: This currently assumes the SDF to live in 0...1 coordinates
        Float coord_hg = coord * res - 0.5f;
        Float int_part = enoki::floor(coord_hg);
        Float tc = int_part + 0.5f;
        Float a = coord_hg - int_part;

        Float a2 = a * a;
        Float a3 = a2 * a;
        ScalarFloat norm_fac = 1.0f / 6.0f;

        Float w0 = norm_fac * fmadd(a, fmadd(a, -a + 3, - 3), 1);
        Float w1 = norm_fac * fmadd(3 * a2, a - 2, 4);
        Float w2 = norm_fac * fmadd(a, 3 * (-a2 + a + 1), 1);
        Float w3 = norm_fac * a3;

        // Original less optimized implementation
        // Float w0 = norm_fac * (    -a3 + 3 * a2 - 3 * a + 1);
        // Float w1 = norm_fac * ( 3 * a3 - 6 * a2         + 4);
        // Float w2 = norm_fac * (-3 * a3 + 3 * a2 + 3 * a + 1);
        // Float w3 = norm_fac * a3;

        Float g0 = w0 + w1;
        ScalarFloat inv_res = 1.f / res;
        Float h0 = inv_res * (tc - 1 + w1 / (w0 + w1));
        Float h1 = inv_res * (tc + 1 + w3 / (w2 + w3));
        return Vector3f(h0, h1, g0);
    }


    Vector3f bspline_offsets_deriv(const Float &coord, ScalarUInt32 res) const {
        Float coord_hg = coord * res - 0.5f;
        Float int_part = enoki::floor(coord_hg);
        Float tc = int_part + 0.5f;
        Float a = coord_hg - int_part;

        Float a2 = a * a;
        ScalarFloat norm_fac = 1.0f / 6.0f;

        // This is the gradient of the bspline kernel
        Float w0 = norm_fac * (-3 * a2 +  6 * a - 3);
        Float w1 = norm_fac * ( 9 * a2 - 12 * a    );
        Float w2 = norm_fac * (-9 * a2 +  6 * a + 3);
        Float w3 = norm_fac * 3 * a2;

        // As in primal version, use the weight values to determine lookup locations
        Float g0 = w0 + w1;
        ScalarFloat inv_res = 1.f / res;
        Float h0 = inv_res * (tc - 1 + w1 / (w0 + w1));
        Float h1 = inv_res * (tc + 1 + w3 / (w2 + w3));
        return Vector3f(h0, h1, g0);
    }

    MTS_INLINE Float eval_sdf(const Vector3f &p, Mask active) const {
        // TODO: This only works on RGB mode for now
        Interaction3f it;
        it.p = p;
        return m_sdf->eval_1(it, active);
    }

    std::pair<Float, Vector3f> eval_sdf_and_gradient(const Vector3f &p, Mask active) const {
        // Compute both value and gradient of the SDF at a position p
        // This is done going through grid3d in order to propagate gradients
        // TODO: It would probably be more efficient to implement at least parts
        // of this using CUDA to prevent having so many gather calls

        Vector3f o_x = bspline_offsets(p.x(), m_sdf->resolution().x());
        Vector3f o_y = bspline_offsets(p.y(), m_sdf->resolution().y());
        Vector3f o_z = bspline_offsets(p.z(), m_sdf->resolution().z());
        Vector3f grad_o_x = bspline_offsets_deriv(p.x(), m_sdf->resolution().x());
        Vector3f grad_o_y = bspline_offsets_deriv(p.y(), m_sdf->resolution().y());
        Vector3f grad_o_z = bspline_offsets_deriv(p.z(), m_sdf->resolution().z());

        // fetch eight linearly interpolated inputs at the different locations computed earlier
        Float tex000 = eval_sdf(Vector3f(o_x.x(), o_y.x(), o_z.x()), active);
        Float tex100 = eval_sdf(Vector3f(o_x.y(), o_y.x(), o_z.x()), active);
        Float tex010 = eval_sdf(Vector3f(o_x.x(), o_y.y(), o_z.x()), active);
        Float tex110 = eval_sdf(Vector3f(o_x.y(), o_y.y(), o_z.x()), active);
        Float tex001 = eval_sdf(Vector3f(o_x.x(), o_y.x(), o_z.y()), active);
        Float tex101 = eval_sdf(Vector3f(o_x.y(), o_y.x(), o_z.y()), active);
        Float tex011 = eval_sdf(Vector3f(o_x.x(), o_y.y(), o_z.y()), active);
        Float tex111 = eval_sdf(Vector3f(o_x.y(), o_y.y(), o_z.y()), active);

        // Interpolate all these results using the precomputed weights
        Float tex00 = lerp(tex001, tex000, o_z.z());
        Float tex01 = lerp(tex011, tex010, o_z.z());
        Float tex10 = lerp(tex101, tex100, o_z.z());
        Float tex11 = lerp(tex111, tex110, o_z.z());
        Float tex0  = lerp(tex01, tex00, o_y.z());
        Float tex1  = lerp(tex11, tex10, o_y.z());
        Float sdf_value = lerp(tex1, tex0, o_x.z());

        Float gx_tex000 = eval_sdf(Vector3f(grad_o_x.x(), o_y.x(), o_z.x()), active);
        Float gx_tex100 = eval_sdf(Vector3f(grad_o_x.y(), o_y.x(), o_z.x()), active);
        Float gx_tex010 = eval_sdf(Vector3f(grad_o_x.x(), o_y.y(), o_z.x()), active);
        Float gx_tex110 = eval_sdf(Vector3f(grad_o_x.y(), o_y.y(), o_z.x()), active);
        Float gx_tex001 = eval_sdf(Vector3f(grad_o_x.x(), o_y.x(), o_z.y()), active);
        Float gx_tex101 = eval_sdf(Vector3f(grad_o_x.y(), o_y.x(), o_z.y()), active);
        Float gx_tex011 = eval_sdf(Vector3f(grad_o_x.x(), o_y.y(), o_z.y()), active);
        Float gx_tex111 = eval_sdf(Vector3f(grad_o_x.y(), o_y.y(), o_z.y()), active);
        Float gy_tex000 = eval_sdf(Vector3f(o_x.x(), grad_o_y.x(), o_z.x()), active);
        Float gy_tex100 = eval_sdf(Vector3f(o_x.y(), grad_o_y.x(), o_z.x()), active);
        Float gy_tex010 = eval_sdf(Vector3f(o_x.x(), grad_o_y.y(), o_z.x()), active);
        Float gy_tex110 = eval_sdf(Vector3f(o_x.y(), grad_o_y.y(), o_z.x()), active);
        Float gy_tex001 = eval_sdf(Vector3f(o_x.x(), grad_o_y.x(), o_z.y()), active);
        Float gy_tex101 = eval_sdf(Vector3f(o_x.y(), grad_o_y.x(), o_z.y()), active);
        Float gy_tex011 = eval_sdf(Vector3f(o_x.x(), grad_o_y.y(), o_z.y()), active);
        Float gy_tex111 = eval_sdf(Vector3f(o_x.y(), grad_o_y.y(), o_z.y()), active);
        Float gz_tex000 = eval_sdf(Vector3f(o_x.x(), o_y.x(), grad_o_z.x()), active);
        Float gz_tex100 = eval_sdf(Vector3f(o_x.y(), o_y.x(), grad_o_z.x()), active);
        Float gz_tex010 = eval_sdf(Vector3f(o_x.x(), o_y.y(), grad_o_z.x()), active);
        Float gz_tex110 = eval_sdf(Vector3f(o_x.y(), o_y.y(), grad_o_z.x()), active);
        Float gz_tex001 = eval_sdf(Vector3f(o_x.x(), o_y.x(), grad_o_z.y()), active);
        Float gz_tex101 = eval_sdf(Vector3f(o_x.y(), o_y.x(), grad_o_z.y()), active);
        Float gz_tex011 = eval_sdf(Vector3f(o_x.x(), o_y.y(), grad_o_z.y()), active);
        Float gz_tex111 = eval_sdf(Vector3f(o_x.y(), o_y.y(), grad_o_z.y()), active);

        // Interpolate all these results using the precomputed weights
        Vector3f sdf_grad;
        {
            Float tex00 = lerp(gx_tex001, gx_tex000, o_z.z());
            Float tex01 = lerp(gx_tex011, gx_tex010, o_z.z());
            Float tex10 = lerp(gx_tex101, gx_tex100, o_z.z());
            Float tex11 = lerp(gx_tex111, gx_tex110, o_z.z());
            Float tex0  = lerp(tex01, tex00, o_y.z());
            Float tex1  = lerp(tex11, tex10, o_y.z());
            sdf_grad.x() = (tex1 - tex0) * grad_o_x.z();
        }
        {
            Float tex00 = lerp(gy_tex001, gy_tex000, o_z.z());
            Float tex01 = lerp(gy_tex011, gy_tex010, o_z.z());
            Float tex10 = lerp(gy_tex101, gy_tex100, o_z.z());
            Float tex11 = lerp(gy_tex111, gy_tex110, o_z.z());
            Float tex0  = (tex01 - tex00) * grad_o_y.z();
            Float tex1  = (tex11 - tex10) * grad_o_y.z();
            sdf_grad.y() = lerp(tex1, tex0, o_x.z());
        }
        {
            Float tex00 = (gz_tex001 - gz_tex000) * grad_o_z.z();
            Float tex01 = (gz_tex011 - gz_tex010) * grad_o_z.z();
            Float tex10 = (gz_tex101 - gz_tex100) * grad_o_z.z();
            Float tex11 = (gz_tex111 - gz_tex110) * grad_o_z.z();
            Float tex0  = lerp(tex01, tex00, o_y.z());
            Float tex1  = lerp(tex11, tex10, o_y.z());
            sdf_grad.z() = lerp(tex1, tex0, o_x.z());
        }
        return {sdf_value, sdf_grad};
    }

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f & /* ray */,
                                                        Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        NotImplementedError("ray_intersect_preliminary");
    }

    Mask ray_test(const Ray3f & /* ray */, Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        NotImplementedError("ray_test");
    }

    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     PreliminaryIntersection3f pi,
                                                     HitComputeFlags flags,
                                                     Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        bool differentiable = false;
        if constexpr (is_diff_array_v<Float>)
            differentiable = parameters_grad_enabled();

        // Disable invalid lanes
        active &= pi.is_valid();

        // Create final surface interaction: Will only have t, p and n set (no UV and UV derivatives)
        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        // Recompute ray intersection to get differentiable t
        if (differentiable && !has_flag(flags, HitComputeFlags::NonDifferentiable)) {
            auto [sdf_value, sdf_grad] = eval_sdf_and_gradient(ray(pi.t), active);
            si.n = -normalize(sdf_grad); // Normalize differentiable normal just in case of numerical issues
            si.sh_frame = Frame3f(si.n); // Build a new coordinate frame around n

            // Compute t gradient: subtract detached SDF value to disable term in forward mode
            Float t_grad_term = -sdf_value / dot(ray.d, si.n);
            Float t = Float(detach(pi.t)) + (t_grad_term - Float(detach(t_grad_term)));
            si.t = select(active, t, math::Infinity<Float>);
            si.p = ray(si.t);
            // These tangent vectors are then used later to potentially re-initialize the sh frame
            si.dp_du = si.sh_frame.s;
            si.dp_dv = si.sh_frame.t;
        }
        return si;
    }

    bool parameters_grad_enabled() const override {
        // For now, just always assume that we want
        // differentiable ray intersections for SDF
        return true;
    }


    //! @}
    // =============================================================


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
            ScalarVector3i res = m_sdf->resolution();
            Log(Info, "Requesting optix geometry...");
            if (!m_optix_data_ptr)
                m_optix_data_ptr = cuda_malloc(sizeof(OptixSdfData));

            if (!m_sdf)
                Log(Error, "SDF Not initialized");

            const auto &sdf_data = m_sdf->data();
            const float * raw_sdf_data = sdf_data.data();


            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
            // Create 3D array and copy data to it:
            cudaArray *d_cuArr;
            cudaMalloc3DArray(&d_cuArr, &channelDesc, make_cudaExtent(res.x(), res.y(), res.z()), 0);
            cudaMemcpy3DParms copyParams = {0};
            copyParams.srcPtr   = make_cudaPitchedPtr((void*) raw_sdf_data, res.x() * sizeof(float), res.y(), res.z());
            copyParams.dstArray = d_cuArr;
            copyParams.extent   = make_cudaExtent(res.x(), res.y(), res.z());
            copyParams.kind     = cudaMemcpyDeviceToDevice;
            cudaMemcpy3D(&copyParams);

            // Setup texture sampler object
            cudaResourceDesc    texRes;
            memset(&texRes, 0, sizeof(cudaResourceDesc));
            texRes.resType = cudaResourceTypeArray;
            texRes.res.array.array  = d_cuArr;
            cudaTextureDesc     texDescr;
            memset(&texDescr, 0, sizeof(cudaTextureDesc));
            texDescr.normalizedCoords = true;
            texDescr.filterMode = cudaFilterModeLinear;
            texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
            texDescr.addressMode[1] = cudaAddressModeClamp;
            texDescr.addressMode[2] = cudaAddressModeClamp;
            texDescr.readMode = cudaReadModeElementType;
            m_cuda_sdf_tex = 0;
            cudaCreateTextureObject(&m_cuda_sdf_tex, &texRes, &texDescr, NULL);


            OptixSdfData data = { bbox(),
                                  m_to_world,
                                  m_to_object,
                                  raw_sdf_data,
                                  res,
                                  m_cuda_sdf_tex};
            cuda_memcpy_to_device(m_optix_data_ptr, &data, sizeof(OptixSdfData));
        }
    }
#endif

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Sdf[" << std::endl
            << "  " << string::indent(get_children_string()) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Volume> m_sdf;
    cudaTextureObject_t m_cuda_sdf_tex;

};

MTS_IMPLEMENT_CLASS_VARIANT(Sdf, Shape)
MTS_EXPORT_PLUGIN(Sdf, "Sdf intersection primitive");
NAMESPACE_END(mitsuba)
