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

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f &ray,
                                                        Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        PreliminaryIntersection3f pi = zero<PreliminaryIntersection3f>();
        return pi;
    }

    Mask ray_test(const Ray3f & /* ray */, Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        return false;
    }

    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     PreliminaryIntersection3f pi,
                                                     HitComputeFlags flags,
                                                     Mask active) const override {
        MTS_MASK_ARGUMENT(active);
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
            texDescr.normalizedCoords = false;
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
