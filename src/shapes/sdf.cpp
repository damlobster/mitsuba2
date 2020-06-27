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
        return m_sdf->bbox();
    }

    ScalarFloat surface_area() const override {
        return 1.f;
    }

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    std::pair<Mask, Float> ray_intersect(const Ray3f &ray,
                                         Float * /*cache*/,
                                         Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        return { false, math::Infinity<Float> };
    }

    Mask ray_test(const Ray3f &ray, Mask active) const override {
        return false;
    }

    void fill_surface_interaction(const Ray3f &ray, const Float * /*cache*/,
                                  SurfaceInteraction3f &si_out, Mask active) const override {
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
            if (!m_optix_data_ptr)
                m_optix_data_ptr = cuda_malloc(sizeof(OptixSdfData));

            ScalarFloat *raw_sdf_data = m_sdf->data().data();
            ScalarVector3i res = m_sdf->resolution();
            OptixSdfData data = { bbox(),
                                  m_to_world,
                                  m_to_object,
                                  raw_sdf_data,
                                  res };
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
};

MTS_IMPLEMENT_CLASS_VARIANT(Sdf, Shape)
MTS_EXPORT_PLUGIN(Sdf, "Sdf intersection primitive");
NAMESPACE_END(mitsuba)
