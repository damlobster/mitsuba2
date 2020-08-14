#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/volume_texture.h>

#if defined(MTS_ENABLE_OPTIX)
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <cuda_texture_types.h>
    #include <texture_indirect_functions.h>

#endif

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER Sdf : public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape, m_to_world, m_to_object, set_children, get_children_string)

    MTS_IMPORT_TYPES(Texture, Volume)

    using typename Base::ScalarSize;

    Sdf(const Properties &props);

    void update() {
        m_to_object = m_to_world.inverse();
    }

    ScalarBoundingBox3f bbox() const override {
        return m_sdf->bbox();
    }

    ScalarFloat surface_area() const override {
        return 1.f;
    }


    Vector3f bspline_offsets(const Float &coord, ScalarUInt32 res) const;


    Vector3f bspline_offsets_deriv(const Float &coord, ScalarUInt32 res) const;

    MTS_INLINE Float eval_sdf(const Vector3f &p, Mask active) const;

    std::pair<Float, Vector3f> eval_sdf_and_gradient(const Vector3f &p, Mask active) const;
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
                                                     Mask active) const override;

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

    void optix_prepare_geometry() override;
#endif

    void traverse(TraversalCallback *callback) override;

    std::string to_string() const override;

    MTS_DECLARE_CLASS()
private:
    ref<Volume> m_sdf;

#if defined(MTS_ENABLE_OPTIX)
    cudaTextureObject_t m_cuda_sdf_tex = 0;
    cudaArray *m_cuArr = nullptr;
#endif

};

MTS_EXTERN_CLASS_RENDER(Sdf)
NAMESPACE_END(mitsuba)
