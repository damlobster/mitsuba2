#pragma once

#include <mitsuba/core/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/core/struct.h>
#include <mitsuba/core/transform.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER SDF : public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape)
    MTS_IMPORT_TYPES(BSDF, Medium)

    using typename Base::ScalarIndex;
    using typename Base::ScalarSize;

    /// Create a new Signed Distance Function (SDF)
    SDF(const std::string &name);

    // =========================================================================
    //! @{ \name Accessors (vertices, faces, normals, etc)
    // =========================================================================


    /// @}
    // =========================================================================

    /// Compute smooth vertex normals and replace the current normal values
    // void recompute_gradient_field();

    // =============================================================
    //! @{ \name Shape interface implementation
    // =============================================================

    virtual Float distance(const Point3f &p, Mask active) const = 0;

    virtual std::pair<Mask, Float>
    ray_intersect(const Ray3f &ray, Float * /*cache*/, Mask active) const override;

    virtual void fill_surface_interaction(const Ray3f &ray,
                                          const Float* /**/,
                                          SurfaceInteraction3f &si,
                                          Mask active = true) const override;

    virtual ScalarBoundingBox3f bbox() const override = 0;

    ScalarSize primitive_count() const override { return 1; }

    ScalarSize effective_primitive_count() const override { return 1; }

    /// @}
    // =========================================================================

protected:
    SDF(const Properties &);
    inline SDF() {}
    virtual ~SDF();

    MTS_DECLARE_CLASS()
protected:
    std::string m_name;
    ScalarBoundingBox3f m_bbox;
    ScalarTransform4f m_to_world;
};

MTS_EXTERN_CLASS_RENDER(SDF)
NAMESPACE_END(mitsuba)

// -----------------------------------------------------------------------
//! @{ \name Enoki accessors for dynamic vectorization
// -----------------------------------------------------------------------

// Enable usage of array pointers for our types
ENOKI_CALL_SUPPORT_TEMPLATE_BEGIN(mitsuba::SDF)
    ENOKI_CALL_SUPPORT_METHOD(fill_surface_interaction)
    /*ENOKI_CALL_SUPPORT_GETTER_TYPE(faces, m_faces, uint8_t*)
    ENOKI_CALL_SUPPORT_GETTER_TYPE(vertices, m_vertices, uint8_t*)*/
ENOKI_CALL_SUPPORT_TEMPLATE_END(mitsuba::SDF)

//! @}
// -----------------------------------------------------------------------
