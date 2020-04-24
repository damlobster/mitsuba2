#pragma once

#include <mitsuba/core/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/core/struct.h>
#include <mitsuba/core/transform.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER SDF : public Mesh<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Mesh, m_is_sdf, m_mesh, m_vertices, m_faces, m_normal_offset, m_bbox, m_vertex_size,
                    m_face_size, m_to_world, m_vertex_count, m_face_count, m_vertex_struct,
                    m_face_struct, bbox)
    MTS_IMPORT_TYPES(BSDF)

    using typename Base::ScalarSize;
    using typename Base::ScalarIndex;
    using typename Base::VertexHolder;
    using typename Base::FaceHolder;
    using typename Base::InputFloat;
    using typename Base::InputPoint3f;

    // =========================================================================
    //! @{ \name Accessors (vertices, faces, normals, etc)
    // =========================================================================

    /// @}
    // =========================================================================

    // =============================================================
    //! @{ \name SDF interface implementation
    // =============================================================

    virtual Float distance(const Interaction3f &inter, Mask active) const = 0;

    virtual void fill_surface_interaction(const Ray3f &ray,
                                          const Float* /**/,
                                          SurfaceInteraction3f &si,
                                          Mask active = true) const override {
        auto si_ = _fill_surface_interaction(ray, nullptr, si, active);
        si[active] = si_;
    };

    virtual SurfaceInteraction3f _fill_surface_interaction(const Ray3f &ray,
                                          const Float* /**/,
                                          const SurfaceInteraction3f &si,
                                          Mask active = true) const = 0;

    virtual std::pair<Mask, Float>
    ray_intersect(const Ray3f &ray, Float *cache, Mask active) const override;

    virtual ScalarBoundingBox3f bbox(ScalarIndex /**/) const override { return bbox(); };
    virtual ScalarBoundingBox3f bbox(ScalarIndex /**/, const ScalarBoundingBox3f &/**/) const override { return bbox(); };

    /*
    virtual void traverse(TraversalCallback *callback) override;

    virtual void parameters_changed() override;
    */

    /// @}
    // =========================================================================

protected:
    SDF(const Properties &);
    virtual ~SDF();

    void initialize_mesh_vertices();

    MTS_DECLARE_CLASS()
protected:
    ScalarInt32 m_sphere_tracing_steps;
};

MTS_EXTERN_CLASS_RENDER(SDF)
NAMESPACE_END(mitsuba)

// -----------------------------------------------------------------------
//! @{ \name Enoki accessors for dynamic vectorization
// -----------------------------------------------------------------------

// Enable usage of array pointers for our types
ENOKI_CALL_SUPPORT_TEMPLATE_BEGIN(mitsuba::SDF)
    ENOKI_CALL_SUPPORT_METHOD(distance)
    ENOKI_CALL_SUPPORT_METHOD(fill_surface_interaction)
    ENOKI_CALL_SUPPORT_METHOD(_fill_surface_interaction)
    ENOKI_CALL_SUPPORT_METHOD(ray_intersect)
ENOKI_CALL_SUPPORT_TEMPLATE_END(mitsuba::SDF)

//! @}
// -----------------------------------------------------------------------
