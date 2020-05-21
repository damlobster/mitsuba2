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
        auto si_ = _fill_surface_interaction(ray, 0.0f, nullptr, si, active);
        si[active] = si_;
    };

    virtual SurfaceInteraction3f _fill_surface_interaction(const Ray3f &ray,
                                const Float delta,
                                const Float* /*cache*/,
                                          SurfaceInteraction3f si,
                                          Mask active = true) const = 0;

    virtual std::pair<Mask, Float>
    ray_intersect(const Ray3f &ray, Float *cache, Mask active) const override {
        auto [hit, t, u1, u2] = _ray_intersect(ray, 0.0f, cache, active);
        return { hit, t };
    };

    virtual std::tuple<Mask, Float, Float, Float>
    _ray_intersect(const Ray3f &ray, Float delta, Float *cache, Mask active) const;

#if defined(MTS_ENABLE_OPTIX)
    virtual void traverse(TraversalCallback *callback) override;
    virtual void parameters_changed(const std::vector<std::string> &/*keys*/ = {}) override;
#endif

    /// @}
    // =========================================================================

protected:
    SDF(const Properties &);
    virtual ~SDF();

    void initialize_mesh_vertices();

    virtual ScalarFloat max_silhouette_delta() const { return ScalarFloat(1.0f / 20.0f); };

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
    ENOKI_CALL_SUPPORT_METHOD(_ray_intersect)
ENOKI_CALL_SUPPORT_TEMPLATE_END(mitsuba::SDF)

//! @}
// -----------------------------------------------------------------------
