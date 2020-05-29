#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>

#include <mitsuba/render/sdf.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class ExplicitSDF final : public SDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(SDF, m_bbox, m_bsdf, m_to_world, initialize_mesh_vertices)
    MTS_IMPORT_TYPES()

    using typename Base::ScalarSize;

    ExplicitSDF(const Properties &props) : Base(props) {
        if(!props.has_property("distance_field")){
            Throw("ExplicitSDF::ExplicitSDF(props): distance_field property is mandatory!");
        }
        m_distance_field = props.volume<Volume<Float, Spectrum>>("distance_field");
        m_bbox = m_distance_field->bbox();

        initialize_mesh_vertices();
    }

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    Float distance(const Interaction3f &it, Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        const ScalarFloat bbox_step = hmax(m_distance_field->bbox().extents() /
                                                  m_distance_field->resolution());

        Mask inside = m_bbox.contains(it.p);
        Float d = m_distance_field->eval_1(it, active && inside);
        masked(d, active && !inside) = m_bbox.distance(it.p) + bbox_step;
        return d;
    }

    SurfaceInteraction3f _fill_surface_interaction(const Ray3f &ray, const Float delta, const Float * /*cache*/,
                                  SurfaceInteraction3f si, Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        si.p = ray(si.t);

        auto [d, n] = m_distance_field->eval_gradient(si, active);
        Normal3f n_detach = detach(n);

        d = (d - delta) / select(delta == 0.0f, max(dot(ray.d, -n_detach), 0.01f), 1.0f);

        select(delta == 0.0f, n, n_detach);

        si.p[active] = fmadd(si.t + d, ray.d, ray.o);
        si.n[active] = n;

        masked(si.t, active) = norm(si.p - ray.o);

        auto [dp_du, dp_dv] = coordinate_system(n_detach);
        si.dp_du[active] = dp_du;
        si.dp_dv[active] = dp_dv;
        si.sh_frame.n[active] = n;
        si.sh_frame.s[active] = dp_du;
        si.sh_frame.t[active] = dp_dv;

        masked(si.time, active) = ray.time;
        si.wi[active] = si.to_local(-ray.d);
        masked(si.shape, active) = this;
        masked(si.prim_index, active) = 0;

        return si;

    }

    //! @}
    // =============================================================

#if defined(MTS_ENABLE_OPTIX)

    virtual void traverse(TraversalCallback *callback) override {
        m_distance_field->traverse(callback);
        Base::traverse(callback);
    }

    virtual void parameters_changed(const std::vector<std::string> &keys) override {
        Base::parameters_changed(keys);
        m_distance_field->parameters_changed(keys);
    }

#endif

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "ExplicitSDF[" << std::endl
            << "  distance_field = "  << string::indent(m_distance_field) << "," << std::endl
            << "  bbox = " << string::indent(m_bbox) << "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()

    ScalarFloat max_silhouette_delta() const override {
        return hmax(0.2f * m_distance_field->bbox().extents() / m_distance_field->resolution());
    }

private:
    ref<Volume<Float, Spectrum>> m_distance_field;
    //ScalarBoundingBox3f m_bbox;
};

MTS_IMPLEMENT_CLASS_VARIANT(ExplicitSDF, SDF)
MTS_EXPORT_PLUGIN(ExplicitSDF, "Grid3d SDF intersection primitive");
NAMESPACE_END(mitsuba)
