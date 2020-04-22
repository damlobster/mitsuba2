#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>

#include <mitsuba/render/sdf.h>

#if defined(MTS_ENABLE_EMBREE)
    #include <embree3/rtcore.h>
#endif

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class ExplicitSDF final : public SDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(SDF, m_bbox)
    MTS_IMPORT_TYPES()

    using typename Base::ScalarSize;

    ExplicitSDF(const Properties &props) : Base(props) {
        if(!props.has_property("distance_field")){
            Throw("ExplicitSDF::ExplicitSDF(props): distance_field property is mandatory!");
        }
        m_distance_field = props.volume<Volume<Float, Spectrum>>("distance_field");
        m_bbox = m_distance_field->bbox();
        m_bbox.min -= 2*math::RayEpsilon<Float>;
        m_bbox.max += 2*math::RayEpsilon<Float>;
    }

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    Float distance(const Interaction3f &it, Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        Mask inside = m_bbox.contains(it.p);
        auto d = m_distance_field->eval_1(it, active && inside);
        masked(d, active && !inside) = m_bbox.distance(it.p);
        return d;
    }

    void fill_surface_interaction(const Ray3f &ray, const Float * /*cache*/,
                                  SurfaceInteraction3f &si_out, Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        SurfaceInteraction3f si(si_out);

        si.p = ray(si.t);

        auto [d, n] = m_distance_field->eval_gradient(si, active);

        si.p = fmadd(ray.d, d, si.p);

        si.sh_frame.n = n;
        auto [dp_du, dp_dv] = coordinate_system(si.sh_frame.n);
        si.dp_du = dp_du;
        si.dp_dv = dp_dv;

        si.n = si.sh_frame.n;
        si.time = ray.time;

        si.wi = select(active, si.to_local(-ray.d), -ray.d);

        si_out[active] = si;
    }

    //! @}
    // =============================================================

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("distance_field", m_distance_field);
        Base::traverse(callback);
    }

    void parameters_changed() override {
        Base::parameters_changed();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "ExplicitSDF[" << std::endl
            << "  distance_field = "  << string::indent(m_distance_field) << "," << std::endl
            << "  bbox = " << string::indent(m_bbox) << "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Volume<Float, Spectrum>> m_distance_field;
};

MTS_IMPLEMENT_CLASS_VARIANT(ExplicitSDF, Shape)
MTS_EXPORT_PLUGIN(ExplicitSDF, "Grid3d SDF intersection primitive");
NAMESPACE_END(mitsuba)
