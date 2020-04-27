#include <mitsuba/render/sdf.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/core/plugin.h>

NAMESPACE_BEGIN(mitsuba)

MTS_VARIANT SDF<Float, Spectrum>::SDF(const Properties &props) : Base(props) {
    m_sphere_tracing_steps = props.int_("sphere_tracing_steps", 50);
    if(m_sphere_tracing_steps<=0)
        Throw("sphere_tracing_steps should be positive");
}

MTS_VARIANT SDF<Float, Spectrum>::~SDF() {}

MTS_VARIANT std::pair<typename SDF<Float, Spectrum>::Mask, Float>
SDF<Float, Spectrum>::ray_intersect(const Ray3f &ray, Float * /*cache*/,
                                         Mask active) const {

    // Taken from Keinert, B. et al. (2014). Enhanced Sphere Tracing.

    ScopedPhase sp(ProfilerPhase::RayIntersectSDF);

    auto [valid, mint, maxt] = bbox().ray_intersect(ray);

    Mask originInside = bbox().contains(ray.o);
    masked(mint, originInside) = ray.mint;
    masked(mint, !originInside) += 10*math::RayEpsilon<Float>;

    active &= valid && mint <= ray.maxt && maxt > ray.mint;

    if(none(active))
        return { active, math::Infinity<Float> };

    Interaction3f it(mint, ray.time, ray.wavelengths, ray(mint));
    const ScalarFloat epsilon = math::RayEpsilon<Float>/2;
    Float omega = 1.2;
    Float candidate_error = math::Infinity<Float>;
    Float candidate_t = mint;
    Float previousRadius = 0;
    Float stepLength = 0;
    const Float functionSign = sign(distance(it, active));

    for (int i = 0; i < m_sphere_tracing_steps; ++i) {

        Float signedRadius = functionSign * distance(it, active);
        Float radius = abs(signedRadius);

        Mask sorFail = omega > 1 && (radius + previousRadius) < stepLength;
        masked(stepLength, active) = select(sorFail, stepLength - omega * stepLength, signedRadius * omega);
        masked(omega, active && sorFail) = 1;

        previousRadius = radius;
        Float error = radius / it.t;

        Mask updatable = active && !sorFail && error < candidate_error;
        masked(candidate_t, updatable) = it.t;
        masked(candidate_error, updatable) = error;

        active &= sorFail || (error >= epsilon && it.t <= maxt);

        if (none_or<false>(active))
            break;

        masked(it.t, active) += stepLength;
        masked(it.p, active) = ray(it.t);
    }

    Mask missed = (it.t > ray.maxt || candidate_error > epsilon); // && !forceHit;
    return { !missed, select(!missed, candidate_t, math::Infinity<Float>) };
}

MTS_VARIANT void SDF<Float, Spectrum>::fill_surface_interaction(const Ray3f & /*ray*/,
                                                                  const Float * /*cache*/,
                                                                  SurfaceInteraction3f & /*si*/,
                                                                  Mask /*active*/) const {
    NotImplementedError("fill_surface_interaction");
}

MTS_IMPLEMENT_CLASS_VARIANT(SDF, Shape)
MTS_INSTANTIATE_CLASS(SDF)
NAMESPACE_END(mitsuba)
