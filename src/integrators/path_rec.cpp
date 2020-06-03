#include <random>
#include <enoki/stl.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/sdf.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-path:

Path tracer (:monosp:`path`)
-------------------------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1 corresponds to
     :math:`\infty`). A value of 1 will only render directly visible light sources. 2 will lead
     to single-bounce (direct-only) illumination, and so on. (Default: -1)
 * - rr_depth
   - |int|
   - Specifies the minimum path depth, after which the implementation will start to use the
     *russian roulette* path termination criterion. (Default: 5)
 * - hide_emitters
   - |bool|
   - Hide directly visible emitters. (Default: no, i.e. |false|)

This integrator implements a basic path tracer and is a **good default choice**
when there is no strong reason to prefer another method.

To use the path tracer appropriately, it is instructive to know roughly how
it works: its main operation is to trace many light paths using *random walks*
starting from the sensor. A single random walk is shown below, which entails
casting a ray associated with a pixel in the output image and searching for
the first visible intersection. A new direction is then chosen at the intersection,
and the ray-casting step repeats over and over again (until one of several
stopping criteria applies).

.. image:: ../images/integrator_path_figure.png
    :width: 95%
    :align: center

At every intersection, the path tracer tries to create a connection to
the light source in an attempt to find a *complete* path along which
light can flow from the emitter to the sensor. This of course only works
when there is no occluding object between the intersection and the emitter.

This directly translates into a category of scenes where
a path tracer can be expected to produce reasonable results: this is the case
when the emitters are easily "accessible" by the contents of the scene. For instance,
an interior scene that is lit by an area light will be considerably harder
to render when this area light is inside a glass enclosure (which
effectively counts as an occluder).

Like the :ref:`direct <integrator-direct>` plugin, the path tracer internally relies on multiple importance
sampling to combine BSDF and emitter samples. The main difference in comparison
to the former plugin is that it considers light paths of arbitrary length to compute
both direct and indirect illumination.

.. _sec-path-strictnormals:

.. Commented out for now
.. Strict normals
   --------------

.. Triangle meshes often rely on interpolated shading normals
   to suppress the inherently faceted appearance of the underlying geometry. These
   "fake" normals are not without problems, however. They can lead to paradoxical
   situations where a light ray impinges on an object from a direction that is
   classified as "outside" according to the shading normal, and "inside" according
   to the true geometric normal.

.. The :paramtype:`strict_normals` parameter specifies the intended behavior when such cases arise. The
   default (|false|, i.e. "carry on") gives precedence to information given by the shading normal and
   considers such light paths to be valid. This can theoretically cause light "leaks" through
   boundaries, but it is not much of a problem in practice.

.. When set to |true|, the path tracer detects inconsistencies and ignores these paths. When objects
   are poorly tesselated, this latter option may cause them to lose a significant amount of the
   incident radiation (or, in other words, they will look dark).

.. note:: This integrator does not handle participating media

 */

template <typename Float, typename Spectrum>
class SDFPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr, SDF, SDFPtr)

    SDFPathIntegrator(const Properties &props) : Base(props) {}

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        // ---------------------- First intersection ----------------------
        SurfaceInteraction3f si = scene->ray_intersect(ray_, active);
        Mask valid_ray = si.is_valid();

        auto [result, result_sil] = sample_rec<is_diff_array_v<Float>>(1, scene, sampler, ray_, si, 1.0f, 1.0f, valid_ray);

        EmitterPtr emitter = si.emitter(scene);
        if (any_or<true>(neq(emitter, nullptr)))
            result[valid_ray] += emitter->eval(si, valid_ray);

        if constexpr(is_diff_array_v<Float>){
            auto [silhouette_result, sdf_d, silhouette_hit] = sample_silhouette(1, scene, sampler, ray_, si, 1.0f, 1.0f, active);
            result_sil[silhouette_hit] += (silhouette_result - detach(result)) * grad_weight(sdf_d);
        }

        return { result + result_sil, valid_ray };
        // return { result, valid_ray };
        // return { result_sil, valid_ray };
    }

    template<bool silhouette_enabled>
    std::pair<Spectrum, Spectrum> sample_rec(const int depth,
                        const Scene *scene,
                        Sampler *sampler,
                        const Ray3f &ray_,
                        const SurfaceInteraction3f &si_,
                        Spectrum throughput,
                        Float eta,
                        Mask active) const {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        constexpr bool sil_enabled = is_diff_array_v<Float> && silhouette_enabled;

        Ray3f ray = ray_;
        SurfaceInteraction3f si = si_;
        Spectrum result(0.f), result_sil(0.0f);

        active &= si.is_valid();

        /* Russian roulette: try to keep path weights equal to one,
            while accounting for the solid angle compression at refractive
            index boundaries. Stop with at least some probability to avoid
            getting stuck (e.g. due to total internal reflection) */
        if (depth > m_rr_depth) {
            Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
            active &= sampler->next_1d(active) < q;
            throughput *= rcp(q);
        }

        // Stop if we've exceeded the number of requested bounces, or
        // if there are no more active lanes. Only do this latter check
        // in GPU mode when the number of requested bounces is infinite
        // since it causes a costly synchronization.
        if ((uint32_t) depth >= (uint32_t) m_max_depth ||
            ((!is_cuda_array_v<Float> || m_max_depth < 0) && none(active))){

            return { 0.0f, 0.0f };
        }

        // --------------------- Emitter sampling ---------------------

        BSDFContext ctx;
        BSDFPtr bsdf = si.bsdf(ray);
        Mask active_e = active && has_flag(bsdf->flags(), BSDFFlags::Smooth);

        if (likely(any_or<true>(active_e))) {
            auto [ds, emitter_val] = scene->sample_emitter_direction(
                si, sampler->next_2d(active_e), false, active_e);
            active_e &= neq(ds.pdf, 0.f);

            // manually test emitter visibility to be able to detect silhouette
            Ray3f ray_e(si.p, ds.d, math::RayEpsilon<Float> * (1.f + hmax(abs(si.p))),
                    ds.dist * (1.f - math::ShadowEpsilon<Float>), si.time, si.wavelengths);
            SurfaceInteraction3f si_e = scene->ray_intersect(ray_e, active_e);
            emitter_val[si_e.t < ray_e.maxt] = 0.0f;

            // Query the BSDF for that emitter-sampled direction
            Vector3f wo = si.to_local(ds.d);
            Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

            // Determine density of sampling that same direction using BSDF sampling
            Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active_e);

            Float mis = select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));
            result[active_e] += mis * bsdf_val * emitter_val;

            if constexpr(sil_enabled){
                auto [silhouette_result, sdf_d, silhouette_hit] = sample_silhouette(depth, scene, sampler, ray_e, si_e, throughput, eta, active_e && neq(si_e.sdf, nullptr));
                result_sil[silhouette_hit] += mis * bsdf_val * (silhouette_result - detach(emitter_val)) * grad_weight(sdf_d);
            }
        }

        // ----------------------- BSDF sampling ----------------------

        // Sample BSDF * cos(theta)
        auto [bs, bsdf_val] = bsdf->sample(ctx, si, sampler->next_1d(active),
                                            sampler->next_2d(active), active);
        bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

        throughput = throughput * bsdf_val;
        active &= any(neq(depolarize(throughput), 0.f));
        if (none_or<false>(active))
            return { result, result_sil };

        eta *= bs.eta;

        // Intersect the BSDF ray against the scene geometry
        ray = si.spawn_ray(si.to_world(bs.wo));
        SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray, active);

        /* Determine probability of having sampled that same
            direction using emitter sampling. */
        EmitterPtr emitter = si_bsdf.emitter(scene, active);
        DirectionSample3f ds(si_bsdf, si);
        ds.object = emitter;

        Spectrum res_bsdf(0.0f);
        if (any_or<true>(neq(emitter, nullptr))) {
            Float emitter_pdf =
                select(neq(emitter, nullptr) && !has_flag(bs.sampled_type, BSDFFlags::Delta),
                        scene->pdf_emitter_direction(si, ds),
                        0.f);

            Float emission_weight = mis_weight(bs.pdf, emitter_pdf);
            masked(res_bsdf, active) += emission_weight * emitter->eval(si_bsdf, active);
        }

        auto [res, res_sil] = sample_rec<sil_enabled>(depth+1, scene, sampler, ray, si_bsdf, throughput, eta, active);
        masked(res_bsdf, active) += res;

        if constexpr(sil_enabled){
            auto [silhouette_result, sdf_d, silhouette_hit] = sample_silhouette(depth, scene, sampler, ray, si, throughput, eta, active);
            res_sil[silhouette_hit] += (silhouette_result - detach(res_bsdf)) * grad_weight(sdf_d);
        }

        result += bsdf_val * res_bsdf;
        result_sil += bsdf_val * res_sil;
        return std::pair(result, result_sil);
    }

    std::tuple<Spectrum, Float, Mask> sample_silhouette(const int depth,
                                             const Scene *scene,
                                             Sampler *sampler,
                                             const Ray3f &ray,
                                             const SurfaceInteraction3f &si_,
                                             Spectrum throughput,
                                             Float eta,
                                             Mask active) const {
        if constexpr(!is_diff_array_v<Float>)
            Throw("Not implemented");

        SurfaceInteraction3f si = si_;

        SDFPtr sdf = (SDFPtr) si.sdf;
        auto active_sil = active && neq(sdf, nullptr);

        Float delta = select(active_sil, sdf->max_silhouette_delta(), 0.0f);
        active_sil &= si.sdf_d <= delta;
        delta = si.sdf_d + 10.0f * math::RayEpsilon<ScalarFloat>;

        auto [hit, t, u1, u2] = sdf->_ray_intersect(ray, delta, nullptr, active_sil);
        masked(si.t, hit) = t;
        si = sdf->_fill_surface_interaction(ray, delta, nullptr, si, hit);

        auto [result, unused] = sample_rec<false>(depth, scene, sampler, ray, si, throughput, eta, hit);
        Float dist = sdf->distance(si, hit);
        return { detach(result), dist, hit };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("PathIntegrator[\n"
            "  max_depth = %i,\n"
            "  rr_depth = %i\n"
            "]", m_max_depth, m_rr_depth);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    MTS_DECLARE_CLASS()

protected:
    inline Float grad_weight(Float sdf_d) const{
        if constexpr(is_diff_array_v<Float>){
            Float d_detach = detach(sdf_d);
            return -sdf_d / d_detach;
            // return d_detach / sdf_d;
        } else {
            return 0.0f;
        }
    }

};

MTS_IMPLEMENT_CLASS_VARIANT(SDFPathIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(SDFPathIntegrator, "SDF Path Tracer integrator");
NAMESPACE_END(mitsuba)

ENOKI_CALL_SUPPORT_TEMPLATE_BEGIN(mitsuba::SDFPathIntegrator)
    ENOKI_CALL_SUPPORT_METHOD(sample_rec)
    ENOKI_CALL_SUPPORT_METHOD(sample_silhouette)
ENOKI_CALL_SUPPORT_TEMPLATE_END(mitsuba::SDFPathIntegrator)