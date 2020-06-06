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

    SDFPathIntegrator(const Properties &props) : Base(props) {
        m_sdf_emitter_samples = props.int_("sdf_emitter_samples", 1);
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        RayDifferential3f ray = ray_;

        // Tracks radiance scaling due to index of refraction changes
        Float eta(1.f);

        // MIS weight for intersected emitters (set by prev. iteration)
        Float emission_weight(1.f);

        Spectrum throughput(1.f), result(0.f), result_sil(0.0f);

        // ---------------------- First intersection ----------------------
        SurfaceInteraction3f si = scene->ray_intersect(ray, active);
        Mask valid_ray = si.is_valid();
        EmitterPtr emitter = si.emitter(scene);

        const ScalarFloat sil_angle = tan(0.5f * math::Pi<ScalarFloat> / 180.0f);
        //const ScalarFloat delta = 1.0f / (4.0f * 16); // <-- FIXME

        for (int depth = 1;; ++depth) {

            // auto [silhouette_result, sdf_d, silhouette_hit] = sample_sdf_silhouette(scene, sampler, ray, delta, si, active);
            // result[silhouette_hit] += emission_weight * throughput * silhouette_result;

            // ---------------- Intersection with emitters ----------------

            auto hit_emitter = active && neq(emitter, nullptr);
            if (any_or<true>(hit_emitter))
                result[hit_emitter] += emission_weight * throughput * emitter->eval(si, hit_emitter);

            auto [silhouette_result, sdf_d, silhouette_hit] = sample_sdf_silhouette(scene, sampler, ray, sil_angle, si, active);
            auto weight = select(hit_emitter, emission_weight, 1.0f);
            result_sil[silhouette_hit] += weight * throughput * silhouette_result;
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
                ((!is_cuda_array_v<Float> || m_max_depth < 0) && none(active)))
                break;

            // --------------------- Emitter sampling ---------------------

            BSDFContext ctx;
            BSDFPtr bsdf = si.bsdf(ray);
            Mask active_e = active && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            if (likely(any_or<true>(active_e))) {
                auto [ds, emitter_val] = scene->sample_emitter_direction(
                    si, sampler->next_2d(active_e), false, active_e); // <-- false because we want to detect SDF silhouette
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
                result[active_e] += mis * throughput * bsdf_val * emitter_val;

                auto [silhouette_result, sdf_d, silhouette_hit] = sample_sdf_silhouette(scene, sampler, ray_e, sil_angle, si_e, active_e);
                result_sil[silhouette_hit] += mis * throughput * silhouette_result;
            }

            // ----------------------- BSDF sampling ----------------------

            // Sample BSDF * cos(theta)
            auto [bs, bsdf_val] = bsdf->sample(ctx, si, sampler->next_1d(active),
                                               sampler->next_2d(active), active);
            bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

            throughput = throughput * bsdf_val;
            active &= any(neq(depolarize(throughput), 0.f));
            if (none_or<false>(active))
                break;

            eta *= bs.eta;

            // Intersect the BSDF ray against the scene geometry
            ray = si.spawn_ray(si.to_world(bs.wo));
            SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray, active);

            /* Determine probability of having sampled that same
               direction using emitter sampling. */
            emitter = si_bsdf.emitter(scene, active);
            DirectionSample3f ds(si_bsdf, si);
            ds.object = emitter;

            if (any_or<true>(neq(emitter, nullptr))) {
                Float emitter_pdf =
                    select(neq(emitter, nullptr) && !has_flag(bs.sampled_type, BSDFFlags::Delta),
                           scene->pdf_emitter_direction(si, ds), 0.f);

                emission_weight = mis_weight(bs.pdf, emitter_pdf);
            }

            si = std::move(si_bsdf);
        }

        //return { result, valid_ray };
        //return { result_sil, valid_ray };
        return { result + result_sil, valid_ray };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("SDFPathIntegrator[\n"
            "  max_depth = %i,\n"
            "  rr_depth = %i\n"
            "]", m_max_depth, m_rr_depth);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

private:
    ScalarInt32 m_sdf_emitter_samples;

    std::tuple<Spectrum, Float, Mask> sample_sdf_silhouette(const Scene* scene, Sampler* sampler, const Ray3f& ray_,
                                                    const ScalarFloat sil_tan_angle, const SurfaceInteraction3f& si_, Mask active) const {
        //
        SurfaceInteraction3f si(si_);
        Ray3f ray(ray_);
        SDFPtr sdf = (SDFPtr) si.sdf;

        auto active_sil = active && neq(sdf, nullptr);
        const auto delta = select(active_sil, sil_tan_angle * si.sdf_t, 0.0f);
        active_sil &= si.sdf_d <= delta;

        // Log(Warn, "%s, %s %s", count(active_sil), count(si.sdf_d <= delta), hsum(si.sdf_d) / 131072);
        //ray.d = rotation_matrix.transtorm_affine(nomralize(si.p - ray.o))
        ray.o = ray(si.sdf_t - delta);
        ray.mint = 0.0f;
        ray.maxt -= si.sdf_t - delta;
        auto [hit, t, u1, u2] = sdf->_ray_intersect(ray, delta, nullptr, active_sil && neq(sdf, nullptr));
        si.t = t;
        si = sdf->_fill_surface_interaction(ray, delta, nullptr, si, hit);

        Point3f sdf_true_p = si.p - delta * si.n;
        auto rotation = rotation_matrix(ray, sdf_true_p);
        Vector3f dir = normalize(si.p - ray.o);
        si.wi[active_sil] = -si.to_local(rotation.transform_affine(detach(dir)));

        BSDFContext ctx;
        BSDFPtr bsdf = si.bsdf(ray);
        Mask active_e = active_sil && has_flag(bsdf->flags(), BSDFFlags::Smooth);

        Spectrum result(0.0f);
        Mask valid = false;

        if (likely(any_or<true>(active_e))) {
            for(int i = 0; i < m_sdf_emitter_samples; ++i){
                auto [ds, emitter_val] = scene->sample_emitter_direction(
                    si, sampler->next_2d(active_e), false, active_e);
                active_e &= neq(ds.pdf, 0.f);
                valid |= active_e;

                // Query the BSDF for that emitter-sampled direction
                Vector3f wo = si.to_local(ds.d);
                Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Determine density of sampling that same direction using BSDF sampling
                Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active_e);

                Float mis = select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));
                masked(result, active_e) += mis * bsdf_val * emitter_val;
            }
        }

        return { result / m_sdf_emitter_samples, si.sdf_d, active_e };
    }

    Transform<Vector4f> rotation_matrix(const Ray3f& ray, Point3f true_sdf) const {
        Vector3f sdf_dir = normalize(true_sdf - ray.o);
        // Vector3f axis = cross(ray.d, sdf_dir); // TODO check if args order is correct
        // Float cosangle = dot(sdf_dir, ray.d);  // TODO check if args order is correct
        Vector3f axis = cross(sdf_dir, ray.d); // TODO check if args order is correct
        Float cosangle = dot(ray.d, sdf_dir);  // TODO check if args order is correct

        Float ax = axis.x(),
              ay = axis.y(),
              az = axis.z();
        Float axy = ax * ay,
              axz = ax * az,
              ayz = ay * az;

        Matrix3f ux(0.f, -az,  ay,
                     az, 0.f, -ax,
                    -ay,  ax, 0.f);

        Matrix3f uu(sqr(ax),     axy,    axz,
                        axy, sqr(ay),    ayz,
                        axz,     ayz, sqr(az));

        Matrix3f R = identity<Matrix3f>() * cosangle + ux + rcp(1 + cosangle) * uu;

        return Transform<Vector4f>(Matrix4f(R));
    };

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(SDFPathIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(SDFPathIntegrator, "Path Tracer integrator with SDF boundary handling");
NAMESPACE_END(mitsuba)