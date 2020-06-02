#include <mitsuba/render/sdf.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/core/plugin.h>

NAMESPACE_BEGIN(mitsuba)



MTS_VARIANT SDF<Float, Spectrum>::SDF(const Properties &props) : Base(props) {
    m_sphere_tracing_steps = props.int_("sphere_tracing_steps", 100);
    if(m_sphere_tracing_steps<=0)
        Throw("sphere_tracing_steps should be greater than 0");

    m_mesh = false;
    m_is_sdf = true;
}

MTS_VARIANT SDF<Float, Spectrum>::~SDF() {}

MTS_VARIANT std::tuple<typename SDF<Float, Spectrum>::Mask, Float, Float, Float>
SDF<Float, Spectrum>::_ray_intersect(const Ray3f &ray, Float delta, Float* cache, Mask active) const {
    ENOKI_MARK_USED(cache);
    // Taken from Keinert, B. et al. (2014). Enhanced Sphere Tracing.

    ScopedPhase sp(ProfilerPhase::RayIntersectSDF);

    ScalarFloat sil_delta = max_silhouette_delta();
    const ScalarFloat epsilon = math::RayEpsilon<Float>;

    auto [valid, mint, maxt] = m_bbox.ray_intersect(ray);

    Mask originInside = mint < ray.mint;
    masked(mint, originInside) = ray.mint + 10 * epsilon;
    masked(mint, !originInside) = ray.mint + sil_delta / 10;

    active &= valid && mint <= ray.maxt && maxt > ray.mint;

    if(none_or<false>(active))
        return { active, maxt, math::Infinity<Float>, math::Infinity<Float> };

    Interaction3f it(mint, ray.time, ray.wavelengths, ray(mint));

    Float omega = 1;
    Float candidate_error = math::Infinity<Float>;
    Float candidate_t = mint;
    Float previousRadius = 0;
    Float stepLength = 0;

    Float d = distance(it, active);
    if constexpr(is_diff_array_v<Float>)
        d = detach(d);

    const Float functionSign = sign(d);
    const Mask active_sil = d > sil_delta;

    Float silhouette_dist = math::Infinity<Float>;
    Float silhouette_t = math::Infinity<Float>;

    for (int i = 0; i < m_sphere_tracing_steps; ++i) {

        Float dist = distance(it, active) - delta;
        if constexpr(is_diff_array_v<Float>)
            dist = detach(dist);

        Float signedRadius = functionSign * dist;
        Float radius = abs(signedRadius);

        auto silhouette = active && active_sil && radius < sil_delta && radius > previousRadius && previousRadius < silhouette_dist;
        masked(silhouette_dist, silhouette) = previousRadius;
        masked(silhouette_t, silhouette) = it.t;

        Mask sorFail = omega > 1 && (radius + previousRadius) < stepLength;

        masked(stepLength, active) = select(sorFail, stepLength - omega * stepLength, signedRadius * omega);
        masked(omega, active && sorFail) = 1;

        previousRadius = radius;
        Float error = radius / it.t;

        Mask updatable = active && !sorFail && error < candidate_error;
        masked(candidate_t, updatable) = it.t;
        masked(candidate_error, updatable) = error;

        active &= sorFail || (error >= epsilon && it.t <= maxt);

        if constexpr (is_cuda_array_v<Float>){
            if (i%5==4 && none(active))
                break;
        }else{
            if (none(active))
                break;
        }

        masked(it.t, active) += stepLength;
        masked(it.p, active) = ray(it.t);
    }

    Mask missed = (candidate_t > ray.maxt || candidate_error > epsilon);

    candidate_t = select(!missed, candidate_t, maxt);

    Mask bad_sil = isinf(silhouette_t) || candidate_t - silhouette_t < sil_delta;
    masked(silhouette_dist, bad_sil) = math::Infinity<Float>;
    masked(silhouette_t, bad_sil) = math::Infinity<Float>;

    return { !missed, candidate_t, silhouette_t, silhouette_dist };
}

MTS_VARIANT void SDF<Float, Spectrum>::initialize_mesh_vertices() {

    if constexpr (is_cuda_array_v<Float>) {
        const int NV = 24, NF= 12;

        // position, normal, position, normal, ...
        InputPoint3f vertices[2*NV] = {
            {0, 0, 1}, {-1, 0, 0}, {0, 1, 0}, {-1, 0, 0},
            {0, 0, 0}, {-1, 0, 0}, {0, 1, 1}, {0, 1, 0},
            {1, 1, 0}, {0, 1, 0},  {0, 1, 0}, {0, 1, 0},
            {1, 1, 1}, {1, 0, 0},  {1, 0, 0}, {1, 0, 0},
            {1, 1, 0}, {1, 0, 0},  {1, 0, 1}, {0, -1, 0},
            {0, 0, 0}, {0, -1, 0}, {1, 0, 0}, {0, -1, 0},
            {1, 1, 0}, {0, 0, -1}, {0, 0, 0}, {0, 0, -1},
            {0, 1, 0}, {0, 0, -1}, {0, 1, 1}, {0, 0, 1},
            {1, 0, 1}, {0, 0, 1},  {1, 1, 1}, {0, 0, 1},
            {0, 1, 1}, {-1, 0, 0}, {1, 1, 1}, {0, 1, 0},
            {1, 0, 1}, {1, 0, 0},  {0, 0, 1}, {0, -1, 0},
            {1, 0, 0}, {0, 0, -1}, {0, 0, 1}, {0, 0, 1}
        };

        const std::array<ScalarIndex, 3> faces[NF] = {
            {0,1,2},    {3,4,5},    {6,7,8},    {9,10,11},
            {12,13,14}, {15,16,17}, {0,18,1},   {3,19,4},
            {6,20,7},   {9,21,10},  {12,22,13}, {15,23,16}
        };

        m_vertex_count = NV;
        m_face_count = NF;

        m_vertex_struct = new Struct();
        for (auto name : { "x", "y", "z", "nx", "ny", "nz" })
            m_vertex_struct->append(name, struct_type_v<InputFloat>);
        m_normal_offset = (ScalarIndex) m_vertex_struct->offset("nx");

        m_face_struct = new Struct();
        for (size_t i = 0; i < 3; ++i)
            m_face_struct->append(tfm::format("i%i", i), struct_type_v<ScalarIndex>);

        m_vertex_size = (ScalarSize) m_vertex_struct->size();
        m_face_size   = (ScalarSize) m_face_struct->size();
        m_vertices    = VertexHolder(new uint8_t[(m_vertex_count + 1) * m_vertex_size]);
        m_faces       = FaceHolder(new uint8_t[(m_face_count + 1) * m_face_size]);

        m_to_world = ScalarTransform4f::translate(m_bbox.min) * ScalarTransform4f::scale(ScalarVector3f(m_bbox.max - m_bbox.min));

        uint8_t* vs = m_vertices.get();
        for(uint i = 0; i < m_vertex_count; i++){
            auto v = m_to_world.transform_affine(vertices[2*i]);
            store_unaligned(vs + (i * m_vertex_size), v);
            auto n = vertices[2*i + 1];
            store_unaligned(vs + (i * m_vertex_size + m_normal_offset), n);
        }

        memcpy(m_faces.get(), faces, m_face_count * m_face_size);

    } else {
        m_face_count = 1;
    }
}

MTS_VARIANT typename SDF<Float, Spectrum>::ScalarBoundingBox3f
SDF<Float, Spectrum>::bbox(ScalarIndex index) const {
    if constexpr( is_cuda_array_v<Float> )
        return Base::bbox(index);
    else
        return bbox();
}

MTS_VARIANT typename SDF<Float, Spectrum>::ScalarBoundingBox3f
SDF<Float, Spectrum>::bbox(ScalarIndex index, const ScalarBoundingBox3f &clip) const {
    if constexpr( is_cuda_array_v<Float> ){
        return Base::bbox(index, clip);
    } else {
        ScalarBoundingBox3f result = bbox(index);
        result.clip(clip);
        return result;
    }
}

#if defined(MTS_ENABLE_OPTIX)
MTS_VARIANT void SDF<Float, Spectrum>::traverse(TraversalCallback * callback) {
    Base::traverse(callback);
}

MTS_VARIANT void SDF<Float, Spectrum>::parameters_changed(const std::vector<std::string> &keys) {
    Base::parameters_changed(keys);
}
#endif

MTS_IMPLEMENT_CLASS_VARIANT(SDF, Object, "sdf")
MTS_INSTANTIATE_CLASS(SDF)
NAMESPACE_END(mitsuba)
