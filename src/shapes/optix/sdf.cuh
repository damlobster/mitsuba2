#pragma once

#include <math.h>
#include <mitsuba/render/optix/common.h>
#include <mitsuba/render/optix/math.cuh>

struct OptixSdfData {
    optix::BoundingBox3f bbox;
    optix::Transform4f to_world;
    optix::Transform4f to_object;
    const float *sdf_data = nullptr;
    optix::Vector3i resolution;
    cudaTextureObject_t sdf_texture;
};

#ifdef __CUDACC__

__device__ bool
intersect_aabb(const Vector3f &ray_o, 
               const Vector3f &ray_d, 
               float &mint, 
               float &maxt) {

    /* First, ensure that the ray either has a nonzero slope on each axis,
        or that its origin on a zero-valued axis is within the box bounds */

    bool active = ((ray_d.x() != 0.f) || (ray_o.x() > 0.f) || (ray_o.x() < 1.f)) &&
                  ((ray_d.y() != 0.f) || (ray_o.y() > 0.f) || (ray_o.y() < 1.f)) &&
                  ((ray_d.z() != 0.f) || (ray_o.z() > 0.f) || (ray_o.z() < 1.f));

    // Compute intersection intervals for each axis
    Vector3f d_rcp(1.0f / ray_d.x(), 1.0f/ ray_d.y(), 1.0f / ray_d.z());
    Vector3f t1 = (Vector3f(0.f) - ray_o) * d_rcp,
             t2 = (Vector3f(1.f) - ray_o) * d_rcp;

    // Ensure proper ordering
    Vector3f t1p = min(t1, t2),
             t2p = max(t1, t2);
    mint = t1p.max();
    maxt = t2p.max();

    active = active && (maxt >= mint);
    return active;
}

__device__ float eval_sdf(const Vector3f &p, const float *sdf, const Vector3i &res) {
    Vector3f p_scaled((res.x() - 1) * p.x(), (res.y() - 1) * p.y(), (res.z() - 1) * p.z());
    Vector3i pi(p_scaled.x(), p_scaled.y(), p_scaled.z());
    pi = max(Vector3i(0,0,0), min(pi, res - 1));

    // If we are trying to evaluate a point outside the SDF: just assume we completely missed the shape for now
    if ((pi.x() < 0) || (pi.y() < 0) || (pi.z() < 0) || 
        (pi.x() + 1 >= res.x()) || (pi.y() + 1 >= res.y()) || (pi.z() + 1 >= res.z())) {
            return 1000.f;
    }

    Vector3f f = p_scaled - Vector3f(pi.x(), pi.y(), pi.z());
    Vector3f rf = Vector3f(1.f, 1.f, 1.f) - f;

    unsigned int index = fmaf(fmaf(pi.z(), res.y(), pi.y()), res.x(), pi.x());
    unsigned int z_offset = res.x() * res.y();
    float v000 = sdf[index],
          v001 = sdf[index + 1],
          v010 = sdf[index + res.x()],
          v011 = sdf[index + res.x() + 1],
          v100 = sdf[index + z_offset],
          v101 = sdf[index + z_offset + 1],
          v110 = sdf[index + z_offset + res.x()],
          v111 = sdf[index + z_offset + res.x() + 1];

    float v00 = fmaf(v000, rf.x(), v001 * f.x()),
          v01 = fmaf(v010, rf.x(), v011 * f.x()),
          v10 = fmaf(v100, rf.x(), v101 * f.x()),
          v11 = fmaf(v110, rf.x(), v111 * f.x());
    float v0  = fmaf(v00, rf.y(), v01 * f.y()),
          v1  = fmaf(v10, rf.y(), v11 * f.y());
    float result = fmaf(v0, rf.z(), v1 * f.z());
    return result;
}

// __device__ bspline_lookup(const Vector3f &p, const cudaTextureObject_t &texture) {

// }


extern "C" __global__ void __intersection__sdf() {
    const OptixHitGroupData *sbt_data = (OptixHitGroupData*) optixGetSbtDataPointer();
    OptixSdfData *sdf = (OptixSdfData *)sbt_data->data;

    float mint = optixGetRayTmin();
    float maxt = min(optixGetRayTmax(), 10.0f); // TODO: Get intersection with the sdf's AABB
    Vector3f ray_o = make_vector3f(optixGetWorldRayOrigin());
    Vector3f ray_d = make_vector3f(optixGetWorldRayDirection());

    // Transform the ray to the SDFs coordinate system
    ray_o = sdf->to_object.transform_point(ray_o);

    // Intersect the transformed ray with the SDFs bounding box [0,1]
    float aabb_mint, aabb_maxt;
    bool intersects = intersect_aabb(ray_o, ray_d, aabb_mint, aabb_maxt);
    if (!intersects)
        return; // TODO: This should actually never happen, right? 

    Vector3i res = sdf->resolution;

    float t = aabb_mint > 0 ? aabb_mint : aabb_maxt;
    t = max(t, mint);
    t += 1e-5; // Small ray epsilon to always query position inside the grid

    while (true) {
        Vector3f p = fmaf(t, ray_d, ray_o);
        // float min_dist1 = eval_sdf(p, sdf->sdf_data, res);
        float min_dist = tex3D<float>(sdf->sdf_texture, p.x() * res.x(), p.y() * res.y(), p.z() * res.z());
        t += min_dist;

        if (t > maxt)
            return;
        if (min_dist < 1e-6f) {
            optixReportIntersection(t, OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE);
            return;
        }
    }
}


extern "C" __global__ void __closesthit__sdf() {
    unsigned int launch_index = calculate_launch_index();

    if (params.out_hit != nullptr) {
        params.out_hit[launch_index] = true;
    } else {
        const OptixHitGroupData *sbt_data = (OptixHitGroupData *) optixGetSbtDataPointer();
        OptixSdfData *sdf = (OptixSdfData *)sbt_data->data;

        /* Compute and store information describing the intersection. This is
           very similar to Sphere::fill_surface_interaction() */

        Vector3f ray_o = make_vector3f(optixGetWorldRayOrigin());
        Vector3f ray_d = make_vector3f(optixGetWorldRayDirection());
        float t = optixGetRayTmax();
        Vector3i res = sdf->resolution;

        Vector3f p = fmaf(t, ray_d, ray_o);
        Vector3f local_p = sdf->to_object.transform_point(p);

        float eps = 0.001f;
        float v0 = eval_sdf(local_p, sdf->sdf_data, res);
        float v0x = eval_sdf(local_p + Vector3f(eps, 0, 0), sdf->sdf_data, res);
        float v0y = eval_sdf(local_p + Vector3f(0, eps, 0), sdf->sdf_data, res);
        float v0z = eval_sdf(local_p + Vector3f(0, 0, eps), sdf->sdf_data, res);  
        
        Vector3f grad(v0x - v0, v0y - v0, v0z - v0);
        Vector3f ns = normalize(grad);

        Vector2f uv = Vector2f(0.f, 0.f);
        Vector3f dp_du = Vector3f(0.f, 0.f, 0.f);
        Vector3f dp_dv = Vector3f(0.f, 0.f, 0.f);
        Vector3f ng = ns;
        write_output_params(params, launch_index,
                            sbt_data->shape_ptr,
                            optixGetPrimitiveIndex(),
                            p, uv, ns, ng, dp_du, dp_dv, t);
    }
}
#endif
