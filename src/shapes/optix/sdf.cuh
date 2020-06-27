#pragma once

#include <math.h>
#include <mitsuba/render/optix/common.h>
#include <mitsuba/render/optix/math.cuh>

struct OptixSdfData {
    optix::BoundingBox3f bbox;
    optix::Transform4f to_world;
    optix::Transform4f to_object;
    float *sdf_data;
    optix::Vector3i resolution;
    optix::Vector3f center;
    float radius;
};

__device__ bool
intersect_aabb(const Vector3f &ray_o, const Vector3f &ray_d, float &mint, float &maxt) {

    /* First, ensure that the ray either has a nonzero slope on each axis,
        or that its origin on a zero-valued axis is within the box bounds */

    bool active = ((ray_d.x() != 0.f) || (ray_o.x() > 0.f) || (ray_o.x() < 1.f)) &&
                  ((ray_d.y() != 0.f) || (ray_o.y() > 0.f) || (ray_o.y() < 1.f)) &&
                  ((ray_d.z() != 0.f) || (ray_o.z() > 0.f) || (ray_o.z() < 1.f));

    // Compute intersection intervals for each axis
    Vector3f d_rcp(1.0f / ray_d.x(), 1.0f/ ray_d.y(), 1.0f / ray_d.z());
    Vector3f t1 = (Vector3f(0.f) - ray_o) * d_rcp,
             t2 = (Vector3f(1.f) - ray_o) * d_rcp;

    // // Ensure proper ordering
    Vector3f t1p = min(t1, t2),
             t2p = max(t1, t2);
    mint = t1p.max();
    maxt = t2p.max();

    // active = active && (maxt >= mint);
    return active;
}

#ifdef __CUDACC__
extern "C" __global__ void __intersection__sdf() {
    const OptixHitGroupData *sbt_data = (OptixHitGroupData*) optixGetSbtDataPointer();
    OptixSdfData *sdf = (OptixSdfData *)sbt_data->data;

    float mint = optixGetRayTmin();
    float maxt = min(optixGetRayTmax(), 10.0f); // TODO: Get intersection with the sdf's AABB
    Vector3f ray_o = make_vector3f(optixGetWorldRayOrigin());
    Vector3f ray_d = make_vector3f(optixGetWorldRayDirection());

    // Transform the ray to the SDFs coordinate system
    ray_o = sdf->to_object.transform_point(ray_o);

    // If the original point is outside the SDF, just terminate
    // if (ray_o.x < 0 || ray_o.x > 1 ||
    //     ray_o.y < 0 || ray_o.y > 1 ||
    //     ray_o.z < 0 || ray_o.z > 1)
    //     return;

    // Intersect the transformed ray with [0,1]



    float t = mint;
    while (true) {
        Vector3f p = fmaf(t, ray_d, ray_o);
        float min_dist = abs(norm(p - sdf->center) - sdf->radius);
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
        OptixSdfData *sphere = (OptixSdfData *)sbt_data->data;

        /* Compute and store information describing the intersection. This is
           very similar to Sphere::fill_surface_interaction() */

        Vector3f ray_o = make_vector3f(optixGetWorldRayOrigin());
        Vector3f ray_d = make_vector3f(optixGetWorldRayDirection());
        float t = optixGetRayTmax();

        Vector3f p = fmaf(t, ray_d, ray_o);
        Vector3f ns = normalize(p - sphere->center); // gradient of the sphere SDF

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