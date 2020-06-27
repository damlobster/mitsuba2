#pragma once

#include <math.h>
#include <mitsuba/render/optix/common.h>
#include <mitsuba/render/optix/math.cuh>

struct OptixSphereSdfData {
    optix::BoundingBox3f bbox;
    optix::Transform4f to_world;
    optix::Transform4f to_object;
    optix::Vector3f center;
    float radius;
    bool flip_normals;
};

#ifdef __CUDACC__
extern "C" __global__ void __intersection__spheresdf() {
    const OptixHitGroupData *sbt_data = (OptixHitGroupData*) optixGetSbtDataPointer();
    OptixSdfData *sphere = (OptixSdfData *)sbt_data->data;

    float mint = optixGetRayTmin();
    float maxt = min(optixGetRayTmax(), 10.0f); // TODO: Get intersection with the sphere's AABB
    Vector3f ray_o = make_vector3f(optixGetWorldRayOrigin());
    Vector3f ray_d = make_vector3f(optixGetWorldRayDirection());

    float t = mint;
    while (true) {
        Vector3f p = fmaf(t, ray_d, ray_o);
        float min_dist = abs(norm(p - sphere->center) - sphere->radius);
        t += min_dist;

        if (t > maxt)
            return;
        if (min_dist < 1e-6f) {
            optixReportIntersection(t, OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE);
            return;
        }
    }
}


extern "C" __global__ void __closesthit__spheresdf() {
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