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

// This function implements a trilinear lookup manually,
// this is most likely slower than calling tex3D
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

__forceinline__ __device__  float lerp(float v0, float v1, float w) {
    return fmaf(w, v1 - v0, v0);
}

__forceinline__ __device__ Vector3f offsets(float coord, unsigned int res) {
    float coord_hg = coord * res - 0.5f;
    float int_part = floorf(coord_hg);
    float tc = int_part + 0.5f;
    float a = coord_hg - int_part;

    float a2 = a * a;
    float a3 = a2 * a;
    float norm_fac = 1.0f / 6.0f;

    float w0 = norm_fac * fmaf(a, fmaf(a, -a + 3, - 3), 1);
    float w1 = norm_fac * fmaf(3 * a2, a - 2, 4);    
    float w2 = norm_fac * fmaf(a, 3 * (-a2 + a + 1), 1);
    float w3 = norm_fac * a3;

    // Original less optimized implementation
    // float w0 = norm_fac * (    -a3 + 3 * a2 - 3 * a + 1);
    // float w1 = norm_fac * ( 3 * a3 - 6 * a2         + 4);
    // float w2 = norm_fac * (-3 * a3 + 3 * a2 + 3 * a + 1);
    // float w3 = norm_fac * a3;

    float g0 = w0 + w1;
    float inv_res = 1.f / res;
    float h0 = inv_res * (tc - 1 + w1 / (w0 + w1));
    float h1 = inv_res * (tc + 1 + w3 / (w2 + w3));
    return Vector3f(h0, h1, g0);
}


// Implements a bspline interpolated lookup into a voxel grid
// The implementation follows https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering
// (without precomputing the weights as a texture for now)
__device__ float bspline_lookup(const Vector3f &p, const cudaTextureObject_t &texture, const Vector3i &res) {

    // Compute the various lookup parameters
    // This code could be vectorized across dimensions 
    Vector3f hg_x = offsets(p.x(), res.x());
    Vector3f hg_y = offsets(p.y(), res.y());
    Vector3f hg_z = offsets(p.z(), res.z());

    // fetch eight linearly interpolated inputs at the different locations computed earlier
    float tex000 = tex3D<float>(texture, hg_x.x(), hg_y.x(), hg_z.x());
    float tex100 = tex3D<float>(texture, hg_x.y(), hg_y.x(), hg_z.x());
    float tex010 = tex3D<float>(texture, hg_x.x(), hg_y.y(), hg_z.x());
    float tex110 = tex3D<float>(texture, hg_x.y(), hg_y.y(), hg_z.x());
    float tex001 = tex3D<float>(texture, hg_x.x(), hg_y.x(), hg_z.y());
    float tex101 = tex3D<float>(texture, hg_x.y(), hg_y.x(), hg_z.y());
    float tex011 = tex3D<float>(texture, hg_x.x(), hg_y.y(), hg_z.y());
    float tex111 = tex3D<float>(texture, hg_x.y(), hg_y.y(), hg_z.y());

    // Interpolate all these results using the precomputed weights
    float tex00 = lerp(tex001, tex000, hg_z.z());
    float tex01 = lerp(tex011, tex010, hg_z.z());
    float tex10 = lerp(tex101, tex100, hg_z.z());
    float tex11 = lerp(tex111, tex110, hg_z.z());
    float tex0  = lerp(tex01, tex00, hg_y.z());
    float tex1  = lerp(tex11, tex10, hg_y.z());
    return lerp(tex1, tex0, hg_x.z());
}


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

    maxt = min(maxt, aabb_maxt);
    float t = aabb_mint > 0 ? aabb_mint : aabb_maxt;
    t = max(t, mint);
    t += 1e-5; // Small ray epsilon to always query position inside the grid

    // while (true) {
    for (int i =0; i < 4096; ++i) {
        Vector3f p = fmaf(t, ray_d, ray_o);
        // float min_dist1 = eval_sdf(p, sdf->sdf_data, res);
        // float min_dist = tex3D<float>(sdf->sdf_texture, p.x(), p.y(), p.z());
        float min_dist = bspline_lookup(p, sdf->sdf_texture, res);
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

        float eps = 0.005f;

        float v0 = bspline_lookup(local_p, sdf->sdf_texture, res);
        float v0x = bspline_lookup(local_p + Vector3f(eps, 0, 0), sdf->sdf_texture, res);
        float v0y = bspline_lookup(local_p + Vector3f(0, eps, 0), sdf->sdf_texture, res);
        float v0z = bspline_lookup(local_p + Vector3f(0, 0, eps), sdf->sdf_texture, res);  

        
        // float v0 = tex3D<float>(sdf->sdf_texture, local_p.x(), local_p.y(), local_p.z());
        // float v0x = tex3D<float>(sdf->sdf_texture, local_p.x() + eps, local_p.y(), local_p.z());
        // float v0y = tex3D<float>(sdf->sdf_texture, local_p.x(), local_p.y() + eps, local_p.z());
        // float v0z = tex3D<float>(sdf->sdf_texture, local_p.x(), local_p.y(), local_p.z() + eps);

        // float v0 = eval_sdf(local_p, sdf->sdf_data, res);
        // float v0x = eval_sdf(local_p + Vector3f(eps, 0, 0), sdf->sdf_data, res);
        // float v0y = eval_sdf(local_p + Vector3f(0, eps, 0), sdf->sdf_data, res);
        // float v0z = eval_sdf(local_p + Vector3f(0, 0, eps), sdf->sdf_data, res);  
        
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
