#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include "ray.cuh"

struct hit_record
{
    point3 p; // 3D point of hit
    vec3 normal; // normal
    float t; // t value in ray equation
    bool front_face;

    __device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }
};

class hittable
{
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif