#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "hittable.cuh"
#include "vec3.cuh"

class sphere : public hittable
{
public:
    __device__ sphere() {}
    __device__ sphere(point3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m) {};

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

public:
    point3 center;
    float radius;
    material *mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    // Sphere equation to detect hit
    vec3 oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius * radius;

    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0) 
        return false;
    float sqrtd = sqrt(discriminant);

    // Find nearest root
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root)
    {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    // Fill hit record
    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

#endif