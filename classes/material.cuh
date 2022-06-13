#ifndef MATERIAL_CUH
#define MATERIAL_CUH

struct hit_record;

#include "ray.cuh"
#include "hittable.cuh"


class material
{
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state) const = 0;
};

// Diffuse
class lambertian : public material
{
public:
    __device__ lambertian(const color& a) : albedo(a) {}

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state) const override
    {
        vec3 scatter_direction = rec.normal + random_unit_vector(local_rand_state);
        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

public:
    color albedo;
};

// Metal
class metal : public material
{
public:
    __device__ metal(const color& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state) const override
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
public:
    float fuzz;
    color albedo;
};

#endif