#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>
#include <curand_kernel.h>

using std::sqrt;

class vec3 {
    public:
        __host__ __device__ vec3() : e{0,0,0} {}
        __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

        __host__ __device__ inline float x() const { return e[0]; }
        __host__ __device__ inline float y() const { return e[1]; }
        __host__ __device__ inline float z() const { return e[2]; }
        __host__ __device__ inline float r() const { return e[0]; }
        __host__ __device__ inline float g() const { return e[1]; }
        __host__ __device__ inline float b() const { return e[2]; }

        __host__ __device__ inline const vec3& operator+() const { return *this; }
        __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
        __host__ __device__ inline float operator[](int i) const { return e[i]; }
        __host__ __device__ inline float& operator[](int i) { return e[i]; }

        __host__ __device__ vec3& operator+=(const vec3 &v)
        {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }

        __host__ __device__ inline vec3& operator-=(const vec3 &v)
        {
            e[0]  -= v.e[0];
            e[1]  -= v.e[1];
            e[2]  -= v.e[2];
            return *this;
        }

        __host__ __device__ inline vec3& operator*=(const vec3 &v)
        {
            e[0]  *= v.e[0];
            e[1]  *= v.e[1];
            e[2]  *= v.e[2];
            return *this;
        }
        
        __host__ __device__ inline vec3& operator/=(const vec3 &v)
        {
            e[0]  /= v.e[0];
            e[1]  /= v.e[1];
            e[2]  /= v.e[2];
            return *this;
        }

        __host__ __device__ vec3& operator*=(const float t)
        {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }

        __host__ __device__ vec3& operator/=(const float t)
        {
            return *this *= 1/t;
        }

        __host__ __device__ float length() const
        {
            return sqrt(length_squared());
        }

        __host__ __device__ float length_squared() const
        {
            return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
        }
    public:
        float e[3];
};

// Type aliases for vec3
using point3 = vec3;   // 3D point
using color = vec3;    // RGB color

// vec3 Utility Functions

__host__ __device__ inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, float t)
{
    return (1/t) * v;
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v)
{
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v)
{
    return v / v.length();
}


#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ inline vec3 random_in_unit_sphere(curandState *local_rand_state)
{
    vec3 p;
    do {
        p = 2.0f*RANDVEC3 - vec3(1,1,1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ inline vec3 random_unit_vector(curandState *local_rand_state)
{
    return unit_vector(random_in_unit_sphere(local_rand_state));
}

__device__ inline vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2 * dot(v, n) * n;
}

#endif