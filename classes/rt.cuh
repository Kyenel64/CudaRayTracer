#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <limits>
#include <memory>

#define pi 3.1415926535897932385

// Utility Functions

__device__ inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

__device__ inline double clamp(double x, double min, double max)
{
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// Common Headers

#include "ray.cuh"
#include "vec3.cuh"

#endif