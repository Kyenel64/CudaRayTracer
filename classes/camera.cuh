#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "rt.cuh"

class camera
{
public:
    __device__ camera(point3 lookFrom, point3 lookAt, vec3 up, float fov, float aspect_ratio, float aperture, float focus_dist) {
            float theta = degrees_to_radians(fov);
            float h = tanf(theta / 2);
            float viewport_height = 2.0 * h;
            float viewport_width = aspect_ratio * viewport_height;
            
            w = unit_vector(lookFrom - lookAt);
            u = unit_vector(cross(up, w));
            v = cross(w, u);

            origin = lookFrom;
            horizontal = focus_dist * viewport_width * u;
            vertical = focus_dist * viewport_height * v;
            lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;

            lens_radius = aperture / 2;
        }

    __device__ ray get_ray(float s, float t, curandState *local_rand_state) const
    {
        vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
};

#endif