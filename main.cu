#include <iostream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "vec3.cuh"
#include "ray.cuh"


struct Properties
{
    // ----------- Assign properties -------------
    const double aspect_ratio = 16.0 / 9.0;
    const int image_width = 1920; 
    const int image_height = (int)(image_width / aspect_ratio);
    int num_pixels = image_width*image_height;
    size_t fb_size = 3 * num_pixels * sizeof(float); // rgb * numpixels * size of float

    // Camera properties
    double viewport_height = 2.0;
    double viewport_width = aspect_ratio * viewport_height;
    double focal_length = 1.0;
    point3 origin = point3(0, 0, 0);
    vec3 horizontal = vec3(viewport_width, 0.0, 0.0);
    vec3 vertical = vec3(0.0, viewport_height, 0.0);
    point3 lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);

};

__device__ void write_color(unsigned char *fb, int pixel_index, color pixel_color) \
{
    fb[pixel_index + 0] = int(255.99 * (pixel_color.x()));
    fb[pixel_index + 1] = int(255.99 * (pixel_color.y()));
    fb[pixel_index + 2] = int(255.99 * (pixel_color.z()));
}

__device__ color ray_color(const ray& r)
{
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

__global__ void render(unsigned char *fb, Properties p)
{
    int max_x = p.image_width;
    int max_y = p.image_height;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x * 3 + i * 3;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);

    ray r(p.origin, p.lower_left_corner + u * p.horizontal + v * p.vertical);
    write_color(fb, pixel_index, ray_color(r));
}

int main()
{
    // ------------------- Variables --------------------
    Properties p;

    // Grid dimension
    int tx = 8;
    int ty = 8;


    // ---------------- memory allocation ---------------
    unsigned char *fb;
    cudaMallocManaged(&fb, p.fb_size);

    // Run render kernel with given sizes.
    dim3 blocks(p.image_width / tx + 1, p.image_height / ty + 1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(fb, p);

    cudaDeviceSynchronize(); // wait for GPU to finish

    // write to jpg
    stbi_flip_vertically_on_write(true);
    stbi_write_jpg("image.jpg", p.image_width, p.image_height, 3, fb, 100);
    
    // free memory
    cudaFree(fb);
}