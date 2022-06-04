#include <iostream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "vec3.cuh"
#include "ray.cuh"

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

__global__ void render(unsigned char *fb, int max_x, int max_y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x * 3 + i * 3;
    color pixel_color(float(i) / max_x, float(j) / max_y, 0.2);
    write_color(fb, pixel_index, pixel_color);
    
}

int main()
{
    int w = 1920;
    int h = 1080;
    int tx = 8;
    int ty = 8;


    int num_pixels = w*h;
    size_t fb_size = 3 * num_pixels * sizeof(float); // rgb * numpixels * size of float


    // memory allocation
    unsigned char *fb;
    cudaMallocManaged(&fb, fb_size);


    // Run render kernel with given sizes.
    dim3 blocks(w / tx + 1, h / ty + 1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(fb, w, h);

    cudaDeviceSynchronize(); // wait for GPU to finish

    // write to jpg
    stbi_write_jpg("image.jpg", w, h, 3, fb, 100);
    

    cudaFree(fb);
}