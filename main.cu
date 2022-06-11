#include <iostream>
#include <chrono>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "rt.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"


// Property variables
struct Properties
{
    // Image properties
    const double aspect_ratio = 16.0 / 9.0;
    const int image_width = 1920; 
    const int image_height = (int)(image_width / aspect_ratio);
    int num_pixels = image_width * image_height;
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

// Write color to array
__device__ void write_color(unsigned char *fb, int pixel_index, color pixel_color) \
{
    fb[pixel_index + 0] = int(255.99 * (pixel_color.x()));
    fb[pixel_index + 1] = int(255.99 * (pixel_color.y()));
    fb[pixel_index + 2] = int(255.99 * (pixel_color.z()));
}

// Return color of pixel
__device__ color ray_color(const ray& r, hittable **world)
{
    // temp hit record
    hit_record rec;
    if ((*world)->hit(r, 0, FLT_MAX, rec)) {
        return 0.5 * (rec.normal + color(1,1,1));
    }

    // background color
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0-t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

// Main render
__global__ void render(unsigned char *fb, Properties p, hittable **world)
{
    // x index and y index
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= p.image_width) || (j >= p.image_height)) 
        return;
    int pixel_index = j * p.image_width * 3 + i * 3;
    // uv offset on viewport
    float u = float(i) / float(p.image_width - 1);
    float v = float(j) / float(p.image_height - 1);

    ray r(p.origin, p.lower_left_corner + u * p.horizontal + v * p.vertical);
    write_color(fb, pixel_index, ray_color(r, world));
}

// Allocate world
__global__ void create_world(hittable **d_list, hittable **d_world)
{
    // Allocate new objects and world
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *d_list = new sphere(vec3(0, 0, -1), 0.5);
        *(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
        *d_world = new hittable_list(d_list, 2);
    }
}

// Deallocate world
__global__ void free_world(hittable **d_list, hittable **d_world) 
{
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
}

int main()
{
    // ------------------- Variables --------------------
    Properties p;

    // Grid dimension
    int tx = 8;
    int ty = 8;

    // -------------------- World -----------------------
    hittable **d_world;
    hittable **d_list;
    cudaMalloc(&d_list, 2 * sizeof(hittable *));
    cudaMalloc(&d_world, sizeof(hittable *));
    create_world<<<1, 1>>>(d_list, d_world);
    cudaDeviceSynchronize();

    // ---------------- memory allocation ---------------
    unsigned char *fb;
    cudaMallocManaged(&fb, p.fb_size);

    // Run render kernel with given sizes.
    dim3 blocks(p.image_width / tx + 1, p.image_height / ty + 1);
    dim3 threads(tx, ty);


    auto start = std::chrono::high_resolution_clock::now();

    // -------------------- RENDER ----------------------
    render<<<blocks, threads>>>(fb, p, d_world);

    // calculate time taken to render
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cerr << "\nFinished in: " << duration.count() / 1000.0 << "ms" << std::endl;


    cudaDeviceSynchronize(); // wait for GPU to finish
    // write to jpg
    stbi_flip_vertically_on_write(true);
    stbi_write_jpg("image.jpg", p.image_width, p.image_height, 3, fb, 100);
    
    // free memory
    free_world<<<1, 1>>>(d_list, d_world);
    cudaFree(fb);
    cudaFree(d_list);
    cudaFree(d_world);
}