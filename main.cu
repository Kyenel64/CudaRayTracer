#include <iostream>
#include <chrono>
#include <curand_kernel.h>

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

    // Render properties
    const int samples_per_pixel = 100;
    const int maxDepth = 10;

    // Camera properties
    double viewport_height = 2.0;
    double viewport_width = aspect_ratio * viewport_height;
    double focal_length = 1.0;
    point3 origin = point3(0, 0, 0);
    vec3 horizontal = vec3(viewport_width, 0.0, 0.0);
    vec3 vertical = vec3(0.0, viewport_height, 0.0);
    point3 lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);

};

// Error checking
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}


// Write color to array
__device__ void write_color(unsigned char *fb, int pixel_index, color pixel_color, int samples_per_pixel) 
{
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Divide color by number of samples. Gamma correct.
    auto scale = 1.0 / samples_per_pixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    fb[pixel_index + 0] = int(256 * clamp(r, 0.0, 0.999));
    fb[pixel_index + 1] = int(256 * clamp(g, 0.0, 0.999));
    fb[pixel_index + 2] = int(256 * clamp(b, 0.0, 0.999));
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

// Initializing values like random values before main render
__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
    // x index and y index
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) 
        return;
    int pixel_index = j * max_x + i;

    // Retrieve a random value for each thread
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

// Main render
__global__ void render(unsigned char *fb, Properties p, hittable **world, curandState *rand_state)
{
    // x index and y index
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= p.image_width) || (j >= p.image_height)) 
        return;
    int pixel_index = j * p.image_width * 3 + i * 3;
    int rand_index = j * p.image_width + i;
    curandState local_rand_state = rand_state[rand_index];
    // uv offset on viewport
    color pixel_color;
    for (int s = 0; s < p.samples_per_pixel; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(p.image_width - 1);
        float v = float(j + curand_uniform(&local_rand_state)) / float(p.image_height - 1);
        ray r(p.origin, p.lower_left_corner + u * p.horizontal + v * p.vertical);
        pixel_color += ray_color(r, world);
    }
    write_color(fb, pixel_index, pixel_color, p.samples_per_pixel);
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
    checkCudaErrors(cudaMalloc(&d_list, 2 * sizeof(hittable *)));
    checkCudaErrors(cudaMalloc(&d_world, sizeof(hittable *)));
    create_world<<<1, 1>>>(d_list, d_world);
    checkCudaErrors(cudaDeviceSynchronize());

    // ---------------- memory allocation ---------------
    unsigned char *fb;
    curandState *d_rand_state;
    checkCudaErrors(cudaMallocManaged(&fb, p.fb_size));
    checkCudaErrors(cudaMallocManaged(&d_rand_state, p.num_pixels * sizeof(curandState)));

    // Run render kernel with given sizes.
    dim3 blocks(p.image_width / tx + 1, p.image_height / ty + 1);
    dim3 threads(tx, ty);


    auto start = std::chrono::high_resolution_clock::now();

    // -------------------- RENDER ----------------------

    render_init<<<blocks, threads>>>(p.image_width, p.image_height, d_rand_state);
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, p, d_world, d_rand_state);

    // calculate time taken to render
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cerr << "\nFinished in: " << duration.count() / 1000.0 << "ms" << std::endl;


    checkCudaErrors(cudaDeviceSynchronize()); // wait for GPU to finish
    // write to jpg
    stbi_flip_vertically_on_write(true);
    stbi_write_jpg("image.jpg", p.image_width, p.image_height, 3, fb, 100);
    
    // free memory
    free_world<<<1, 1>>>(d_list, d_world);
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_rand_state));
}