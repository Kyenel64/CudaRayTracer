#include <iostream>
#include <chrono>
#include <curand_kernel.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "rt.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"
#include "camera.cuh"
#include "material.cuh"

#define object_count 5

// Property variables
struct Properties
{
    // Image properties
    const float aspect_ratio = 16.0 / 9.0;
    const int image_width = 1920; 
    const int image_height = (int)(image_width / aspect_ratio);
    int num_pixels = image_width * image_height;
    size_t fb_size = 3 * num_pixels * sizeof(float); // rgb * numpixels * size of float

    // Render properties
    const int samples_per_pixel = 300;
    const int max_depth = 10;

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
    float r = pixel_color.x();
    float g = pixel_color.y();
    float b = pixel_color.z();

    // Divide color by number of samples. Gamma correct.
    float scale = 1.0 / samples_per_pixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    fb[pixel_index + 0] = int(256 * clamp(r, 0.0, 0.999));
    fb[pixel_index + 1] = int(256 * clamp(g, 0.0, 0.999));
    fb[pixel_index + 2] = int(256 * clamp(b, 0.0, 0.999));
}

// Return color of pixel
__device__ vec3 ray_color(const ray& r, hittable **world, curandState *local_rand_state, Properties p) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1, 1, 1);
    for(int i = 0; i < p.max_depth; i++) 
    {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) 
        {
            ray scattered;
            vec3 atteuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, atteuation, scattered, local_rand_state))
            {
                // multiply current color value by reflection albedo
                cur_attenuation *= atteuation;
                cur_ray = scattered;
            }
            else
            {
                return vec3(0, 0, 0);
            }
        }
        else
        {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
   return vec3(0, 0, 0); // exceeded recursion
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
__global__ void render(unsigned char *fb, Properties p, hittable **world, curandState *rand_state, camera **camera)
{
    // initialize variables and random state
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= p.image_width) || (j >= p.image_height)) 
        return;
    int pixel_index = j * p.image_width * 3 + i * 3;
    int rand_index = j * p.image_width + i;
    curandState local_rand_state = rand_state[rand_index];

    // calculate pixel color
    color pixel_color;
    for (int s = 0; s < p.samples_per_pixel; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(p.image_width);
        float v = float(j + curand_uniform(&local_rand_state)) / float(p.image_height);
        ray r = (*camera)->get_ray(u, v);
        pixel_color += ray_color(r, world, &local_rand_state, p);
    }

    // write color
    write_color(fb, pixel_index, pixel_color, p.samples_per_pixel);
}

// Allocate world
__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera)
{
    // Allocate new objects and world
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_list[0] = new sphere(vec3(0, 0, -1), 0.5, new lambertian(vec3(204, 77, 77)));
        d_list[1] = new sphere(vec3(0, -100.5, -1), 100, new lambertian(vec3(204, 204, 0)));
        d_list[2] = new sphere(vec3(1, 0, -1), 0.5, new metal(vec3(204, 180, 66), 0.5));
        d_list[3] = new sphere(vec3(-1, 0, -1), 0.5, new dielectric(1.5));
        d_list[4] = new sphere(vec3(-1, 0, -1), -0.4, new dielectric(1.5));

        *d_world = new hittable_list(d_list, object_count);
        *d_camera = new camera();
    }
}

// Deallocate world
__global__ void free_world(hittable **d_list, hittable **d_world, camera **d_camera) 
{
    for (int i = 0; i < object_count; i++)
    {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main()
{
    // ------------------- Variables --------------------
    Properties p;

    // Grid dimension
    int tx = 8;
    int ty = 8;

    // -------------------- World -----------------------
    hittable **d_list;
    checkCudaErrors(cudaMalloc(&d_list, object_count * sizeof(hittable *)));
    hittable **d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(hittable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc(&d_camera, sizeof(camera*)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaDeviceSynchronize());

    // ---------------- memory allocation ---------------
    unsigned char *fb;
    checkCudaErrors(cudaMallocManaged(&fb, p.fb_size));
    curandState *d_rand_state;
    checkCudaErrors(cudaMallocManaged(&d_rand_state, p.num_pixels * sizeof(curandState)));

    // Run render kernel with given sizes.
    dim3 blocks(p.image_width / tx + 1, p.image_height / ty + 1);
    dim3 threads(tx, ty);


    render_init<<<blocks, threads>>>(p.image_width, p.image_height, d_rand_state);
    checkCudaErrors(cudaDeviceSynchronize());
    

    // -------------------- RENDER ----------------------
    auto start = std::chrono::high_resolution_clock::now();
    render<<<blocks, threads>>>(fb, p, d_world, d_rand_state, d_camera);
    // calculate time taken to render
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cerr << "\nFinished in: " << duration.count() / 1000.0 << "ms" << std::endl;


    checkCudaErrors(cudaDeviceSynchronize()); // wait for GPU to finish
    // write to jpg
    stbi_flip_vertically_on_write(true);
    stbi_write_jpg("renders/image.jpg", p.image_width, p.image_height, 3, fb, 100);
    
    // free memory
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_camera));
}