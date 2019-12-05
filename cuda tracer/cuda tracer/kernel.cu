#include <iostream>
#include <fstream>
#include <time.h>
#include <float.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "camera.h"
#include "hittable.h"
#include "hittablelist.h"
#include "sphere.h"
#include "material.h"

//Limited version of checkCudaErrors from helper_cuda.h

#define checkCudaErrors(val) check_errors( (val), #val, __FILE__, __LINE__ )

void check_errors(cudaError_t  error, const char* const func, const char* const file, const int line) {
	if (error) {
		std::cerr << "Cuda Error: " << static_cast<unsigned int>(error) << "at FILE: " << file << " LINE: " << line << " function: " << func << std::endl;
		cudaDeviceReset();
		exit(99);
	}
}

//Limits to 5 ray bounces
__device__ vec3 color(const ray& r, hittable** world, curandState *local_rand_state) {
	ray c_ray = r;//pass by reference then copy??
	vec3 c_att = vec3(1.0, 1.0, 1.0);
	for (int i = 0.0f; i < 7.0f; i++) {
		hit_record rec;
		if ((*world)->hit(c_ray, .001f, FLT_MAX, rec)) {
			ray scat;
			vec3 att;
			if (rec.mat->scatter(c_ray, rec, att, scat, local_rand_state)) {
				c_ray = scat;
				c_att = c_att * att;
			}
			else {
				return vec3(0, 0, 0);
			}
		}
		else {
			vec3 direction = (c_ray.direction.normalize());//v.v.agnitude
			float t = ((direction.y + 1.0f) * 0.5f);
			vec3 c = vec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + vec3(0.5f, 0.7f, 1.0f) * t;
			return c * c_att;
		}
	}
	return vec3(0.0, 0.0, 0.0); //No Collision
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int width, int length, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= width) || (j >= length)) {
		return;
	}
	int index = j * width+ i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, index, 0, &rand_state[index]);
}

__global__ void render(vec3* img, int width, int length, int rays, camera** cam, hittable** world, curandState* rand_state) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= length)
		return;
	int index = x + y * width;
	curandState rand = rand_state[index];
	vec3 tot(0, 0, 0);
	for (int i = 0.0f; i < rays; i++) {
		float u = float(x + curand_uniform(&rand)) / float(width);
		float v = float(y + curand_uniform(&rand)) / float(length);
		tot = tot + color((*cam)->get_ray(u, v, &rand), world, &rand);
	}
	rand_state[index] = rand;
	tot = tot / float(rays);
	tot.x = sqrt(tot.x);
	tot.y = sqrt(tot.y);
	tot.z = sqrt(tot.z);
	img[index] = tot;
}


#define RND (curand_uniform(&local_rand_state))


__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, int width, int length, curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;
		d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000.0f, new lambertian(vec3(.5, .5, .5)));
		int i = 1;
		for (int a = -1; a < 2; a++) {
			for (int b = -1; b < 2; b++) {
				float mat = RND;
				vec3 center(a + RND, .2, b + RND);
				if (mat < 0.8f) {
					d_list[i++] = new sphere(center, 0.2,
						new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
				}
				else if (mat < 0.95f) {
					d_list[i++] = new sphere(center, 0.2,
						new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
				}
				else {
					d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
		d_list[i++] = new sphere(vec3(0, 1, 0), 1.0f, new dielectric(1.5f));
		d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0f, new lambertian(vec3(0.4, 0.2,0.1)));
		d_list[i++] = new sphere(vec3(4, 1, 0), 1.0f, new metal(vec3(0.7, 0.6, 0.5), 0.0f));
		*rand_state = local_rand_state;
		*d_world = new hittable_list(d_list, 3 * 3 + 1 + 3);

		vec3 lookfrom(13, 2, 3);
		vec3 lookat(0, 0, 0);
		float dist_to_focus = 10.0;
		float aperture = 0.1;
		*d_camera = new camera(lookfrom, lookat, vec3(0, 1, 0), 30.0f, float(width) / float(length), aperture, dist_to_focus);
	}
}




__global__ void free_world(hittable** d_list, hittable** d_world,camera** d_camera) {
	for (int i = 0; i < 3 * 3 + 1 + 3; i++) {
		delete ((sphere *)d_list[i])->mat;
		delete d_list[i];
	}
	delete* d_world;
	delete* d_camera;
}


int main() {
	//cudaProfilerStart();
	int width = 600;
	int length = 300;
	int thread_count = 16;
	int rays = 7;

	int area = length * width;
	
	std::cerr << "Rendering a " << width << "x" << length << " image " << std::endl;
	
	vec3* img;
	checkCudaErrors(cudaMallocManaged((void **)&img, area * sizeof(vec3)));
	
	
	// allocate random state
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, (area*sizeof(curandState))));

	
	curandState *d_rand_state2;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));
	
	rand_init<<<1,1>>>(d_rand_state2);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	camera **d_camera;
	checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

	hittable **d_list;
	int num_hitables = 3 * 3 + 1 + 3;
	checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hittable *)));
	
	hittable **d_world;
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));

	create_world<<<1,1>>>(d_list, d_world , d_camera , width , length, d_rand_state2);
	
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	//small squares so that each block does similiar amount of work, if some pixels take alot longer efficiency is impacted
	//multiple of 32 block count

	dim3 blocks((width/thread_count)+1,(length/ thread_count)+1); //num blocks
	dim3 threads(thread_count, thread_count); //block size

	render_init<<<blocks,threads>>>(width,length,d_rand_state);
	
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	//cudaProfilerStop();
	
	render<<<blocks,threads>>>(img, width, length,rays, d_camera ,d_world,d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());//waits till gpu is done

	std::ofstream file("drawing2.ppm");
	if (!(file.is_open())) {
		std::cerr << "Unable to open file" << std::endl;
		return 0;
	}
	file << "P3\n" << width << " " << length << "\n255\n";
	for (int j = length - 1; j >= 0; j--) {
		for (int i = 0; i < width; i++) {
			int r = int(255.99 * (img[i + j * width].x));
			int g = int(255.99 * (img[i + j * width].y));
			int b = int(255.99 * (img[i + j * width].z));
			file << r << " " << g << " " << b << " " << std::endl;
		}
	}
	checkCudaErrors(cudaDeviceSynchronize());
	free_world<<<1, 1>>>(d_list, d_world,d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(img));
	cudaDeviceReset();
	return 0;
}

