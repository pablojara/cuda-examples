
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"
#include "cpu_bitmap.h"

#include <stdio.h>
#include <math.h>
#include <iostream>

#define INF 2e10f
#define rnd(x) (x * rand() / RAND_MAX)
#define SPHERES 20
#define DIM 1024

struct Sphere {
	float r, g, b;
	float radius;
	float x, y, z;

	__device__ float hit(float ox, float oy, float* n) {
		float dx = ox - x;
		float dy = oy - y;
		if(dx*dx + dy*dy < radius*radius) {
			float dz = sqrtf(radius*radius - dx*dx - dy*dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}

		return -INF;
	}
};

//__constant__ Sphere s[SPHERES];

__global__ void kernel(Sphere* s, unsigned char* ptr) {
	//map from threadIdx/blockIdx to pixel position.
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox = (x - DIM/2);
	float oy = (y - DIM/2);
	
	float r = 1, g = 1, b = 1;
	float maxz = -INF;
	for(int i = 0; i < SPHERES; i++) {
		float n;
		float t = s[i].hit(ox, oy, &n);
		if(t > maxz) {
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
			maxz = t;
		}
	}

	ptr[offset * 4 + 0] = (int)(r*255);
	ptr[offset * 4 + 1] = (int)(g*255);
	ptr[offset * 4 + 2] = (int)(b*255);
	ptr[offset * 4 + 3] = 255;
}

struct DataBlock {
    unsigned char   *dev_bitmap;
    Sphere          *s;
};


int main() {
	DataBlock   data;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	std::cout<<"Creating cpu bitmap"<<std::endl;
	CPUBitmap bitmap(DIM, DIM, &data);
	unsigned char* dev_bitmap;
	Sphere* s;


	//allocate memory on the GPU for the output bitmap.
	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());

	//allocated memory for the sphere data set.
	cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES);

	//allocated temp memory, initialize it, copy to memory on the gpu/constant and then free our temp memory.
	Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
	for(int i = 0; i < SPHERES; i++) {
		temp_s[i].r = rnd(1.f);
		temp_s[i].g = rnd(1.f);
		temp_s[i].b = rnd(1.f);
		temp_s[i].x = rnd(1000.f) - 500.f;
		temp_s[i].y = rnd(1000.f) - 500.f;
		temp_s[i].z = rnd(1000.f) - 500.f;
		temp_s[i].radius = rnd(100.f) + 20;
	}

	//Copy the sphere memory from host to the device.
	cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);

	////Copy the sphere memory from the host to constant memory.
	//cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES);

	free(temp_s);

	//generate teh bitmap for the sphere data.
	dim3 grids(DIM/16, DIM/16);
	dim3 threads(16, 16);
	kernel<<<grids, threads>>>(s, dev_bitmap);

	//copy the bitmap from GPU to host.
	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

	bitmap.display_and_exit();

	//free host our memory.
	cudaFree(dev_bitmap);
	cudaFree(s);

	return 0;
}