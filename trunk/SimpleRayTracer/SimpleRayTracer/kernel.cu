
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
#define DIM 512

struct Sphere {
	float r, g, b;
	float radius;
	float x, y, z;

	__device__ float hit(float ox, float oy, float* n) {
		float dx = ox - x;
		float dy = oy - y;
		if(dx * dx + dy * dy > radius * radius) {
			float dz = sqrtf(radius * radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}

		return -INF;
	}
};

Sphere* s;

int main() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	std::cout<<"Creating cpu bitmap"<<std::endl;
	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;

	//allocate memory on the GPU for the output bitmap.
	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());

	//allocated memory for the sphere data set.
	cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES);

	return 0;
}