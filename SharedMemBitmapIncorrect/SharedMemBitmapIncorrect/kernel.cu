
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"
#include "cpu_bitmap.h"

#include <stdio.h>
#include <math.h>

#define DIM 1024
#define PI 3.14

__global__ void kernel(unsigned char* ptr) {
	//map from threadIdx/blockIdx to pixel position.
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	__shared__ float shared[16][16];

	//calculate the value at that position.
	const float period = 128.f;

	shared[threadIdx.x][threadIdx.y] = 255 * (sinf(x*2.f*PI/period)+1.f) * (sinf(y*2.f*PI/period)+1.f) / 4.f;

	//__syncthreads();

	ptr[offset*4 + 0] = 0;
	ptr[offset*4 + 1] = shared[15-threadIdx.x][15 - threadIdx.y];
	ptr[offset*4 + 2] = 0;
	ptr[offset*4 + 3] = 255;
}

int main(int argc, char** argv) {
	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;

	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());

	dim3 grids(DIM/16, DIM/16);
	dim3 threads(16, 16);

	kernel<<<grids, threads>>>(dev_bitmap);

	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

	bitmap.display_and_exit();

	cudaFree(dev_bitmap);
}