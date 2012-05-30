
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define INF 2e10f

struct Sphere {
	float r, g, b;
	float radius;
	float x, y, z;

	__device__ float hit(float ox, float oy, float*n) {
	}
};