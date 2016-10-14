#include "cuda_runtime.h"
#include <stdio.h>

typedef struct 
{
	int width;
	int height;
	int stride;
	float* elements;
} Matrix;

__global__ void matmul_kernel(Matrix a, Matrix b, Matrix res);

int main()
{
	float *m1, *m2;	// Matrix 1 & 2
	float *dev_m1, *dev_m2;
	float *resultMatrix, dev_resultMatrix;


}

__global__ void matmul_kernel(Matrix a, Matrix b, Matrix res)
{

}

