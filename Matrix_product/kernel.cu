#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct 
{
	int width;
	int height;
	float* elements;
} Matrix;

#define BLOCK_SIZE 16

__global__ void matmul_kernel(Matrix A, Matrix B, Matrix C);

// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	// Load A and B to device memory
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc((void**)&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc((void**)&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

	// Allocation C in device memory
	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc((void**)&d_C.elements, size);

	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	matmul_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	// Read C from device memory
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

int main(int argc, char *argv[])
{
	Matrix A, B, C;
	A.width = 128; A.height = 64;
	B.width = 64; B.height = 256;
	C.width = B.width; C.height = A.height;
	FILE *fp = fopen("result.txt", "w");
	if (fp == NULL)
	{
		printf("Can't open file!\n");
		exit(EXIT_FAILURE);
	}
	// Allocate memory for Matrix
	size_t size = A.width * A.height * sizeof(float);
	A.elements = (float*)malloc(size);
	size = B.width * B.height * sizeof(float);
	B.elements = (float*)malloc(size);
	size = C.width * C.height * sizeof(float);
	C.elements = (float*)malloc(size);

	// Fill data for A and B
	for (int r = 0; r < A.height; ++r)
	{
		for (int c = 0; c < A.width; ++c)
		{
			A.elements[r * A.width + c] = 3.0;
		}
	}
	for (int r = 0; r < B.height; ++r)
	{
		for (int c = 0; c < B.width; ++c)
		{
			B.elements[r * B.width + c] = 4.0;
		}
	}

	MatMul(A, B, C);

	for (int r = 0; r < C.height; ++r)
	{
		for (int c = 0; c < C.width; ++c)
			fprintf(fp, "%3.1lf\t", C.elements[r * C.width + c]);
		fprintf(fp, "\n");
	}	

	fclose(fp);
	free(A.elements);
	free(B.elements);
	free(C.elements);
}

__global__ void matmul_kernel(Matrix A, Matrix B, Matrix C)
{
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	for (int e = 0; e < A.width; ++e)
		Cvalue += A.elements[e + row * A.width] 
				  * B.elements[col + e * B.width]; 

	C.elements[row * C.width + col] = Cvalue;
}

