// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network matrix operations functions.
// Created: 03/06/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/matrixoperations.cuh"

/*
	Kernel for calculating element-wise product of two matrices by doing grid stride.
*/
__global__ void __CalculateElementWiseProduct(float* inputMatrixA, float* inputMatrixB, uint numElements, float* outputMatrix)
{
	for (uint elementIndex = blockIdx.x * blockDim.x + threadIdx.x; elementIndex < numElements; elementIndex += gridDim.x * blockDim.x)
	{
		outputMatrix[elementIndex] = inputMatrixA[elementIndex] * inputMatrixB[elementIndex];
	}
}

void CalculateElementWiseProduct(float* inputMatrixA, float* inputMatrixB, uint numElements, float* outputMatrix, cudaStream_t deviceCalculationStream)
{
	const uint c_numBlocks = 128;
	const uint c_numThreadsPerBlock = 128;
	dim3 blockDimensions(c_numThreadsPerBlock);
	dim3 gridDimensions(min(c_numBlocks, DivideUp(numElements, c_numThreadsPerBlock)));
	LAUNCH_KERNEL_ASYNC(__CalculateElementWiseProduct, gridDimensions, blockDimensions, deviceCalculationStream)(inputMatrixA, inputMatrixB, numElements,
		outputMatrix);
	CudaAssert(cudaGetLastError());
}

/*
	Kernel for calculating element-wise sum of two matrices by doing grid stride.
*/
__global__ void __CalculateElementWiseSum(float* inputMatrixA, float* inputMatrixB, uint numElements, float* outputMatrix)
{
	for (uint elementIndex = blockIdx.x * blockDim.x + threadIdx.x; elementIndex < numElements; elementIndex += gridDim.x * blockDim.x)
	{
		outputMatrix[elementIndex] = inputMatrixA[elementIndex] + inputMatrixB[elementIndex];
	}
}

void CalculateElementWiseSum(float* inputMatrixA, float* inputMatrixB, uint numElements, float* outputMatrix, cudaStream_t deviceCalculationStream)
{
	const uint c_numBlocks = 128;
	const uint c_numThreadsPerBlock = 128;
	dim3 blockDimensions(c_numThreadsPerBlock);
	dim3 gridDimensions(min(c_numBlocks, DivideUp(numElements, c_numThreadsPerBlock)));
	LAUNCH_KERNEL_ASYNC(__CalculateElementWiseSum, gridDimensions, blockDimensions, deviceCalculationStream)(inputMatrixA, inputMatrixB, numElements,
		outputMatrix);
	CudaAssert(cudaGetLastError());
}

/*
	Kernel for calculating element-wise scale of matrix by doing grid stride.
*/
__global__ void __CalculateElementWiseScale(float* inputMatrix, float scale, uint numElements, float* outputMatrix)
{
	for (uint elementIndex = blockIdx.x * blockDim.x + threadIdx.x; elementIndex < numElements; elementIndex += gridDim.x * blockDim.x)
	{
		outputMatrix[elementIndex] = inputMatrix[elementIndex] * scale;
	}
}

void CalculateElementWiseScale(float* inputMatrix, float scale, uint numElements, float* outputMatrix, cudaStream_t deviceCalculationStream)
{
	const uint c_numBlocks = 128;
	const uint c_numThreadsPerBlock = 128;
	dim3 blockDimensions(c_numThreadsPerBlock);
	dim3 gridDimensions(min(c_numBlocks, DivideUp(numElements, c_numThreadsPerBlock)));
	LAUNCH_KERNEL_ASYNC(__CalculateElementWiseScale, gridDimensions, blockDimensions, deviceCalculationStream)(inputMatrix, scale, numElements,
		outputMatrix);
	CudaAssert(cudaGetLastError());
}