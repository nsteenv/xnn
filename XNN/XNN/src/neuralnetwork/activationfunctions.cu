// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network activation functions.
// Created: 12/27/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/activationfunctions.cuh"

/*
	Linear activations are calculated as: activation = preactivation
*/
__global__ void ApplyLinearActivation(float* preactivations, uint numPreactivations, float* activations)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numPreactivations; activationIndex += gridDim.x * blockDim.x)
	{
		activations[activationIndex] = preactivations[activationIndex];
	}
}

/*
	ReLu activations are calculated as: activation = max(0, preactivation)
*/
__global__ void ApplyReLuActivation(float* preactivations, uint numPreactivations, float* activations)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numPreactivations; activationIndex += gridDim.x * blockDim.x)
	{
		activations[activationIndex] = preactivations[activationIndex] < 0.0f ? 0.0f : preactivations[activationIndex];
	}
}

/*
	Sigmoid activations are calculated as: activation = 1 / (1 + exp(-preactivation))
*/
__global__ void ApplySigmoidActivation(float* preactivations, uint numPreactivations, float* activations)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numPreactivations; activationIndex += gridDim.x * blockDim.x)
	{
		activations[activationIndex] = __fdividef(1.0f, 1.0f + __expf(-preactivations[activationIndex]));
	}
}

/*
	Tanh activations are calculated as: activation = tanh(preactivation)
	= (exp(preactivation) - exp(-preactivation)) / (exp(preactivation) + exp(-preactivation))
	= (exp(2 * preactivation) - 1) / (exp(2 * preactivation) + 1)
*/
__global__ void ApplyTanhActivation(float* preactivations, uint numPreactivations, float* activations)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numPreactivations; activationIndex += gridDim.x * blockDim.x)
	{
		activations[activationIndex] = 1.0f - __fdividef(2.0f, __expf(2.0f * preactivations[activationIndex]) + 1.0f);
	}
}

void ApplyActivation(ActivationType activationType, float* preactivations, uint numPreactivations, float* activations, cudaStream_t deviceCalculationStream)
{
	const uint c_numBlocks = 128;
	const uint c_numThreadsPerBlock = 128;

	dim3 blockDimensions(c_numThreadsPerBlock);
	dim3 gridDimensions(min(c_numBlocks, DivideUp(numPreactivations, c_numThreadsPerBlock)));
	if (activationType == ActivationType::Linear)
	{
		LAUNCH_KERNEL_ASYNC(ApplyLinearActivation, gridDimensions, blockDimensions, deviceCalculationStream)(preactivations, numPreactivations, activations);
	}
	else if (activationType == ActivationType::ReLu)
	{
		LAUNCH_KERNEL_ASYNC(ApplyReLuActivation, gridDimensions, blockDimensions, deviceCalculationStream)(preactivations, numPreactivations, activations);
	}
	else if (activationType == ActivationType::Sigmoid)
	{
		LAUNCH_KERNEL_ASYNC(ApplySigmoidActivation, gridDimensions, blockDimensions, deviceCalculationStream)(preactivations, numPreactivations, activations);
	}
	else if (activationType == ActivationType::Tanh)
	{
		LAUNCH_KERNEL_ASYNC(ApplyTanhActivation, gridDimensions, blockDimensions, deviceCalculationStream)(preactivations, numPreactivations, activations);
	}
	else
	{
		ShipAssert(false, "Unknown activation type!");
	}
	CudaAssert(cudaGetLastError());
}

__global__ void ApplyLinearActivationGradient(float* activationGradients, float* activations, uint numActivations, float* preactivationGradients)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numActivations; activationIndex += gridDim.x * blockDim.x)
	{
		preactivationGradients[activationIndex] = activationGradients[activationIndex];
	}
}

__global__ void ApplyReLuActivationGradient(float* activationGradients, float* activations, uint numActivations, float* preactivationGradients)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numActivations; activationIndex += gridDim.x * blockDim.x)
	{
		preactivationGradients[activationIndex] = activationGradients[activationIndex] * (activations[activationIndex] > 0.0f ? 1.0f : 0.0f);
	}
}

__global__ void ApplySigmoidActivationGradient(float* activationGradients, float* activations, uint numActivations, float* preactivationGradients)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numActivations; activationIndex += gridDim.x * blockDim.x)
	{
		preactivationGradients[activationIndex] = activationGradients[activationIndex] * activations[activationIndex] * (1.0f - activations[activationIndex]);
	}
}

__global__ void ApplyTanhActivationGradient(float* activationGradients, float* activations, uint numActivations, float* preactivationGradients)
{
	for (uint activationIndex = blockIdx.x * blockDim.x + threadIdx.x; activationIndex < numActivations; activationIndex += gridDim.x * blockDim.x)
	{
		preactivationGradients[activationIndex] = activationGradients[activationIndex] * (1.0f - activations[activationIndex] * activations[activationIndex]);
	}
}

void CalculatePreactivationGradients(ActivationType activationType, float* activationGradients, float* activations, uint numActivations, float* preactivationGradients,
	cudaStream_t deviceCalculationStream)
{
	const uint c_numBlocks = 128;
	const uint c_numThreadsPerBlock = 128;

	dim3 blockDimensions(c_numThreadsPerBlock);
	dim3 gridDimensions(min(c_numBlocks, DivideUp(numActivations, c_numThreadsPerBlock)));
	if (activationType == ActivationType::Linear)
	{
		LAUNCH_KERNEL_ASYNC(ApplyLinearActivationGradient, gridDimensions, blockDimensions, deviceCalculationStream)(activationGradients, activations, numActivations,
			preactivationGradients);
	}
	else if (activationType == ActivationType::ReLu)
	{
		LAUNCH_KERNEL_ASYNC(ApplyReLuActivationGradient, gridDimensions, blockDimensions, deviceCalculationStream)(activationGradients, activations, numActivations,
			preactivationGradients);
	}
	else if (activationType == ActivationType::Sigmoid)
	{
		LAUNCH_KERNEL_ASYNC(ApplySigmoidActivationGradient, gridDimensions, blockDimensions, deviceCalculationStream)(activationGradients, activations, numActivations,
			preactivationGradients);
	}
	else if (activationType == ActivationType::Tanh)
	{
		LAUNCH_KERNEL_ASYNC(ApplyTanhActivationGradient, gridDimensions, blockDimensions, deviceCalculationStream)(activationGradients, activations, numActivations,
			preactivationGradients);
	}
	else
	{
		ShipAssert(false, "Unknown activation type!");
	}
	CudaAssert(cudaGetLastError());
}