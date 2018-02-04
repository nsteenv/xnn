// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network dropout layer.
// Created: 02/16/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/dropoutlayer.cuh"

const uint DropoutLayer::c_numCurandBlocks = 96;
const uint DropoutLayer::c_numCurandThreadsPerBlock = 128;

DropoutLayer::DropoutLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, curandState* curandStatesBuffer,
	uint indexInTier, uint tierSize, uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, bool holdsInputData, float dropProbability,
	bool useHostDropoutFilter, bool holdsActivationGradients)
{
	m_layerType = LayerType::Dropout;
	m_parallelismMode = parallelismMode;
	m_deviceCalculationStream = deviceCalculationStream;
	m_deviceMemoryStream = deviceMemoryStream;
	m_indexInTier = indexInTier;
	m_tierSize = tierSize;
	m_curandStatesBuffer = curandStatesBuffer;

	m_inputNumChannels = m_activationNumChannels = inputNumChannels;
	m_inputDataWidth = m_activationDataWidth = inputDataWidth;
	m_inputDataHeight = m_activationDataHeight = inputDataHeight;
	m_inputDataSize = m_activationDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = holdsInputData;

	m_dropProbability = dropProbability;
	m_useHostDropoutFilter = useHostDropoutFilter;

	// Allocating input data buffer.
	m_inputBufferSize = m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);
	if (m_holdsInputData)
	{
		CudaAssert(cudaMalloc<float>(&m_inputDataBuffer, m_inputBufferSize));
	}

	// Allocating input gradients buffer.
	CudaAssert(cudaMalloc<float>(&m_inputGradientsBuffer, m_inputBufferSize));

	// Allocating dropout filter buffer.
	m_dropoutFilterSize = m_inputBufferSize;
	CudaAssert(cudaMalloc<float>(&m_dropoutFilter, m_dropoutFilterSize));

	// Allocating activation data buffers.
	m_activationBufferSize = m_inputBufferSize;
	CudaAssert(cudaMalloc<float>(&m_activationDataBuffer, m_activationBufferSize));

	// Allocating activation gradients buffer.
	m_holdsActivationGradients = holdsActivationGradients;
	if (m_holdsActivationGradients)
	{
		CudaAssert(cudaMalloc<float>(&m_activationGradientsBuffer, m_activationBufferSize));
	}
}

void DropoutLayer::Reinitialize(uint newInputDataCount)
{
	Layer::Reinitialize(newInputDataCount);

	m_dropoutFilterSize = m_inputBufferSize;
}

void DropoutLayer::CopyDropoutFilterFromHost(float* hostDropoutFilter)
{
	CudaAssert(cudaMemcpyAsync(m_dropoutFilter, hostDropoutFilter, m_dropoutFilterSize, cudaMemcpyHostToDevice, m_deviceMemoryStream));
	SynchronizeMemoryOperations();
}

DropoutLayer::~DropoutLayer()
{
	CudaAssert(cudaFree(m_dropoutFilter));
}

void DropoutLayer::LoadInputs()
{
	CommonLoadInputs();
}

/*
	Fills dropout filter with random values sampled from interval (0,1].
*/
__global__ void FillFilter(float* dropoutFilter, const uint dropoutFilterLength, curandState* curandStatesBuffer)
{
	const uint c_dropoutFilterOffset = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Saving state to register for efficiency.
	curandState localState = curandStatesBuffer[c_dropoutFilterOffset];
	
	for (uint dropoutFilterIndex = c_dropoutFilterOffset; dropoutFilterIndex < dropoutFilterLength; dropoutFilterIndex += gridDim.x * blockDim.x)
	{
		dropoutFilter[dropoutFilterIndex] = curand_uniform(&localState);
	}

	// Copying state back to global memory.
	// We need to do this since each generation of random number changes the state of the generator.
	curandStatesBuffer[c_dropoutFilterOffset] = localState;
}

/*
	Drops filter values which are not above the drop probability, setting others to 1.
*/
__global__ void DropFilterValues(float* dropoutFilter, const uint dropoutFilterLength, float dropProbability)
{
	for (uint dropoutFilterIndex = blockIdx.x * blockDim.x + threadIdx.x; dropoutFilterIndex < dropoutFilterLength; dropoutFilterIndex += gridDim.x * blockDim.x)
	{
		dropoutFilter[dropoutFilterIndex] = dropoutFilter[dropoutFilterIndex] > dropProbability ? 1.0f : 0.0f;
	}
}

void DropoutLayer::CreateDropoutFilter()
{
	// Filling dropout filter with random values.
	const uint c_dropoutFilterLength = (uint)(m_dropoutFilterSize / sizeof(float));
	dim3 fillBlockDimensions(c_numCurandThreadsPerBlock);
	dim3 fillGridDimensions(c_numCurandBlocks);
	LAUNCH_KERNEL_ASYNC(FillFilter, fillGridDimensions, fillBlockDimensions, m_deviceCalculationStream)(m_dropoutFilter, c_dropoutFilterLength, m_curandStatesBuffer);
	CudaAssert(cudaGetLastError());

	// Dropping filter values which are not above the drop probability.
	const uint c_numBlocks = 128;
	const uint c_numThreadsPerBlock = 128;
	dim3 dropBlockDimensions(c_numThreadsPerBlock);
	dim3 dropGridDimensions(min(c_numBlocks, DivideUp(c_dropoutFilterLength, c_numThreadsPerBlock)));
	LAUNCH_KERNEL_ASYNC(DropFilterValues, dropGridDimensions, dropBlockDimensions, m_deviceCalculationStream)(m_dropoutFilter, c_dropoutFilterLength, m_dropProbability);
	CudaAssert(cudaGetLastError());
}

void DropoutLayer::ApplyDropoutFilter()
{
	CalculateElementWiseProduct(m_inputDataBuffer, m_dropoutFilter, (uint)(m_dropoutFilterSize / sizeof(float)), m_activationDataBuffer, m_deviceCalculationStream);
}

void DropoutLayer::DoForwardProp(PropagationMode propagationMode)
{
	if (propagationMode == PropagationMode::Train)
	{
		if (!m_useHostDropoutFilter)
		{
			CreateDropoutFilter();
		}
		ApplyDropoutFilter();
	}
	else
	{
		// Scaling inputs by probability that they will not be dropped, which is a reasonable approximation to taking the geometric mean
		// of the predictive distributions produced by the exponentially-many dropout networks.
		CalculateElementWiseScale(m_inputDataBuffer, 1.0f - m_dropProbability, (uint)(m_inputBufferSize / sizeof(float)), m_activationDataBuffer, m_deviceCalculationStream);
	}
}

void DropoutLayer::DoBackwardProp()
{
	CalculateElementWiseProduct(m_activationGradientsBuffer, m_dropoutFilter, (uint)(m_dropoutFilterSize / sizeof(float)), m_inputGradientsBuffer, m_deviceCalculationStream);
}