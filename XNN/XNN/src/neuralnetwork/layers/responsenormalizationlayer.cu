// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network response normalization layer.
// Created: 02/09/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/responsenormalizationlayer.cuh"

ResponseNormalizationLayer::ResponseNormalizationLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream,
	uint indexInTier, uint tierSize, uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, bool holdsInputData, uint depth, float bias,
	float alphaCoeff, float betaCoeff, bool holdsActivationGradients)
{
	m_layerType = LayerType::ResponseNormalization;
	m_parallelismMode = parallelismMode;
	m_deviceCalculationStream = deviceCalculationStream;
	m_deviceMemoryStream = deviceMemoryStream;
	m_indexInTier = indexInTier;
	m_tierSize = tierSize;

	m_inputNumChannels = m_activationNumChannels = inputNumChannels;
	m_inputDataWidth = m_activationDataWidth = inputDataWidth;
	m_inputDataHeight = m_activationDataHeight = inputDataHeight;
	m_inputDataSize = m_activationDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = holdsInputData;

	m_depth = depth;
	m_bias = bias;
	// Adjusting alpha coefficient upfront, according to formula.
	m_alphaCoeff = alphaCoeff / (float)depth;
	m_betaCoeff = betaCoeff;

	// Allocating input data buffer.
	m_inputBufferSize = m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);
	if (m_holdsInputData)
	{
		CudaAssert(cudaMalloc<float>(&m_inputDataBuffer, m_inputBufferSize));
	}

	// Allocating input gradients buffer.
	CudaAssert(cudaMalloc<float>(&m_inputGradientsBuffer, m_inputBufferSize));

	// Allocating activation data buffer.
	m_activationBufferSize = m_inputBufferSize;
	CudaAssert(cudaMalloc<float>(&m_activationDataBuffer, m_activationBufferSize));

	// Allocating activation gradients buffer.
	m_holdsActivationGradients = holdsActivationGradients;
	if (m_holdsActivationGradients)
	{
		CudaAssert(cudaMalloc<float>(&m_activationGradientsBuffer, m_activationBufferSize));
	}
}

void ResponseNormalizationLayer::LoadInputs()
{
	CommonLoadInputs();
}

/*
	Applies response normalization on input data.

	Needs to be function with template parameters since loops must have constant parameters to be unrolled.
*/
template <uint c_blockWidth, uint c_dataPerThread, uint c_blockHeight, bool c_lastBatch>
__global__ void ApplyResponseNormalization(float* dataBuffer, const uint dataWidth, const uint dataHeight, const uint dataSize, const uint dataCount,
	const int numChannels, int depth, float bias, float alphaCoeff, float betaCoeff, float* activationsBuffer)
{
	// Blocks are positioned in a grid so that each block works on one pixel from one channel.
	// Looking horizontally, first couple of blocks cover pixels (X,Y) where X=1, from each image, then pixels (X,Y) where X=2, from each image, etc.
	// Looking vertically, first couple of blocks cover pixels (X,Y) where Y=1, for each channel, then pixels (X,Y) where Y=2, for each channel, etc.
	const uint c_blocksPerPixelX = gridDim.x / dataWidth;
	const uint c_blocksPerPixelY = gridDim.y / dataHeight;
	const uint c_pixelX = blockIdx.x / c_blocksPerPixelX;
	const uint c_pixelY = blockIdx.y / c_blocksPerPixelY;
	const uint c_channel = (blockIdx.y % c_blocksPerPixelY) * c_blockHeight + threadIdx.y;
	
	// Positioning data and activations buffer.
	const uint c_dataOffset = (blockIdx.x % c_blocksPerPixelX) * c_blockWidth * c_dataPerThread + threadIdx.x;
	const uint c_pixelOffset = (c_pixelY * dataWidth + c_pixelX) * dataCount + c_dataOffset;
	const uint c_bufferOffset = c_channel * dataSize * dataCount + c_pixelOffset;
	activationsBuffer += c_bufferOffset;

	// Initializing buffer for this thread calculated activations.
	float threadActivations[c_dataPerThread];
	#pragma unroll
	for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
	{
		threadActivations[dataIndex] = 0.f;
	}

	// Calculating this thread activations.
	const int c_actualStartChannel = (int)c_channel - depth / 2;
	const int c_startChannel = max(c_actualStartChannel, 0);
	const int c_endChannel = min(c_actualStartChannel + depth, numChannels);
	for (int currChannel = c_startChannel; currChannel < c_endChannel; ++currChannel)
	{
		#pragma unroll
		for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
		{
			if (!c_lastBatch || c_dataOffset + dataIndex * c_blockWidth < dataCount)
			{
				float channelData = dataBuffer[currChannel * dataSize * dataCount + c_pixelOffset + dataIndex * c_blockWidth];
				threadActivations[dataIndex] += channelData * channelData;
			}
		}
	}

	// Writing this thread calculated activations into preactivations buffer.
	#pragma unroll
	for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
	{
		if (!c_lastBatch || c_dataOffset + dataIndex * c_blockWidth < dataCount)
		{
			threadActivations[dataIndex] = bias + alphaCoeff * threadActivations[dataIndex];
			activationsBuffer[dataIndex * c_blockWidth] = dataBuffer[c_bufferOffset + dataIndex * c_blockWidth] *
				__powf(threadActivations[dataIndex], -betaCoeff);
		}
	}
}

void ResponseNormalizationLayer::DoForwardProp(PropagationMode propagationMode)
{
	uint blockWidth = 32;
	uint blockHeight = 4;
	dim3 blockDimensions(blockWidth, blockHeight);
	uint dataPerThread = 4;
	bool lastBatch = m_inputDataCount % (blockWidth * dataPerThread) != 0;
	dim3 gridDimensions(DivideUp(m_inputDataCount, blockWidth * dataPerThread) * m_inputDataWidth,
		(m_inputNumChannels / blockHeight) * m_inputDataHeight);

	if (lastBatch)
	{
		LAUNCH_KERNEL_ASYNC((ApplyResponseNormalization<32, 4, 4, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
			m_inputDataHeight, m_inputDataSize, m_inputDataCount, (int)m_inputNumChannels, (int)m_depth, m_bias, m_alphaCoeff, m_betaCoeff, m_activationDataBuffer);
	}
	else
	{
		LAUNCH_KERNEL_ASYNC((ApplyResponseNormalization<32, 4, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
			m_inputDataHeight, m_inputDataSize, m_inputDataCount, (int)m_inputNumChannels, (int)m_depth, m_bias, m_alphaCoeff, m_betaCoeff, m_activationDataBuffer);
	}
}

/*
	Calculates response normalization input gradients.

	Needs to be function with template parameters since loops must have constant parameters to be unrolled.
*/
template <uint c_blockWidth, uint c_dataPerThread, uint c_blockHeight, bool c_lastBatch>
__global__ void CalculateResponseNormalizationInputGradients(float* dataBuffer, float* activationsBuffer, float* activationsGradientBuffer, const uint dataWidth,
	const uint dataHeight, const uint dataSize, const uint dataCount, const int numChannels, int depth, float bias, float alphaCoeff, float betaCoeff, float* gradientsBuffer)
{
	// Blocks are positioned in a grid so that each block works on one pixel from one channel.
	// Looking horizontally, first couple of blocks cover pixels (X,Y) where X=1, from each image, then pixels (X,Y) where X=2, from each image, etc.
	// Looking vertically, first couple of blocks cover pixels (X,Y) where Y=1, for each channel, then pixels (X,Y) where Y=2, for each channel, etc.
	const uint c_blocksPerPixelX = gridDim.x / dataWidth;
	const uint c_blocksPerPixelY = gridDim.y / dataHeight;
	const uint c_pixelX = blockIdx.x / c_blocksPerPixelX;
	const uint c_pixelY = blockIdx.y / c_blocksPerPixelY;
	const uint c_channel = (blockIdx.y % c_blocksPerPixelY) * c_blockHeight + threadIdx.y;

	// Positioning data and gradients buffer.
	const uint c_dataOffset = (blockIdx.x % c_blocksPerPixelX) * c_blockWidth * c_dataPerThread + threadIdx.x;
	const uint c_pixelOffset = (c_pixelY * dataWidth + c_pixelX) * dataCount + c_dataOffset;
	const uint c_bufferOffset = c_channel * dataSize * dataCount + c_pixelOffset;
	gradientsBuffer += c_bufferOffset;

	// Initializing buffer for this thread calculated gradients.
	float threadGradients[c_dataPerThread];
	#pragma unroll
	for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
	{
		threadGradients[dataIndex] = 0.f;
	}

	// Calculating this thread gradients.
	const int c_actualStartChannel = (int)c_channel - depth + depth / 2 + 1;
	const int c_startChannel = max(c_actualStartChannel, 0);
	const int c_endChannel = min(c_actualStartChannel + depth, numChannels);
	for (int currChannel = c_startChannel; currChannel < c_endChannel; ++currChannel)
	{
		#pragma unroll
		for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
		{
			if (!c_lastBatch || c_dataOffset + dataIndex * c_blockWidth < dataCount)
			{
				const uint c_position = currChannel * dataSize * dataCount + c_pixelOffset + dataIndex * c_blockWidth;
				float data = dataBuffer[c_position];
				float activation = activationsBuffer[c_position];
				float activationGradient = activationsGradientBuffer[c_position];
				threadGradients[dataIndex] += activationGradient * activation * (data == 0.f ? 0.f : __powf(__fdividef(activation, data), 1.0f / betaCoeff));
			}
		}
	}

	// Writing this thread calculated gradients into gradients buffer.
	#pragma unroll
	for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
	{
		if (!c_lastBatch || c_dataOffset + dataIndex * c_blockWidth < dataCount)
		{
			const uint c_position = c_bufferOffset + dataIndex * c_blockWidth;
			float data = dataBuffer[c_position];
			float activation = activationsBuffer[c_position];
			float activationGradient = activationsGradientBuffer[c_position];
			gradientsBuffer[dataIndex * c_blockWidth] = -2.0f * alphaCoeff * betaCoeff * data * threadGradients[dataIndex] +
				activationGradient * (data == 0.f ? 0.f : __fdividef(activation, data));
		}
	}
}

void ResponseNormalizationLayer::DoBackwardProp()
{
	uint blockWidth = 32;
	uint blockHeight = 4;
	dim3 blockDimensions(blockWidth, blockHeight);
	uint dataPerThread = 4;
	bool lastBatch = m_inputDataCount % (blockWidth * dataPerThread) != 0;
	dim3 gridDimensions(DivideUp(m_inputDataCount, blockWidth * dataPerThread) * m_inputDataWidth,
		(m_inputNumChannels / blockHeight) * m_inputDataHeight);

	if (lastBatch)
	{
		LAUNCH_KERNEL_ASYNC((CalculateResponseNormalizationInputGradients<32, 4, 4, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
			m_activationDataBuffer, m_activationGradientsBuffer, m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, (int)m_inputNumChannels,
			(int)m_depth, m_bias, m_alphaCoeff, m_betaCoeff, m_inputGradientsBuffer);
	}
	else
	{
		LAUNCH_KERNEL_ASYNC((CalculateResponseNormalizationInputGradients<32, 4, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
			m_activationDataBuffer, m_activationGradientsBuffer, m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, (int)m_inputNumChannels,
			(int)m_depth, m_bias, m_alphaCoeff, m_betaCoeff, m_inputGradientsBuffer);
	}
}