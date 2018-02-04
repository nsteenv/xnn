// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network max pool layer.
// Created: 02/06/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/maxpoollayer.cuh"

MaxPoolLayer::MaxPoolLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, uint indexInTier,
	uint tierSize, uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, bool holdsInputData, uint unitWidth, uint unitHeight,
	int paddingX, int paddingY, uint unitStride, bool holdsActivationGradients)
{
	m_layerType = LayerType::MaxPool;
	m_parallelismMode = parallelismMode;
	m_deviceCalculationStream = deviceCalculationStream;
	m_deviceMemoryStream = deviceMemoryStream;
	m_indexInTier = indexInTier;
	m_tierSize = tierSize;

	m_inputNumChannels = inputNumChannels;
	m_inputDataWidth = inputDataWidth;
	m_inputDataHeight = inputDataHeight;
	m_inputDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = holdsInputData;

	m_unitWidth = unitWidth;
	m_unitHeight = unitHeight;
	m_paddingX = paddingX;
	m_paddingY = paddingY;
	m_unitStride = unitStride;
	m_numUnitsX = 1 + (uint)ceil((double)(m_paddingX + m_inputDataWidth - m_unitWidth) / m_unitStride);
	m_numUnitsY = 1 + (uint)ceil((double)(m_paddingY + m_inputDataHeight - m_unitHeight) / m_unitStride);

	m_activationNumChannels = inputNumChannels;
	m_activationDataWidth = m_numUnitsX;
	m_activationDataHeight = m_numUnitsY;
	m_activationDataSize = m_activationDataWidth * m_activationDataHeight;

	// Allocating input data buffer.
	m_inputBufferSize = m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);
	if (m_holdsInputData)
	{
		CudaAssert(cudaMalloc<float>(&m_inputDataBuffer, m_inputBufferSize));
	}

	// Allocating input gradients buffer.
	CudaAssert(cudaMalloc<float>(&m_inputGradientsBuffer, m_inputBufferSize));

	// Allocating activation data buffer.
	m_activationBufferSize = m_activationNumChannels * m_activationDataSize * m_inputDataCount * sizeof(float);
	CudaAssert(cudaMalloc<float>(&m_activationDataBuffer, m_activationBufferSize));

	// Allocating activation gradients buffer.
	m_holdsActivationGradients = holdsActivationGradients;
	if (m_holdsActivationGradients)
	{
		CudaAssert(cudaMalloc<float>(&m_activationGradientsBuffer, m_activationBufferSize));
	}
}

void MaxPoolLayer::Reinitialize(uint newInputDataCount)
{
	Layer::Reinitialize(newInputDataCount);

	m_activationBufferSize = m_activationNumChannels * m_activationDataSize * m_inputDataCount * sizeof(float);
}

void MaxPoolLayer::LoadInputs()
{
	CommonLoadInputs();
}

/*
	Applies max pool on input data.

	Needs to be function with template parameters since loops must have constant parameters to be unrolled.
*/
template <uint c_blockWidth, uint c_dataPerThread, uint c_blockHeight, uint c_channelsPerThread, bool c_lastBatch>
__global__ void ApplyPooling(float* dataBuffer, const uint dataWidth, const uint dataHeight, const uint dataSize, const uint dataCount, const uint numChannels,
	const uint unitWidth, const uint unitHeight, const int paddingLeft, const int paddingTop, const uint unitStride, const uint numUnitsX, const uint numUnitsY,
	float* activationsBuffer)
{
	// Blocks are positioned in a grid so that blocks in the same row work on same channel and same vertical position of the pooling unit,
	// and blocks in the same collumn work on same data and same horizontal position of the pooling unit.
	// Blocks are sorted first by pooling position, than by data/channel index.
	const uint c_blocksPerUnitX = gridDim.x / numUnitsX;
	const uint c_blocksPerUnitY = gridDim.y / numUnitsY;
	const uint c_unitX = blockIdx.x / c_blocksPerUnitX;
	const uint c_unitY = blockIdx.y / c_blocksPerUnitY;
	const uint c_numUnits = numUnitsX * numUnitsY;
	const uint c_dataOffset = (blockIdx.x % c_blocksPerUnitX) * c_blockWidth * c_dataPerThread + threadIdx.x;
	const uint c_channelOffset = ((blockIdx.y % c_blocksPerUnitY) * c_blockHeight + threadIdx.y) * c_channelsPerThread;
	if (c_channelOffset >= numChannels)
	{
		// If number of channels is not divisible by block height times number of channels per block (numChannels <= 3), we will have dormant threads.
		return;
	}
	
	// Positioning data buffer.
	dataBuffer += c_channelOffset * dataSize * dataCount + c_dataOffset;

	// Positioning activations buffer.
	activationsBuffer += (c_channelOffset * c_numUnits + c_unitY * numUnitsX + c_unitX) * dataCount + c_dataOffset;

	// Initializing buffer for this thread calculated activations.
	float threadActivations[c_channelsPerThread][c_dataPerThread];
	#pragma unroll
	for (uint channel = 0; channel < c_channelsPerThread; ++channel)
	{
		#pragma unroll
		for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
		{
			threadActivations[channel][dataIndex] = -FLT_MAX;
		}
	}

	// Calculating this thread activations.
	const int c_dataPositionX = -paddingLeft + c_unitX * unitStride;
	const int c_dataPositionY = -paddingTop + c_unitY * unitStride;
	const uint c_dataStartPositionX = max(0, c_dataPositionX);
	const uint c_dataEndPositionX = min(c_dataPositionX + unitHeight, dataWidth);
	const uint c_dataStartPositionY = max(0, c_dataPositionY);
	const uint c_dataEndPositionY = min(c_dataPositionY + unitWidth, dataHeight);
	for (uint currDataPositionY = c_dataStartPositionY; currDataPositionY < c_dataEndPositionY; ++currDataPositionY)
	{
		for (uint currDataPositionX = c_dataStartPositionX; currDataPositionX < c_dataEndPositionX; ++currDataPositionX)
		{
			const uint c_currDataPosition = currDataPositionY * dataWidth + currDataPositionX;
			#pragma unroll
			for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
			{
				if (!c_lastBatch || c_dataOffset + dataIndex * c_blockWidth < dataCount)
				{
					#pragma unroll
					for (uint channel = 0; channel < c_channelsPerThread; ++channel)
					{
						threadActivations[channel][dataIndex] = fmaxf(threadActivations[channel][dataIndex],
							dataBuffer[(channel * dataSize + c_currDataPosition) * dataCount + dataIndex * c_blockWidth]);
					}
				}
			}
		}
	}

	// Writing this thread calculated activations into activations buffer.
	#pragma unroll
	for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
	{
		if (!c_lastBatch || c_dataOffset + dataIndex * c_blockWidth < dataCount)
		{
			#pragma unroll
			for (uint channel = 0; channel < c_channelsPerThread; ++channel)
			{
				activationsBuffer[channel * c_numUnits * dataCount + dataIndex * c_blockWidth] = threadActivations[channel][dataIndex];
			}
		}
	}
}

void MaxPoolLayer::DoForwardProp(PropagationMode propagationMode)
{
	if (m_unitStride == 1 && (m_unitWidth >= 6 || m_unitHeight >= 6) && (m_numUnitsX > 1 || m_numUnitsY > 1))
	{
		ShipAssert(false, "Currently not supported!");
	}
	else
	{
		uint blockWidth = 32;
		uint blockHeight = 4;
		dim3 blockDimensions(blockWidth, blockHeight);
		uint channelsPerThread = m_inputNumChannels % 16 == 0 ? 4 : 1;
		uint dataPerThread = m_inputDataCount % 128 == 0 ? 4 : (m_inputDataCount % 64 == 0 ? 2 : 1);
		bool lastBatch = m_inputDataCount % (blockWidth * dataPerThread) != 0;
		dim3 gridDimensions(DivideUp(m_inputDataCount, blockWidth * dataPerThread) * m_numUnitsX,
			DivideUp(m_inputNumChannels, blockHeight * channelsPerThread) * m_numUnitsY);

		if (lastBatch)
		{
			if (channelsPerThread == 4)
			{
				LAUNCH_KERNEL_ASYNC((ApplyPooling<32, 1, 4, 4, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
					m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_inputNumChannels, m_unitWidth, m_unitHeight, m_paddingX, m_paddingY,
					m_unitStride, m_numUnitsX, m_numUnitsY, m_activationDataBuffer);
			}
			else
			{
				LAUNCH_KERNEL_ASYNC((ApplyPooling<32, 1, 4, 1, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
					m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_inputNumChannels, m_unitWidth, m_unitHeight, m_paddingX, m_paddingY,
					m_unitStride, m_numUnitsX, m_numUnitsY, m_activationDataBuffer);
			}
		}
		else
		{
			if (dataPerThread == 4)
			{
				if (channelsPerThread == 4)
				{
					LAUNCH_KERNEL_ASYNC((ApplyPooling<32, 4, 4, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_inputNumChannels, m_unitWidth, m_unitHeight, m_paddingX, m_paddingY,
						m_unitStride, m_numUnitsX, m_numUnitsY, m_activationDataBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((ApplyPooling<32, 4, 4, 1, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_inputNumChannels, m_unitWidth, m_unitHeight, m_paddingX, m_paddingY,
						m_unitStride, m_numUnitsX, m_numUnitsY, m_activationDataBuffer);
				}
			}
			else if (dataPerThread == 2)
			{
				if (channelsPerThread == 4)
				{
					LAUNCH_KERNEL_ASYNC((ApplyPooling<32, 2, 4, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_inputNumChannels, m_unitWidth, m_unitHeight, m_paddingX, m_paddingY,
						m_unitStride, m_numUnitsX, m_numUnitsY, m_activationDataBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((ApplyPooling<32, 2, 4, 1, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_inputNumChannels, m_unitWidth, m_unitHeight, m_paddingX, m_paddingY,
						m_unitStride, m_numUnitsX, m_numUnitsY, m_activationDataBuffer);
				}
			}
			else
			{
				if (channelsPerThread == 4)
				{
					LAUNCH_KERNEL_ASYNC((ApplyPooling<32, 1, 4, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_inputNumChannels, m_unitWidth, m_unitHeight, m_paddingX, m_paddingY,
						m_unitStride, m_numUnitsX, m_numUnitsY, m_activationDataBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((ApplyPooling<32, 1, 4, 1, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_inputNumChannels, m_unitWidth, m_unitHeight, m_paddingX, m_paddingY,
						m_unitStride, m_numUnitsX, m_numUnitsY, m_activationDataBuffer);
				}
			}
		}

		CudaAssert(cudaGetLastError());
	}
}

/*
	Calculates max pooling input gradients.

	Needs to be function with template parameters since loops must have constant parameters to be unrolled.
*/
template <uint c_blockWidth, uint c_dataPerThread, uint c_blockHeight, uint c_channelsPerThread, bool c_lastBatch>
__global__ void CalculatePoolingGradients(float* dataBuffer, float* activationsBuffer, float* activationGradientsBuffer, const uint dataWidth, const uint dataHeight,
	const uint dataSize, const uint dataCount, const uint numChannels, const uint unitWidth, const uint unitHeight, const int paddingLeft, const int paddingTop,
	const uint unitStride, const uint numUnitsX, const uint numUnitsY, float* inputGradientsBuffer)
{
	// Blocks are positioned in a grid so that blocks in the same row work on same channel and same vertical pixel index of the data,
	// and blocks in the same collumn work on same data and same horizontal pixel index of the data.
	// Blocks are sorted first by pixel position, than by data/channel index.
	const uint c_blocksPerPixelX = gridDim.x / dataWidth;
	const uint c_blocksPerPixelY = gridDim.y / dataHeight;
	const uint c_pixelX = blockIdx.x / c_blocksPerPixelX;
	const uint c_pixelY = blockIdx.y / c_blocksPerPixelY;
	const uint c_pixelIndex = c_pixelY * dataWidth + c_pixelX;
	const uint c_numUnits = numUnitsX * numUnitsY;
	const uint c_dataOffset = (blockIdx.x % c_blocksPerPixelX) * c_blockWidth * c_dataPerThread + threadIdx.x;
	const uint c_channelOffset = (blockIdx.y % c_blocksPerPixelY) * c_blockHeight * c_channelsPerThread + threadIdx.y;

	// Positioning input data and input gradients buffer.
	const uint c_dataBufferOffset = (c_channelOffset * dataSize + c_pixelIndex) * dataCount + c_dataOffset;
	dataBuffer += c_dataBufferOffset;
	inputGradientsBuffer += c_dataBufferOffset;

	// Positioning activations and activations gradients buffer.
	const uint c_activationsBufferOffset = c_channelOffset * c_numUnits * dataCount + c_dataOffset;
	activationsBuffer += c_activationsBufferOffset;
	activationGradientsBuffer += c_activationsBufferOffset;

	// Calculating indexes of units whose activation this pixel affected.
	const uint c_firstUnitX = (uint)paddingLeft + c_pixelX < unitWidth ? 0 : ((uint)paddingLeft + c_pixelX - unitWidth) / unitStride + 1;
	const uint c_firstUnitY = (uint)paddingTop + c_pixelY < unitHeight ? 0 : ((uint)paddingTop + c_pixelY - unitHeight) / unitStride + 1;
	const uint c_lastUnitX = min(numUnitsX, ((uint)paddingLeft + c_pixelX) / unitStride + 1);
	const uint c_lastUnitY = min(numUnitsY, ((uint)paddingTop + c_pixelY) / unitStride + 1);

	// Calculating this thread input gradients.
	for (uint channel = 0; channel < c_channelsPerThread; ++channel)
	{
		const uint c_dataChannelOffset = channel * c_blockHeight * dataSize * dataCount;
		const uint c_activationChannelOffset = channel * c_blockHeight * c_numUnits * dataCount;
		for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
		{
			if (!c_lastBatch || c_dataOffset + dataIndex * c_blockWidth < dataCount)
			{
				float calculatedGradient = 0.f;
				const uint c_dataIndexOffset = dataIndex * c_blockWidth;
				float data = dataBuffer[c_dataChannelOffset + c_dataIndexOffset];
				for (uint unitY = c_firstUnitY; unitY < c_lastUnitY; ++unitY)
				{
					for (uint unitX = c_firstUnitX; unitX < c_lastUnitX; ++unitX)
					{
						const uint c_unitOffset = c_activationChannelOffset + (unitY * numUnitsX + unitX) * dataCount + c_dataIndexOffset;
						float activation = activationsBuffer[c_unitOffset];
						float activationGradient = activationGradientsBuffer[c_unitOffset];

						calculatedGradient += fabs(data - activation) < 0.000000001f ? activationGradient : 0.f;
					}
				}

				// Writing this thread calculated input gradient into input gradients buffer.
				inputGradientsBuffer[c_dataChannelOffset + c_dataIndexOffset] = calculatedGradient;
			}
		}
	}
}

void MaxPoolLayer::DoBackwardProp()
{
	uint blockWidth = 32;
	uint blockHeight = 4;
	dim3 blockDimensions(blockWidth, blockHeight);
	uint channelsPerThread = 2;
	uint dataPerThread = m_inputDataCount % 128 == 0 ? 4 : (m_inputDataCount % 64 == 0 ? 2 : 1);
	bool lastBatch = m_inputDataCount % (blockWidth * dataPerThread) != 0;
	dim3 gridDimensions(DivideUp(m_inputDataCount, blockWidth * dataPerThread) * m_inputDataWidth,
		DivideUp(m_inputNumChannels, blockHeight * channelsPerThread) * m_inputDataHeight);

	if (dataPerThread == 4)
	{
		if (lastBatch)
		{
			LAUNCH_KERNEL_ASYNC((CalculatePoolingGradients<32, 4, 4, 2, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
				m_activationDataBuffer, m_activationGradientsBuffer, m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_inputNumChannels,
				m_unitWidth, m_unitHeight, m_paddingX, m_paddingY, m_unitStride, m_numUnitsX, m_numUnitsY, m_inputGradientsBuffer);
		}
		else
		{
			LAUNCH_KERNEL_ASYNC((CalculatePoolingGradients<32, 4, 4, 2, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
				m_activationDataBuffer, m_activationGradientsBuffer, m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_inputNumChannels,
				m_unitWidth, m_unitHeight, m_paddingX, m_paddingY, m_unitStride, m_numUnitsX, m_numUnitsY, m_inputGradientsBuffer);
		}
	}
	else if (dataPerThread == 2)
	{
		if (lastBatch)
		{
			LAUNCH_KERNEL_ASYNC((CalculatePoolingGradients<32, 2, 4, 2, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
				m_activationDataBuffer, m_activationGradientsBuffer, m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_inputNumChannels,
				m_unitWidth, m_unitHeight, m_paddingX, m_paddingY, m_unitStride, m_numUnitsX, m_numUnitsY, m_inputGradientsBuffer);
		}
		else
		{
			LAUNCH_KERNEL_ASYNC((CalculatePoolingGradients<32, 2, 4, 2, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
				m_activationDataBuffer, m_activationGradientsBuffer, m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_inputNumChannels,
				m_unitWidth, m_unitHeight, m_paddingX, m_paddingY, m_unitStride, m_numUnitsX, m_numUnitsY, m_inputGradientsBuffer);
		}
	}
	else
	{
		if (lastBatch)
		{
			LAUNCH_KERNEL_ASYNC((CalculatePoolingGradients<32, 1, 4, 2, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
				m_activationDataBuffer, m_activationGradientsBuffer, m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_inputNumChannels,
				m_unitWidth, m_unitHeight, m_paddingX, m_paddingY, m_unitStride, m_numUnitsX, m_numUnitsY, m_inputGradientsBuffer);
		}
		else
		{
			LAUNCH_KERNEL_ASYNC((CalculatePoolingGradients<32, 1, 4, 2, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
				m_activationDataBuffer, m_activationGradientsBuffer, m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_inputNumChannels,
				m_unitWidth, m_unitHeight, m_paddingX, m_paddingY, m_unitStride, m_numUnitsX, m_numUnitsY, m_inputGradientsBuffer);
		}
	}

	CudaAssert(cudaGetLastError());
}