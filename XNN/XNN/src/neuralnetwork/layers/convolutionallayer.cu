// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network convolutional layer.
// Created: 01/03/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/convolutionallayer.cuh"

const uint ConvolutionalLayer::c_biasesGradientsSumsPerThread = 128;
const uint ConvolutionalLayer::c_biasesGradientsPartialSumThreadsPerBlock = 128;

ConvolutionalLayer::ConvolutionalLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, uint indexInTier,
	uint tierSize, uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, bool holdsInputData, uint numFilters, uint filterWidth,
	uint filterHeight, uint numFilterChannels, bool initializeWeights, float weightsDeviation, bool initializeBiases, float biasesInitialValue,
	float filtersUpdateMomentum, float filtersUpdateDecay, float filtersUpdateLearningRateProgressStep, float filtersUpdateStartingLearningRate,
	float filtersUpdateLearningRateUpdateFactor, float biasesUpdateMomentum, float biasesUpdateDecay, float biasesUpdateLearningRateProgressStep,
	float biasesUpdateStartingLearningRate, float biasesUpdateLearningRateUpdateFactor, int paddingX, int paddingY, uint stride,
	ActivationType activationType, bool holdsActivationGradients)
{
	m_layerType = LayerType::Convolutional;
	m_parallelismMode = parallelismMode;
	m_deviceCalculationStream = deviceCalculationStream;
	m_deviceMemoryStream = deviceMemoryStream;
	m_indexInTier = indexInTier;
	m_tierSize = tierSize;
	m_activationType = activationType;

	m_inputNumChannels = inputNumChannels;
	m_inputDataWidth = inputDataWidth;
	m_inputDataHeight = inputDataHeight;
	m_inputDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = holdsInputData;

	m_numFilters = numFilters;
	m_filterWidth = filterWidth;
	m_filterHeight = filterHeight;
	m_filterSize = m_filterWidth * m_filterHeight;
	m_numFilterChannels = numFilterChannels;

	m_filtersUpdateMomentum = filtersUpdateMomentum;
	m_filtersUpdateDecay = filtersUpdateDecay;
	m_filtersUpdateLearningRateProgressStep = filtersUpdateLearningRateProgressStep;
	m_filtersUpdateStartingLearningRate = filtersUpdateStartingLearningRate;
	m_filtersUpdateLearningRateUpdateFactor = filtersUpdateLearningRateUpdateFactor;

	m_biasesUpdateMomentum = biasesUpdateMomentum;
	m_biasesUpdateDecay = biasesUpdateDecay;
	m_biasesUpdateLearningRateProgressStep = biasesUpdateLearningRateProgressStep;
	m_biasesUpdateStartingLearningRate = biasesUpdateStartingLearningRate;
	m_biasesUpdateLearningRateUpdateFactor = biasesUpdateLearningRateUpdateFactor;

	m_paddingX = paddingX;
	m_paddingY = paddingY;
	m_stride = stride;
	m_numPatchesX = 1 + (uint)ceil((double)(2 * paddingX + m_inputDataWidth - m_filterWidth) / m_stride);
	m_numPatchesY = 1 + (uint)ceil((double)(2 * paddingY + m_inputDataHeight - m_filterHeight) / m_stride);

	m_activationNumChannels = m_numFilters;
	m_activationDataWidth = m_numPatchesX;
	m_activationDataHeight = m_numPatchesY;
	m_activationDataSize = m_activationDataWidth * m_activationDataHeight;

	// Allocating input data buffer.
	m_inputBufferSize = m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);
	if (m_holdsInputData)
	{
		CudaAssert(cudaMalloc<float>(&m_inputDataBuffer, m_inputBufferSize));
	}

	// Allocating input gradients buffer.
	CudaAssert(cudaMalloc<float>(&m_inputGradientsBuffer, m_inputBufferSize));

	// Allocating filters buffers.
	m_filtersBufferSize = m_numFilters * m_filterSize * m_numFilterChannels * sizeof(float);
	CudaAssert(cudaMalloc<float>(&m_filtersBuffer, m_filtersBufferSize));
	CudaAssert(cudaMalloc<float>(&m_filtersGradientsBuffer, m_filtersBufferSize));
	m_preactivationGradientsPerChunkWidth = m_inputNumChannels > 3 ? 3 : 4;
	m_filtersGradientsPerChunkBufferSize = DivideUp(m_numPatchesX, m_preactivationGradientsPerChunkWidth) *
		DivideUp(m_numPatchesY, m_preactivationGradientsPerChunkWidth) * m_filtersBufferSize;
	CudaAssert(cudaMalloc<float>(&m_filtersGradientsPerChunkBuffer, m_filtersGradientsPerChunkBufferSize));
	CudaAssert(cudaMalloc<float>(&m_filtersUpdateBuffer, m_filtersBufferSize));

	// Initializing filter weights.
	if (initializeWeights)
	{
		InitializeParamsFromDistribution(m_filtersBuffer, m_filtersBufferSize, weightsDeviation);
		InitializeParamsToValue(m_filtersUpdateBuffer, m_filtersBufferSize, 0.f);
	}

	// Allocating biases buffers.
	m_biasesBufferSize = m_numFilters * sizeof(float);
	CudaAssert(cudaMalloc<float>(&m_biasesBuffer, m_biasesBufferSize));
	CudaAssert(cudaMalloc<float>(&m_biasesGradientsBuffer, m_biasesBufferSize));
	CudaAssert(cudaMalloc<float>(&m_biasesUpdateBuffer, m_biasesBufferSize));

	// Allocating buffer for holding partial sums for calculating biases gradients.
	m_biasesGradientsPartialSumBlocks = DivideUp(DivideUp(m_inputDataCount * m_numPatchesY * m_numPatchesX, c_biasesGradientsSumsPerThread),
		c_biasesGradientsPartialSumThreadsPerBlock);
	CudaAssert(cudaMalloc<float>(&m_biasesGradientsPartialSumsBuffer,
		DivideUp(m_inputDataCount * m_numPatchesY * m_numPatchesX, c_biasesGradientsSumsPerThread) * m_biasesBufferSize));

	// Initializing biases.
	if (initializeBiases)
	{
		InitializeParamsToValue(m_biasesBuffer, m_biasesBufferSize, biasesInitialValue);
		InitializeParamsToValue(m_biasesUpdateBuffer, m_biasesBufferSize, 0.f);
	}

	// Allocating preactivation and activation data buffers.
	m_activationBufferSize = m_numFilters * m_activationDataSize * m_inputDataCount * sizeof(float);
	CudaAssert(cudaMalloc<float>(&m_preactivationDataBuffer, m_activationBufferSize));
	CudaAssert(cudaMalloc<float>(&m_activationDataBuffer, m_activationBufferSize));

	// Allocating preactivation gradients buffer.
	CudaAssert(cudaMalloc<float>(&m_preactivationGradientsBuffer, m_activationBufferSize));

	// Allocating activation gradients buffer.
	m_holdsActivationGradients = holdsActivationGradients;
	if (m_holdsActivationGradients)
	{
		CudaAssert(cudaMalloc<float>(&m_activationGradientsBuffer, m_activationBufferSize));
	}
}

void ConvolutionalLayer::Reinitialize(uint newInputDataCount)
{
	Layer::Reinitialize(newInputDataCount);

	m_biasesGradientsPartialSumBlocks = DivideUp(DivideUp(m_inputDataCount * m_numPatchesY * m_numPatchesX, c_biasesGradientsSumsPerThread),
		c_biasesGradientsPartialSumThreadsPerBlock);

	m_activationBufferSize = m_numFilters * m_activationDataSize * m_inputDataCount * sizeof(float);
}

void ConvolutionalLayer::CopyFiltersFromHost(float* hostFiltersBuffer)
{
	CudaAssert(cudaMemcpyAsync(m_filtersBuffer, hostFiltersBuffer, m_filtersBufferSize, cudaMemcpyHostToDevice, m_deviceMemoryStream));
	SynchronizeMemoryOperations();
}

void ConvolutionalLayer::CopyFiltersUpdateFromHost(float* hostFiltersUpdateBuffer)
{
	CudaAssert(cudaMemcpyAsync(m_filtersUpdateBuffer, hostFiltersUpdateBuffer, m_filtersBufferSize, cudaMemcpyHostToDevice, m_deviceMemoryStream));
	SynchronizeMemoryOperations();
}

void ConvolutionalLayer::CopyBiasesFromHost(float* hostBiasesBuffer)
{
	CudaAssert(cudaMemcpyAsync(m_biasesBuffer, hostBiasesBuffer, m_biasesBufferSize, cudaMemcpyHostToDevice, m_deviceMemoryStream));
	SynchronizeMemoryOperations();
}

void ConvolutionalLayer::CopyBiasesUpdateFromHost(float* hostBiasesUpdateBuffer)
{
	CudaAssert(cudaMemcpyAsync(m_biasesUpdateBuffer, hostBiasesUpdateBuffer, m_biasesBufferSize, cudaMemcpyHostToDevice, m_deviceMemoryStream));
	SynchronizeMemoryOperations();
}

ConvolutionalLayer::~ConvolutionalLayer()
{
	CudaAssert(cudaFree(m_filtersBuffer));
	CudaAssert(cudaFree(m_filtersGradientsBuffer));
	CudaAssert(cudaFree(m_filtersGradientsPerChunkBuffer));
	CudaAssert(cudaFree(m_filtersUpdateBuffer));

	CudaAssert(cudaFree(m_biasesBuffer));
	CudaAssert(cudaFree(m_biasesGradientsBuffer));
	CudaAssert(cudaFree(m_biasesGradientsPartialSumsBuffer));
	CudaAssert(cudaFree(m_biasesUpdateBuffer));

	CudaAssert(cudaFree(m_preactivationDataBuffer));
	CudaAssert(cudaFree(m_preactivationGradientsBuffer));
}

void ConvolutionalLayer::LoadInputs()
{
	CommonLoadInputs();
}

/*
	Applies filters on data from input (which has less than or equal to 3 channels).
	Each thread applies specified number of filters to specified number of input data.
	Grid is organized in a way that different collumns work on different data, and different rows work on different filters or same filters but different patch.
	Rows are sorted first by patch than by filters.

	Needs to be function with template parameters since loops must have constant parameters to be unrolled.
*/
template <uint c_blockWidth, uint c_dataPerThread, uint c_blockHeight, uint c_filtersPerThread, uint c_numChannels, uint c_cacheLength, bool c_lastBatch>
__global__ void ApplyFiltersOnInputData(float* dataBuffer, const uint dataWidth, const uint dataHeight, const uint dataSize, const uint dataCount, const int paddingX,
	const int paddingY, float* filtersBuffer, const uint filterWidth, const uint filterHeight, const uint filterSize, const uint numFilters, const uint stride,
	const uint numPatchesX, const uint numPatchesY, float* preactivationsBuffer)
{
	const uint c_dataPerBlock = c_blockWidth * c_dataPerThread;
	const uint c_filtersPerBlock = c_blockHeight * c_filtersPerThread;

	// Since same filters are used across threads in same block row and same data in threads across same block collumn,
	// we will benefit from caching data and filters into shared memory.
	// In each pass we are caching cache length pixels from each channel of filters/data.
	const uint c_cacheSize = c_cacheLength * c_numChannels;
	__shared__ float dataCache[c_cacheSize][c_dataPerBlock];
	__shared__ float filtersCache[c_cacheSize][c_filtersPerBlock];
	
	// Positioning filters buffer, it will be loaded into cache, one window by window, where window has dimensions FiltersPerBlock x CacheLength.
	const uint c_blocksPerPatch = numFilters / c_filtersPerBlock;
	const uint c_filtersOffset = (blockIdx.y % c_blocksPerPatch) * c_filtersPerBlock;
	const uint c_threadIndex = threadIdx.y * c_blockWidth + threadIdx.x;
	// Filter cache index represents column in filters data cache window, i.e. which filter are we caching.
	const uint c_filtersCacheIndex = c_threadIndex % c_filtersPerBlock;
	// Filter cache position represents row in filters data cache window, i.e. which filter pixel are we caching.
	const uint c_filtersCachePosition = c_threadIndex / c_filtersPerBlock;
	filtersBuffer += c_filtersOffset + c_filtersCachePosition * numFilters + c_filtersCacheIndex;

	// Positioning data buffer.
	const uint c_dataOffset = blockIdx.x * c_dataPerBlock + threadIdx.x;
	dataBuffer += c_dataOffset;

	// Positioning preactivations buffer.
	const uint c_numPatches = numPatchesX * numPatchesY;
	const uint c_patchIndex = blockIdx.y / c_blocksPerPatch;
	preactivationsBuffer += (c_filtersOffset + threadIdx.y * c_filtersPerThread) * dataCount * c_numPatches + c_patchIndex * dataCount + c_dataOffset;

	// Initializing buffer for this thread calculated preactivations.
	float threadPreactivations[c_filtersPerThread][c_dataPerThread];
	#pragma unroll
	for (uint filterIndex = 0; filterIndex < c_filtersPerThread; ++filterIndex)
	{
		#pragma unroll
		for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
		{
			threadPreactivations[filterIndex][dataIndex] = 0.f;
		}
	}

	// Calculating this thread preactivations.
	const uint c_filtersCacheMaxPosition = c_blockWidth / c_filtersPerThread;
	const bool c_blockFitsInFilters = c_blockWidth % c_filtersPerThread == 0;
	const bool c_blockCoversFilters = c_cacheLength % c_filtersCacheMaxPosition == 0;
	const bool c_blockCoversCache = c_cacheLength % c_blockHeight == 0;
	const int c_dataPositionX = -paddingX + (c_patchIndex % numPatchesX) * stride;
	const int c_dataPositionY = -paddingY + (c_patchIndex / numPatchesX) * stride;
	
	for (uint filterPosition = 0; filterPosition < filterSize; filterPosition += c_cacheLength)
	{
		// Loading filters cache from filter position.
		// Each thread in block loads some of the filters data cache window pixels (window with FiltersPerBlock x CacheLength dimensions).
		// Perfect case is when block totally covers the filters data cache window,
		// worse case is when it just fits the filters data cache window so we have to cover it in couple passes,
		// worst case is when it doesn't fit the filters data cache window so some of block threads are resident.
		if (c_blockFitsInFilters || c_filtersCachePosition < c_filtersCacheMaxPosition)
		{
			#pragma unroll
			for (uint passedCachePosition = 0; passedCachePosition < c_cacheLength; passedCachePosition += c_filtersCacheMaxPosition)
			{
				const uint c_currCachePosition = passedCachePosition + c_filtersCachePosition;
				if (c_blockCoversFilters || c_currCachePosition < c_cacheLength)
				{
					if (filterPosition + c_currCachePosition < filterSize)
					{
						#pragma unroll
						for (uint channel = 0; channel < c_numChannels; ++channel)
						{
							filtersCache[channel * c_cacheLength + c_currCachePosition][c_filtersCacheIndex] =
								filtersBuffer[(channel * filterSize + filterPosition + passedCachePosition) * numFilters];
						}
					}
					else
					{
						#pragma unroll
						for (uint channel = 0; channel < c_numChannels; ++channel)
						{
							filtersCache[channel * c_cacheLength + c_currCachePosition][c_filtersCacheIndex] = 0.f;
						}
					}
				}
			}
		}

		// Loading data cache from filter position in data patch.
		// Each thread in a block loads some data pixel into the data cache, and threads in the same column load
		// different pixels from same data, while threads in the same row load pixels on the same position but from
		// different data.
		#pragma unroll
		for (uint passedCachePosition = 0; passedCachePosition < c_cacheLength; passedCachePosition += c_blockHeight)
		{
			const uint c_currCachePosition = passedCachePosition + threadIdx.y;
			const uint c_currFilterPosition = filterPosition + c_currCachePosition;
			if (c_currFilterPosition < filterSize && (c_blockCoversCache || c_currCachePosition < c_cacheLength))
			{
				const int c_currDataPositionX = c_dataPositionX + c_currFilterPosition % filterWidth;
				const int c_currDataPositionY = c_dataPositionY + c_currFilterPosition / filterWidth;
				if (c_currDataPositionX >= 0 && c_currDataPositionX < dataWidth && c_currDataPositionY >= 0 && c_currDataPositionY < dataHeight)
				{
					float* currDataBufferPosition = dataBuffer + (c_currDataPositionY * dataWidth + c_currDataPositionX) * dataCount;
					#pragma unroll
					for (uint channel = 0; channel < c_numChannels; ++channel)
					{
						#pragma unroll
						for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
						{
							if (!c_lastBatch || c_dataOffset + dataIndex * c_blockWidth < dataCount)
							{
								dataCache[c_currCachePosition + channel * c_cacheLength][threadIdx.x * c_dataPerThread + dataIndex] =
									currDataBufferPosition[channel * dataCount * dataSize + dataIndex * c_blockWidth];
							}
							else
							{
								dataCache[c_currCachePosition + channel * c_cacheLength][threadIdx.x * c_dataPerThread + dataIndex] = 0.f;
							}
						}
					}
				}
				else
				{
					// Fill padded positions with zeros.
					#pragma unroll
					for (uint channel = 0; channel < c_numChannels; ++channel)
					{
						#pragma unroll
						for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
						{
							dataCache[c_currCachePosition + channel * c_cacheLength][threadIdx.x * c_dataPerThread + dataIndex] = 0.f;
						}
					}
				}
			}
		}

		__syncthreads();

		// Applying loaded filter cache to loaded data cache.
		#pragma unroll
		for (uint cacheIndex = 0; cacheIndex < c_cacheSize; ++cacheIndex)
		{
			#pragma unroll
			for (uint filterIndex = 0; filterIndex < c_filtersPerThread; ++filterIndex)
			{
				#pragma unroll
				for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
				{
					threadPreactivations[filterIndex][dataIndex] += dataCache[cacheIndex][threadIdx.x * c_dataPerThread + dataIndex] *
						filtersCache[cacheIndex][threadIdx.y * c_filtersPerThread + filterIndex];
				}
			}
		}

		__syncthreads();
	}

	// Writing this thread calculated preactivations into preactivations buffer.
	#pragma unroll
	for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
	{
		if (!c_lastBatch || c_dataOffset + dataIndex * c_blockWidth < dataCount)
		{
			#pragma unroll
			for (uint filterIndex = 0; filterIndex < c_filtersPerThread; ++filterIndex)
			{
				preactivationsBuffer[dataIndex * c_blockWidth + filterIndex * dataCount * c_numPatches] = threadPreactivations[filterIndex][dataIndex];
			}
		}
	}
}

/*
	Applies filters on filtered data (resulted from previously applied filters).
	Each thread applies specified number of filters to specified number of input data.
	Grid is organized in a way that different collumns work on different data, and different rows work on different filters or same filters but different patch.
	Rows are sorted first by patch than by filters.

	Needs to be function with template parameters since loops must have constant parameters to be unrolled.
*/
template <uint c_blockWidth, uint c_dataPerThread, uint c_blockHeight, uint c_filtersPerThread, uint c_cacheLength, bool c_lastBatch>
__global__ void ApplyFiltersOnFilteredData(float* dataBuffer, const uint dataWidth, const uint dataHeight, const uint dataSize, const uint dataCount,
	const int paddingX, const int paddingY, float* filtersBuffer, const uint filterWidth, const uint filterHeight, const uint filterSize, const uint numFilters,
	const uint numFilterChannels, const uint stride, const uint numPatchesX, const uint numPatchesY, float* preactivationsBuffer)
{
	const uint c_dataPerBlock = c_blockWidth * c_dataPerThread;
	const uint c_filtersPerBlock = c_blockHeight * c_filtersPerThread;

	// Since same filters are used across threads in same block row and same data in threads across same block collumn,
	// we will benefit from caching data and filters into shared memory.
	// In each pass we are caching one pixel from cache length channels of filters/data.
	__shared__ float dataCache[c_cacheLength][c_dataPerBlock];
	__shared__ float filtersCache[c_cacheLength][c_filtersPerBlock];

	// Positioning filters buffer, it will be loaded into cache, one window by window, where window has dimensions FiltersPerBlock x CacheLength.
	const uint c_blocksPerPatch = numFilters / c_filtersPerBlock;
	const uint c_filtersOffset = (blockIdx.y % c_blocksPerPatch) * c_filtersPerBlock;
	const uint c_threadIndex = threadIdx.y * c_blockWidth + threadIdx.x;
	// Filter cache index represents column in filters data cache window, i.e. which filter are we caching.
	const uint c_filtersCacheIndex = c_threadIndex % c_filtersPerBlock;
	// Filter cache position represents row in filters data cache window, i.e. which filter channel are we caching.
	const uint c_filtersCachePosition = c_threadIndex / c_filtersPerBlock;
	filtersBuffer += c_filtersOffset + c_filtersCachePosition * numFilters * filterSize + c_filtersCacheIndex;

	// Positioning data buffer.
	const uint c_dataOffset = blockIdx.x * c_dataPerBlock + threadIdx.x;
	dataBuffer += threadIdx.y * dataCount * dataSize + c_dataOffset;

	// Positioning preactivations buffer.
	const uint c_numPatches = numPatchesX * numPatchesY;
	const uint c_patchIndex = blockIdx.y / c_blocksPerPatch;
	preactivationsBuffer += (c_filtersOffset + threadIdx.y) * dataCount * c_numPatches + c_patchIndex * dataCount + c_dataOffset;

	// Initializing buffer for this thread calculated preactivations.
	float threadPreactivations[c_filtersPerThread][c_dataPerThread];
	#pragma unroll
	for (uint filterIndex = 0; filterIndex < c_filtersPerThread; ++filterIndex)
	{
		#pragma unroll
		for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
		{
			threadPreactivations[filterIndex][dataIndex] = 0.f;
		}
	}

	// Calculating this thread preactivations.
	const uint c_filtersCacheMaxPosition = c_blockWidth / c_filtersPerThread;
	const bool c_blockCoversFilters = c_cacheLength % c_filtersCacheMaxPosition == 0;
	const bool c_blockCoversCache = c_cacheLength % c_blockHeight == 0;
	const int c_dataPositionX = -paddingX + (c_patchIndex % numPatchesX) * stride;
	const int c_dataPositionY = -paddingY + (c_patchIndex / numPatchesX) * stride;
	const uint c_dataStartPositionX = max(0, c_dataPositionX);
	const uint c_dataEndPositionX = min(c_dataPositionX + filterWidth, dataWidth);
	const uint c_dataStartPositionY = max(0, c_dataPositionY);
	const uint c_dataEndPositionY = min(c_dataPositionY + filterHeight, dataHeight);
	for (uint currDataPositionY = c_dataStartPositionY; currDataPositionY < c_dataEndPositionY; ++currDataPositionY)
	{
		const uint c_currFilterPositionY = currDataPositionY - c_dataPositionY;
		for (uint currDataPositionX = c_dataStartPositionX; currDataPositionX < c_dataEndPositionX; ++currDataPositionX)
		{
			const uint c_currFilterPositionX = currDataPositionX - c_dataPositionX;
			const uint c_currFilterPosition = c_currFilterPositionY * filterWidth + c_currFilterPositionX;
			const uint c_currDataPosition = currDataPositionY * dataWidth + currDataPositionX;
			for (uint currChannelPosition = 0; currChannelPosition < numFilterChannels; currChannelPosition += c_cacheLength)
			{
				// Loading filters cache from filter position.
				if (c_filtersCachePosition < c_filtersCacheMaxPosition)
				{
					#pragma unroll
					for (uint passedCachePosition = 0; passedCachePosition < c_cacheLength; passedCachePosition += c_filtersCacheMaxPosition)
					{
						const uint c_currCachePosition = passedCachePosition + c_filtersCachePosition;
						if (c_blockCoversFilters || c_currCachePosition < c_cacheLength)
						{
							filtersCache[c_currCachePosition][c_filtersCacheIndex] =
								filtersBuffer[((currChannelPosition + passedCachePosition)* filterSize + c_currFilterPosition) * numFilters];
						}
					}
				}

				// Loading data cache from filter position in data patch.
				float* currDataBufferPosition = dataBuffer + (currChannelPosition * dataSize + c_currDataPosition) * dataCount;
				#pragma unroll
				for (uint passedCachePosition = 0; passedCachePosition < c_cacheLength; passedCachePosition += c_blockHeight)
				{
					const uint c_currCachePosition = passedCachePosition + threadIdx.y;
					if (c_blockCoversCache || c_currCachePosition < c_cacheLength)
					{
						#pragma unroll
						for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
						{
							if (!c_lastBatch || c_dataOffset + dataIndex * c_blockWidth < dataCount)
							{
								dataCache[c_currCachePosition][threadIdx.x + dataIndex * c_blockWidth] =
									currDataBufferPosition[passedCachePosition * dataCount * dataSize + dataIndex * c_blockWidth];
							}
							else
							{
								dataCache[c_currCachePosition][threadIdx.x + dataIndex * c_blockWidth] = 0.f;
							}
						}
					}
				}

				__syncthreads();

				// Applying loaded filter cache to loaded data cache.
				#pragma unroll
				for (uint cacheIndex = 0; cacheIndex < c_cacheLength; ++cacheIndex)
				{
					#pragma unroll
					for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
					{
						#pragma unroll
						for (uint filterIndex = 0; filterIndex < c_filtersPerThread; ++filterIndex)
						{
							threadPreactivations[filterIndex][dataIndex] += dataCache[cacheIndex][dataIndex * c_blockWidth + threadIdx.x] *
								filtersCache[cacheIndex][filterIndex * c_blockHeight + threadIdx.y];
						}
					}
				}

				__syncthreads();
			}
		}
	}

	// Writing this thread calculated preactivations into preactivations buffer.
	#pragma unroll
	for (uint filterIndex = 0; filterIndex < c_filtersPerThread; ++filterIndex)
	{
		#pragma unroll
		for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
		{
			if (!c_lastBatch || c_dataOffset + dataIndex * c_blockWidth < dataCount)
			{
				preactivationsBuffer[dataIndex * c_blockWidth + filterIndex * c_blockHeight * dataCount * c_numPatches] = threadPreactivations[filterIndex][dataIndex];
			}
		}
	}
}

void ConvolutionalLayer::CalculatePreactivations()
{
	uint dataPerThread = m_inputDataCount % 128 == 0 ? 4 : (m_inputDataCount % 64 == 0 ? 2 : 1);
	uint filtersPerThread = m_numFilters % 64 == 0 ? 16 : ((m_inputNumChannels <= 3 && m_numFilters % 48 == 0) ? 12 : (m_numFilters % 32 == 0 ? 8 : 4));
	uint blockWidth = 32;
	uint blockHeight = (m_numFilters % 128 == 0 && m_numFilterChannels % 8 == 0 && dataPerThread < 4) ? 8 : 4;
	dim3 blockDimensions(blockWidth, blockHeight);
	dim3 gridDimensions(DivideUp(m_inputDataCount, blockWidth * dataPerThread), (m_activationDataSize * m_numFilters) / (blockHeight * filtersPerThread));
	bool lastBatch = m_inputDataCount % (blockWidth * dataPerThread) != 0;

	if (lastBatch)
	{
		if (m_inputNumChannels < 3)
		{
			ShipAssert(false, "Currently not supported!");
		}
		else if (m_inputNumChannels == 3)
		{
			if (m_numFilters % 64 == 0)
			{
				LAUNCH_KERNEL_ASYNC((ApplyFiltersOnInputData<32, 1, 4, 16, 3, 4, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
					m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
					m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
			}
			else if (m_numFilters % 48 == 0)
			{
				LAUNCH_KERNEL_ASYNC((ApplyFiltersOnInputData<32, 1, 4, 12, 3, 4, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
					m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
					m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
			}
			else if (m_numFilters % 32 == 0)
			{
				LAUNCH_KERNEL_ASYNC((ApplyFiltersOnInputData<32, 1, 4, 8, 3, 4, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
					m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
					m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
			}
			else
			{
				LAUNCH_KERNEL_ASYNC((ApplyFiltersOnInputData<32, 1, 4, 4, 3, 4, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
					m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
					m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
			}
		}
		else if (m_inputNumChannels % 8 == 0)
		{
			if (m_numFilters % 128 == 0)
			{
				LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 1, 8, 16, 8, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
					m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
					m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
			}
			else if (m_numFilters % 64 == 0)
			{
				LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 1, 4, 16, 8, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
					m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
					m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
			}
			else if (m_numFilters % 32 == 0)
			{
				LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 1, 4, 8, 8, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
					m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
					m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
			}
			else
			{
				LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 1, 4, 4, 8, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
					m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
					m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
			}
		}
		else if (m_inputNumChannels % 4 == 0)
		{
			if (m_numFilters % 128 == 0)
			{
				LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 1, 4, 16, 4, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
					m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
					m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
			}
			else if (m_numFilters % 64 == 0)
			{
				LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 1, 4, 16, 4, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
					m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
					m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
			}
			else if (m_numFilters % 32 == 0)
			{
				LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 1, 4, 8, 4, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
					m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
					m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
			}
			else
			{
				LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 1, 4, 4, 4, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
					m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
					m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
			}
		}		
	}
	else
	{
		if (m_inputNumChannels < 3)
		{
			ShipAssert(false, "Currently not supported!");
		}
		else if (m_inputNumChannels == 3)
		{
			if (m_inputDataCount % 128 == 0)
			{
				if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnInputData<32, 4, 4, 16, 3, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 48 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnInputData<32, 4, 4, 12, 3, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnInputData<32, 4, 4, 8, 3, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnInputData<32, 4, 4, 4, 3, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
			}
			else if (m_inputDataCount % 64 == 0)
			{
				if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnInputData<32, 2, 4, 16, 3, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 48 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnInputData<32, 2, 4, 12, 3, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnInputData<32, 2, 4, 8, 3, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnInputData<32, 2, 4, 4, 3, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
			}
			else if (m_inputDataCount % 32 == 0)
			{
				if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnInputData<32, 1, 4, 16, 3, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 48 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnInputData<32, 1, 4, 12, 3, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnInputData<32, 1, 4, 8, 3, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnInputData<32, 1, 4, 4, 3, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
			}
		}
		else if (m_inputNumChannels % 8 == 0)
		{
			if (m_inputDataCount % 128 == 0)
			{
				if (m_numFilters % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 4, 4, 16, 8, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 4, 4, 16, 8, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 4, 4, 8, 8, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 4, 4, 4, 8, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
			}
			else if (m_inputDataCount % 64 == 0)
			{
				if (m_numFilters % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 2, 8, 16, 8, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 2, 4, 16, 8, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 2, 4, 8, 8, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 2, 4, 4, 8, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
			}
			else if (m_inputDataCount % 32 == 0)
			{
				if (m_numFilters % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 1, 8, 16, 8, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 1, 4, 16, 8, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 1, 4, 8, 8, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 1, 4, 4, 8, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
			}
		}
		else if (m_inputNumChannels % 4 == 0)
		{
			if (m_inputDataCount % 128 == 0)
			{
				if (m_numFilters % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 4, 4, 16, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 4, 4, 16, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 4, 4, 8, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 4, 4, 4, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
			}
			else if (m_inputDataCount % 64 == 0)
			{
				if (m_numFilters % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 2, 4, 16, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 2, 4, 16, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 2, 4, 8, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 2, 4, 4, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
			}
			else if (m_inputDataCount % 32 == 0)
			{
				if (m_numFilters % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 1, 4, 16, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 1, 4, 16, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 1, 4, 8, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((ApplyFiltersOnFilteredData<32, 1, 4, 4, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer, m_inputDataWidth,
						m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize, m_numFilters,
						m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationDataBuffer);
				}
			}
		}		
	}

	CudaAssert(cudaGetLastError());
}

/*
Does grid stride and adds biases to preactivations.
*/
__global__ void AddFilterBiases(float* preactivations, float* biases, const uint width, const uint height)
{
	for (uint y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += gridDim.y * blockDim.y)
	{
		int laneId = threadIdx.x % warpSize;
		int biasValue;
		if (laneId == 0)
		{
			biasValue = biases[y];
		}
		biasValue = __shfl(biasValue, 0);

		for (uint x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += gridDim.x * blockDim.x)
		{
			preactivations[y * width + x] += biasValue;
		}
	}
}

void ConvolutionalLayer::AddBiases()
{
	dim3 blockDimensions(Config::MAX_NUM_THREADS, 1);
	const uint c_width = m_inputDataCount * m_numPatchesX * m_numPatchesY;
	const uint c_blocksPerWidth = max((uint)1, c_width / (uint)Config::MAX_NUM_THREADS);
	uint gridX = c_blocksPerWidth;
	if (c_blocksPerWidth >= 128)
	{
		gridX = 128;
	}
	else if (c_blocksPerWidth >= 64)
	{
		gridX = 64;
	}
	else if (c_blocksPerWidth >= 32)
	{
		gridX = 32;
	}
	dim3 gridDimensions(gridX, 64);
	LAUNCH_KERNEL_ASYNC(AddFilterBiases, gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationDataBuffer, m_biasesBuffer, c_width, m_numFilters);
	CudaAssert(cudaGetLastError());
}

void ConvolutionalLayer::CalculateActivations()
{
	ApplyActivation(m_activationType, m_preactivationDataBuffer, (uint)(m_activationBufferSize / sizeof(float)), m_activationDataBuffer, m_deviceCalculationStream);
}

void ConvolutionalLayer::DoForwardProp(PropagationMode propagationMode)
{
	CalculatePreactivations();
	AddBiases();
	CalculateActivations();
}

/*
	Calculates partial sums for biases gradients.
*/
__global__ void __CalculateBiasesGradientsPartialSums(float* preactivationGradients, const uint numElementsToSum, float* partialSumsBuffer)
{
	float partialSum = 0.f;
	const uint c_preactivationsGradientsOffset = blockIdx.y * numElementsToSum;
	for (uint partialSumIndex = blockIdx.x * blockDim.x + threadIdx.x; partialSumIndex < numElementsToSum; partialSumIndex += gridDim.x * blockDim.x)
	{
		partialSum += preactivationGradients[c_preactivationsGradientsOffset + partialSumIndex];
	}

	partialSumsBuffer[blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x] = partialSum;
}

/*
	Calculates biases gradients, each thread calculating gradient for one bias.
*/
__global__ void __CalculateConvolutionalBiasesGradients(float* partialSumsBuffer, const uint numFilters, const uint numPartialSums, const uint batchSize,
	float* biasesGradients)
{
	const uint c_filterIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const uint c_filterPartialSumsOffset = c_filterIndex * numPartialSums;

	if (c_filterIndex < numFilters)
	{
		float biasGradient = 0.f;
		for (uint partialSumIndex = 0; partialSumIndex < numPartialSums; ++partialSumIndex)
		{
			biasGradient += partialSumsBuffer[c_filterPartialSumsOffset + partialSumIndex];
		}

		biasesGradients[c_filterIndex] = biasGradient / (float)batchSize;
	}
}

void ConvolutionalLayer::CalculateBiasesGradients()
{
	// Summing biases into temp buffer.
	const uint c_width = m_inputDataCount * m_numPatchesY * m_numPatchesX;
	dim3 blockDimensions(c_biasesGradientsPartialSumThreadsPerBlock);
	dim3 gridDimensions(m_biasesGradientsPartialSumBlocks, m_numFilters);
	LAUNCH_KERNEL_ASYNC(__CalculateBiasesGradientsPartialSums, gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
		c_width, m_biasesGradientsPartialSumsBuffer);
	CudaAssert(cudaGetLastError());

	// Summing from partial sums buffer to biases gradients buffer.
	const uint c_numThreadsPerBlock = 128;
	const uint c_numBlocks = DivideUp(m_numFilters, c_numThreadsPerBlock);
	const uint c_batchSize = m_parallelismMode == ParallelismMode::Model ? m_inputDataCount : m_tierSize * m_inputDataCount;
	LAUNCH_KERNEL_ASYNC(__CalculateConvolutionalBiasesGradients, dim3(c_numBlocks), dim3(c_numThreadsPerBlock), m_deviceCalculationStream)(m_biasesGradientsPartialSumsBuffer,
		m_numFilters, m_biasesGradientsPartialSumBlocks * c_biasesGradientsPartialSumThreadsPerBlock, c_batchSize, m_biasesGradientsBuffer);
	CudaAssert(cudaGetLastError());
}

/*
	Calculates weights gradients on input data.
	Each thread calculates gradient for specified number of weights per thread, for specified number of filters per thread,
	in one chunk of preactivations.

	Needs to be function with template parameters since loops must have constant parameters to be unrolled.
*/
template <uint c_blockWidth, uint c_filtersPerThread, uint c_blockHeight, uint c_pixelsPerThread, uint c_dataPerLoad, uint c_pixelsPerLoad, uint c_numChannels, bool c_lastBatch>
__global__ void CalculateInputDataWeightsGradients(float* inputBuffer, const uint dataWidth, const uint dataHeight, const uint dataSize, const uint dataCount,
	const int paddingX, const int paddingY, float* preactivationGradients, const uint filterWidth, const uint filterHeight, const uint filterSize, const uint numFilters,
	const uint stride, const uint numPatchesX, const uint numPatchesY, const uint preactivationGradientsPerChunkWidth, float* filtersGradientsPerChunkBuffer)
{
	const uint c_filtersPerBlock = c_blockWidth * c_filtersPerThread;
	const uint c_pixelsPerBlock = c_blockHeight * c_pixelsPerThread;
	const uint c_pixelsPerCache = c_blockHeight * c_pixelsPerLoad;

	// Since same filters are used across threads in same block row and same gradients in threads across same block collumn,
	// we will benefit from caching input and gradients into shared memory.
	__shared__ float inputCache[c_pixelsPerCache * c_numChannels][c_dataPerLoad];
	__shared__ int inputPixelOffsetsCache[c_pixelsPerBlock];
	__shared__ float gradientsCache[c_filtersPerBlock][c_dataPerLoad + 1];

	// Positioning inputs buffer.
	const uint c_threadIndex = threadIdx.y * c_blockWidth + threadIdx.x;
	const uint c_cacheLoadIndex = c_threadIndex / c_dataPerLoad;
	const uint c_dataLoadIndex = c_threadIndex % c_dataPerLoad;
	inputBuffer += c_dataLoadIndex;

	// Positioning preactivation gradients buffer.
	const uint c_blocksPerChunk = numFilters / c_filtersPerBlock;
	const uint c_filterOffset = (blockIdx.x % c_blocksPerChunk) * c_filtersPerBlock;
	const uint c_numPatches = numPatchesX * numPatchesY;
	preactivationGradients += c_filterOffset * c_numPatches * dataCount + c_dataLoadIndex;

	// Positioning gradients buffer.
	const uint c_chunkIndex = blockIdx.x / c_blocksPerChunk;
	const uint c_pixelOffset = blockIdx.y * c_pixelsPerBlock;
	filtersGradientsPerChunkBuffer += c_chunkIndex * c_numChannels * numFilters * filterSize + (c_pixelOffset + threadIdx.y) * numFilters + c_filterOffset + threadIdx.x;

	// Initializing buffer for this thread calculated gradients.
	float threadGradients[c_numChannels][c_pixelsPerThread][c_filtersPerThread];
	#pragma unroll
	for (uint channelIndex = 0; channelIndex < c_numChannels; ++channelIndex)
	{
		#pragma unroll
		for (uint pixelIndex = 0; pixelIndex < c_pixelsPerThread; ++pixelIndex)
		{
			#pragma unroll
			for (uint filterIndex = 0; filterIndex < c_filtersPerThread; ++filterIndex)
			{
				threadGradients[channelIndex][pixelIndex][filterIndex] = 0.f;
			}
		}
	}

	// Calculating this thread gradients.	
	const uint c_numChunksX = (numPatchesX + preactivationGradientsPerChunkWidth - 1) / preactivationGradientsPerChunkWidth;
	const uint c_chunkY = c_chunkIndex / c_numChunksX;
	const uint c_chunkX = c_chunkIndex % c_numChunksX;
	const uint c_patchY = c_chunkY * preactivationGradientsPerChunkWidth;
	const uint c_patchX = c_chunkX * preactivationGradientsPerChunkWidth;
	const uint c_firstPatchY = c_patchY;
	const uint c_firstPatchX = c_patchX;
	const uint c_lastPatchY = min(numPatchesY, c_patchY + preactivationGradientsPerChunkWidth);
	const uint c_lastPatchX = min(numPatchesX, c_patchX + preactivationGradientsPerChunkWidth);
	const uint c_filterPixelY = (c_pixelOffset + c_threadIndex) / filterWidth;
	const uint c_filterPixelX = (c_pixelOffset + c_threadIndex) % filterWidth;
	for (uint patchY = c_firstPatchY; patchY < c_lastPatchY; ++patchY)
	{
		const int c_inputPixelY = (int)(c_filterPixelY + patchY * stride) - paddingY;
		for (uint patchX = c_firstPatchX; patchX < c_lastPatchX; ++patchX)
		{
			const int c_inputPixelX = (int)(c_filterPixelX + patchX * stride) - paddingX;
			const uint c_patch = patchY * numPatchesX + patchX;

			// Loading input pixels offsets cache.
			__syncthreads();
			if (c_threadIndex < c_pixelsPerBlock)
			{
				const int c_inputPixelOffset = (c_inputPixelY * (int)dataWidth + c_inputPixelX) * (int)dataCount;
				inputPixelOffsetsCache[c_threadIndex] = (c_inputPixelY >= 0 && c_inputPixelY < dataHeight &&
					c_inputPixelX >= 0 && c_inputPixelX < dataWidth) ? c_inputPixelOffset : -1;
			}
			__syncthreads();
			
			// Load input pixels and gradient pixels for data per load images, and calculate filter gradients on them.
			for (uint dataIndex = 0; dataIndex < dataCount; dataIndex += c_dataPerLoad)
			{
				const uint cacheLoadSlide = (c_blockWidth * c_blockHeight) / c_dataPerLoad;

				// Load gradients cache.
				if (!c_lastBatch || dataIndex + c_dataLoadIndex < dataCount)
				{					
					#pragma unroll
					for (uint filterIndex = 0; filterIndex < c_filtersPerBlock; filterIndex += cacheLoadSlide)
					{
						const uint c_filterToLoad = ((c_cacheLoadIndex + filterIndex) % c_filtersPerThread) * c_blockWidth + (c_cacheLoadIndex + filterIndex) / c_filtersPerThread;
						if (c_filtersPerBlock % cacheLoadSlide == 0 || c_cacheLoadIndex + filterIndex < c_filtersPerBlock)
						{
							gradientsCache[c_cacheLoadIndex + filterIndex][c_dataLoadIndex] = preactivationGradients[c_filterToLoad * c_numPatches * dataCount +
								c_patch * dataCount + dataIndex];
						}
					}
				}
				else
				{
					#pragma unroll
					for (uint filterIndex = 0; filterIndex < c_filtersPerBlock; filterIndex += cacheLoadSlide)
					{
						if (c_filtersPerBlock % cacheLoadSlide == 0 || c_cacheLoadIndex + filterIndex < c_filtersPerBlock)
						{
							gradientsCache[c_cacheLoadIndex + filterIndex][c_dataLoadIndex] = 0.f;
						}
					}
				}

				// Load inputs, cache per cache, and calculate gradients.
				#pragma unroll
				for (uint pixelIndex = 0; pixelIndex < c_pixelsPerThread; pixelIndex += c_pixelsPerLoad)
				{
					// Load inputs cache.
					#pragma unroll
					for (uint loadPixelIndex = 0; loadPixelIndex < c_pixelsPerCache; loadPixelIndex += cacheLoadSlide)
					{
						if (c_pixelsPerCache % cacheLoadSlide == 0 || c_cacheLoadIndex + loadPixelIndex < c_pixelsPerCache)
						{
							const uint c_filterPixel = pixelIndex * c_blockHeight + c_cacheLoadIndex + loadPixelIndex;
							if (c_pixelOffset + c_filterPixel < filterSize && (!c_lastBatch || dataIndex + c_dataLoadIndex < dataCount))
							{
								const int c_inputPixelOffset = inputPixelOffsetsCache[c_filterPixel];
								if (c_inputPixelOffset >= 0)
								{
									#pragma unroll
									for (uint channelIndex = 0; channelIndex < c_numChannels; ++channelIndex)
									{
										inputCache[channelIndex * c_pixelsPerCache + c_cacheLoadIndex + loadPixelIndex][c_dataLoadIndex] =
											inputBuffer[channelIndex * dataSize * dataCount + c_inputPixelOffset + dataIndex];
									}
								}
								else
								{
									#pragma unroll
									for (uint channelIndex = 0; channelIndex < c_numChannels; ++channelIndex)
									{
										inputCache[channelIndex * c_pixelsPerCache + c_cacheLoadIndex + loadPixelIndex][c_dataLoadIndex] = 0.f;
									}
								}
							}
							else
							{
								#pragma unroll
								for (uint channelIndex = 0; channelIndex < c_numChannels; ++channelIndex)
								{
									inputCache[channelIndex * c_pixelsPerCache + c_cacheLoadIndex + loadPixelIndex][c_dataLoadIndex] = 0.f;
								}
							}
						}
					}

					__syncthreads();

					#pragma unroll
					for (uint channelIndex = 0; channelIndex < c_numChannels; ++channelIndex)
					{
						#pragma unroll
						for (uint loadedDataIndex = 0; loadedDataIndex < c_dataPerLoad; ++loadedDataIndex)
						{
							#pragma unroll
							for (uint loadedPixelIndex = 0; loadedPixelIndex < c_pixelsPerLoad; ++loadedPixelIndex)
							{
								#pragma unroll
								for (uint filterIndex = 0; filterIndex < c_filtersPerThread; ++filterIndex)
								{
									threadGradients[channelIndex][pixelIndex + loadedPixelIndex][filterIndex] +=
										inputCache[channelIndex * c_pixelsPerCache + loadedPixelIndex * c_blockHeight + threadIdx.y][loadedDataIndex] *
										gradientsCache[threadIdx.x * c_filtersPerThread + filterIndex][loadedDataIndex];
								}
							}
						}
					}

					__syncthreads();
				}
			}
		}
	}

	// Writing this thread calculated gradients into gradients buffer.
	#pragma unroll
	for (uint pixelIndex = 0; pixelIndex < c_pixelsPerThread; ++pixelIndex)
	{
		if (c_pixelOffset + pixelIndex * c_blockHeight + threadIdx.y < filterSize)
		{
			#pragma unroll
			for (uint channelIndex = 0; channelIndex < c_numChannels; ++channelIndex)
			{
				#pragma unroll
				for (uint filterIndex = 0; filterIndex < c_filtersPerThread; ++filterIndex)
				{
					filtersGradientsPerChunkBuffer[(channelIndex * filterSize + pixelIndex * c_blockHeight) * numFilters + filterIndex * c_blockWidth] =
						threadGradients[channelIndex][pixelIndex][filterIndex];
				}
			}
		}
	}
}

/*
	Calculates weights gradients on filtered data (resulted from previously applied filters).
	Each thread calculates gradient for one weight in specified number of filters per thread and specified number of filter channels per thread,
	in one chunk of preactivations.

	Needs to be function with template parameters since loops must have constant parameters to be unrolled.
*/
template <uint c_blockWidth, uint c_filtersPerThread, uint c_blockHeight, uint c_channelsPerThread, uint c_dataPerLoad, bool c_lastBatch>
__global__ void CalculateFilteredDataWeightsGradients(float* inputBuffer, const uint dataWidth, const uint dataHeight, const uint dataSize, const uint dataCount,
	const int paddingX, const int paddingY, float* preactivationGradients, const uint filterWidth, const uint filterHeight, const uint filterSize, const uint numFilters,
	const uint numFilterChannels, const uint stride, const uint numPatchesX, const uint numPatchesY, const uint preactivationGradientsPerChunkWidth,
	float* filtersGradientsPerChunkBuffer)
{
	const uint c_filtersPerBlock = c_blockWidth * c_filtersPerThread;
	const uint c_channelsPerBlock = c_blockHeight * c_channelsPerThread;

	// Since same filters are used across threads in same block row and same gradients in threads across same block collumn,
	// we will benefit from caching input and gradients into shared memory.
	__shared__ float inputCache[c_channelsPerBlock][c_dataPerLoad];
	__shared__ float gradientsCache[c_filtersPerBlock][c_dataPerLoad + 1];

	// Positioning inputs buffer.
	const uint c_threadIndex = threadIdx.y * c_blockWidth + threadIdx.x;
	const uint c_cacheLoadIndex = c_threadIndex / c_dataPerLoad;
	const uint c_dataLoadIndex = c_threadIndex % c_dataPerLoad;
	const uint c_channelOffset = blockIdx.y * c_channelsPerBlock;
	inputBuffer += (c_channelOffset + c_cacheLoadIndex) * dataSize * dataCount + c_dataLoadIndex;

	// Positioning preactivation gradients buffer.
	const uint c_blocksPerChunk = numFilters / c_filtersPerBlock;
	const uint c_filterOffset = (blockIdx.x % c_blocksPerChunk) * c_filtersPerBlock;
	const uint c_numPatches = numPatchesX * numPatchesY;
	preactivationGradients += (c_filterOffset + c_cacheLoadIndex) * c_numPatches * dataCount + c_dataLoadIndex;

	// Positioning gradients buffer.
	const uint c_chunkIndex = blockIdx.x / c_blocksPerChunk;
	const uint c_filterPixel = blockIdx.z;
	filtersGradientsPerChunkBuffer += (c_chunkIndex * numFilterChannels + c_channelOffset + threadIdx.y) * numFilters * filterSize +
		c_filterPixel * numFilters + c_filterOffset + threadIdx.x;

	// Initializing buffer for this thread calculated gradients.
	float threadGradients[c_channelsPerThread][c_filtersPerThread];
	#pragma unroll
	for (uint channelIndex = 0; channelIndex < c_channelsPerThread; ++channelIndex)
	{
		#pragma unroll
		for (uint filterIndex = 0; filterIndex < c_filtersPerThread; ++filterIndex)
		{
			threadGradients[channelIndex][filterIndex] = 0.f;
		}
	}

	// Calculating this thread gradients.
	const uint c_filterPixelY = c_filterPixel / filterWidth;
	const uint c_filterPixelX = c_filterPixel % filterWidth;
	const uint c_numChunksX = (numPatchesX + preactivationGradientsPerChunkWidth - 1) / preactivationGradientsPerChunkWidth;
	const uint c_chunkY = c_chunkIndex / c_numChunksX;
	const uint c_chunkX = c_chunkIndex % c_numChunksX;
	const uint c_patchY = c_chunkY * preactivationGradientsPerChunkWidth;
	const uint c_patchX = c_chunkX * preactivationGradientsPerChunkWidth;
	const uint c_firstPatchY = (uint)max((int)c_patchY, (-(int)c_filterPixelY + paddingY + (int)stride - 1) / (int)stride);
	const uint c_firstPatchX = (uint)max((int)c_patchX, (-(int)c_filterPixelX + paddingX + (int)stride - 1) / (int)stride);
	const uint c_lastPatchY = min(numPatchesY, min(c_patchY + preactivationGradientsPerChunkWidth, (dataHeight - c_filterPixelY + (uint)paddingY + stride - 1) / stride));
	const uint c_lastPatchX = min(numPatchesX, min(c_patchX + preactivationGradientsPerChunkWidth, (dataWidth - c_filterPixelX + (uint)paddingX + stride - 1) / stride));
	float* inputCacheLoad = &inputCache[c_cacheLoadIndex][c_dataLoadIndex];
	float* gradientsCacheLoad = &gradientsCache[c_cacheLoadIndex][c_dataLoadIndex];
	for (uint patchY = c_firstPatchY; patchY < c_lastPatchY; ++patchY)
	{
		const uint c_inputPixelY = c_filterPixelY + patchY * stride - (uint)paddingY;
		for (uint patchX = c_firstPatchX; patchX < c_lastPatchX; ++patchX)
		{
			const uint c_patch = patchY * numPatchesX + patchX;
			const uint c_inputPixelX = c_filterPixelX + patchX * stride - (uint)paddingX;
			const uint c_inputPixel = (c_inputPixelY * dataWidth + c_inputPixelX) * dataCount;

			// Load input pixels and gradient pixels for data per load images, and calculate filter gradients on them.
			for (uint dataIndex = 0; dataIndex < dataCount; dataIndex += c_dataPerLoad)
			{
				const uint cacheLoadSlide = (c_blockWidth * c_blockHeight) / c_dataPerLoad;

				if (!c_lastBatch || dataIndex + c_dataLoadIndex < dataCount)
				{
					// Load inputs cache.
					if (c_cacheLoadIndex < c_channelsPerBlock)
					{
						#pragma unroll
						for (uint channelIndex = 0; channelIndex < c_channelsPerBlock; channelIndex += cacheLoadSlide)
						{
							if (c_channelsPerBlock % cacheLoadSlide == 0 || c_cacheLoadIndex + channelIndex < c_channelsPerBlock)
							{
								inputCacheLoad[channelIndex * c_dataPerLoad] = inputBuffer[channelIndex * dataSize * dataCount + c_inputPixel + dataIndex];
							}
						}
					}

					// Load gradients cache.
					if (c_cacheLoadIndex < c_filtersPerBlock)
					{
						#pragma unroll
						for (uint filterIndex = 0; filterIndex < c_filtersPerBlock; filterIndex += cacheLoadSlide)
						{
							if (c_filtersPerBlock % cacheLoadSlide == 0 || c_cacheLoadIndex + filterIndex < c_filtersPerBlock)
							{
								gradientsCacheLoad[filterIndex * (c_dataPerLoad + 1)] = preactivationGradients[filterIndex * c_numPatches * dataCount + c_patch * dataCount + dataIndex];
							}
						}
					}
				}
				else
				{
					#pragma unroll
					for (uint channelIndex = 0; channelIndex < c_channelsPerBlock; channelIndex += cacheLoadSlide)
					{
						if (c_channelsPerBlock % cacheLoadSlide == 0 || c_cacheLoadIndex + channelIndex < c_channelsPerBlock)
						{
							inputCacheLoad[channelIndex * c_dataPerLoad] = 0.f;
						}
					}

					#pragma unroll
					for (uint filterIndex = 0; filterIndex < c_filtersPerBlock; filterIndex += cacheLoadSlide)
					{
						if (c_filtersPerBlock % cacheLoadSlide == 0 || c_cacheLoadIndex + filterIndex < c_filtersPerBlock)
						{
							gradientsCacheLoad[filterIndex * (c_dataPerLoad + 1)] = 0.f;
						}
					}
				}

				__syncthreads();

				#pragma unroll
				for (uint loadedDataIndex = 0; loadedDataIndex < c_dataPerLoad; ++loadedDataIndex)
				{
					#pragma unroll
					for (uint filterIndex = 0; filterIndex < c_filtersPerThread; ++filterIndex)
					{
						#pragma unroll
						for (uint channelIndex = 0; channelIndex < c_channelsPerThread; ++channelIndex)
						{
							threadGradients[channelIndex][filterIndex] += inputCache[channelIndex * c_blockHeight + threadIdx.y][loadedDataIndex] *
								gradientsCache[filterIndex * c_blockWidth + threadIdx.x][loadedDataIndex];
						}
					}
				}

				__syncthreads();
			}
		}
	}

	// Writing this thread calculated gradients into gradients buffer.
	#pragma unroll
	for (uint channelIndex = 0; channelIndex < c_channelsPerThread; ++channelIndex)
	{
		#pragma unroll
		for (uint filterIndex = 0; filterIndex < c_filtersPerThread; ++filterIndex)
		{
			filtersGradientsPerChunkBuffer[channelIndex * c_blockHeight * numFilters * filterSize + filterIndex * c_blockWidth] = threadGradients[channelIndex][filterIndex];
		}
	}
}

/*
	Aggregates calculated weights gradients from chunks.
*/
__global__ void AggregateWeightsGradientsFromChunks(float* filtersGradientsPerChunkBuffer, uint numChunks, uint filtersBufferLength, const uint batchSize,
	float* filtersGradientsBuffer)
{
	const uint c_filterIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (c_filterIndex < filtersBufferLength)
	{
		float filterGradient = 0.f;
		for (uint chunkIndex = 0; chunkIndex < numChunks; ++chunkIndex)
		{
			filterGradient += filtersGradientsPerChunkBuffer[chunkIndex * filtersBufferLength + c_filterIndex];
		}

		filtersGradientsBuffer[c_filterIndex] = filterGradient / (float)batchSize;
	}
}

void ConvolutionalLayer::CalculateWeightsGradients()
{
	// Calculating weights gradients for chunks of preactivations.
	uint numChunksX = DivideUp(m_numPatchesX, m_preactivationGradientsPerChunkWidth);
	uint numChunksY = DivideUp(m_numPatchesY, m_preactivationGradientsPerChunkWidth);
	uint numChunks = numChunksX * numChunksY;
	dim3 gridDimensions;
	uint dataPerLoad, blockWidth, blockHeight;
	if (m_inputNumChannels > 3)
	{
		uint filtersPerThread = m_numFilters % 64 == 0 ? 4 : (m_numFilters % 32 == 0 ? 2 : 1);
		blockWidth = m_numFilters % 128 == 0 ? 32 : 16;
		uint channelsPerThread = m_inputNumChannels % 64 == 0 ? 8 : (m_inputNumChannels % 48 == 0 ? 6 : (m_inputNumChannels % 32 == 0 ? 8 : 4));
		blockHeight = (m_inputNumChannels / channelsPerThread) % 8 == 0 ? 8 : 4;
		dataPerLoad = (filtersPerThread * channelsPerThread) < 32 ? 32 : 16;
		gridDimensions = dim3(numChunks * (m_numFilters / (blockWidth * filtersPerThread)), m_inputNumChannels / (blockHeight * channelsPerThread), m_filterSize);
	}
	else
	{
		uint filtersPerThread = 1;
		uint pixelsPerThread = 16;
		blockHeight = 16;
		blockWidth = 16;
		dataPerLoad = 32;
		if (m_numFilters % 64 == 0)
		{
			filtersPerThread = 4;
			pixelsPerThread = 2;
			blockHeight = 16;
			blockWidth = 16;
			dataPerLoad = 32;
		}
		else if (m_numFilters % 48 == 0)
		{
			filtersPerThread = 3;
			pixelsPerThread = 4;
			blockHeight = 16;
			blockWidth = 16;
			dataPerLoad = 32;
		}
		else if (m_numFilters % 32 == 0)
		{
			filtersPerThread = 2;
			pixelsPerThread = 2;
			blockHeight = 8;
			blockWidth = 16;
			dataPerLoad = 16;
		}

		gridDimensions = dim3(numChunks * (m_numFilters / (blockWidth * filtersPerThread)), DivideUp(m_filterSize, blockHeight * pixelsPerThread));
	}
	dim3 blockDimensions(blockWidth, blockHeight);
	bool lastBatch = m_inputDataCount % dataPerLoad != 0;

	if (lastBatch)
	{
		if (m_inputNumChannels < 3)
		{
			ShipAssert(false, "Currently not supported!");
		}
		else if (m_inputNumChannels == 3)
		{
			if (m_numFilters % 64 == 0)
			{
				LAUNCH_KERNEL_ASYNC((CalculateInputDataWeightsGradients<16, 4, 16, 2, 32, 2, 3, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
					m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
					m_filterSize, m_numFilters, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
			}
			else if (m_numFilters % 48 == 0)
			{
				LAUNCH_KERNEL_ASYNC((CalculateInputDataWeightsGradients<16, 3, 16, 4, 32, 2, 3, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
					m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
					m_filterSize, m_numFilters, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
			}
			else if (m_numFilters % 32 == 0)
			{
				LAUNCH_KERNEL_ASYNC((CalculateInputDataWeightsGradients<16, 2, 8, 2, 16, 2, 3, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
					m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
					m_filterSize, m_numFilters, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
			}
			else
			{
				LAUNCH_KERNEL_ASYNC((CalculateInputDataWeightsGradients<16, 1, 16, 16, 32, 2, 3, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
					m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
					m_filterSize, m_numFilters, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
			}
		}
		else
		{
			if (m_inputNumChannels % 64 == 0)
			{
				if (m_numFilters % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<32, 4, 8, 8, 16, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 4, 8, 8, 16, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 2, 8, 8, 32, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 1, 8, 8, 32, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
			}
			else if (m_inputNumChannels % 48 == 0)
			{
				if (m_numFilters % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<32, 4, 8, 6, 32, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 4, 8, 6, 32, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 2, 8, 6, 32, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 1, 8, 6, 32, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
			}
			else if (m_inputNumChannels % 32 == 0)
			{
				if (m_numFilters % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<32, 4, 4, 8, 16, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 4, 4, 8, 16, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 2, 4, 8, 32, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 1, 4, 8, 32, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
			}
			else
			{
				if (m_numFilters % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<32, 4, 4, 4, 32, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 4, 4, 4, 32, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 2, 4, 4, 32, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 1, 4, 4, 32, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
			}
		}
	}
	else
	{
		if (m_inputNumChannels < 3)
		{
			ShipAssert(false, "Currently not supported!");
		}
		else if (m_inputNumChannels == 3)
		{
			if (m_numFilters % 64 == 0)
			{
				LAUNCH_KERNEL_ASYNC((CalculateInputDataWeightsGradients<16, 4, 16, 2, 32, 2, 3, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
					m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
					m_filterSize, m_numFilters, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
			}
			else if (m_numFilters % 48 == 0)
			{
				LAUNCH_KERNEL_ASYNC((CalculateInputDataWeightsGradients<16, 3, 16, 4, 32, 2, 3, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
					m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
					m_filterSize, m_numFilters, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
			}
			else if (m_numFilters % 32 == 0)
			{
				LAUNCH_KERNEL_ASYNC((CalculateInputDataWeightsGradients<16, 2, 8, 2, 16, 2, 3, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
					m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
					m_filterSize, m_numFilters, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
			}
			else
			{
				LAUNCH_KERNEL_ASYNC((CalculateInputDataWeightsGradients<16, 1, 16, 16, 32, 2, 3, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
					m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
					m_filterSize, m_numFilters, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
			}
		}
		else
		{
			if (m_inputNumChannels % 64 == 0)
			{
				if (m_numFilters % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<32, 4, 8, 8, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 4, 8, 8, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 2, 8, 8, 32, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 1, 8, 8, 32, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
			}
			else if (m_inputNumChannels % 48 == 0)
			{
				if (m_numFilters % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<32, 4, 8, 6, 32, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 4, 8, 6, 32, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 2, 8, 6, 32, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 1, 8, 6, 32, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
			}
			else if (m_inputNumChannels % 32 == 0)
			{
				if (m_numFilters % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<32, 4, 4, 8, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 4, 4, 8, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 2, 4, 8, 32, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 1, 4, 8, 32, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
			}
			else
			{
				if (m_numFilters % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<32, 4, 4, 4, 32, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else if (m_numFilters % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 4, 4, 4, 32, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 2, 4, 4, 32, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredDataWeightsGradients<16, 1, 4, 4, 32, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_inputDataBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_preactivationGradientsBuffer, m_filterWidth, m_filterHeight,
						m_filterSize, m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_preactivationGradientsPerChunkWidth, m_filtersGradientsPerChunkBuffer);
				}
			}
		}
	}

	// Aggregating calculated weights gradients from chunks.
	const uint c_numThreadsPerBlock = 128;
	const uint c_filtersBufferLength = (uint)(m_filtersBufferSize / sizeof(float));
	const uint c_numBlocks = DivideUp(c_filtersBufferLength, c_numThreadsPerBlock);
	const uint c_batchSize = m_parallelismMode == ParallelismMode::Model ? m_inputDataCount : m_tierSize * m_inputDataCount;
	LAUNCH_KERNEL_ASYNC(AggregateWeightsGradientsFromChunks, dim3(c_numBlocks), dim3(c_numThreadsPerBlock), m_deviceCalculationStream)(m_filtersGradientsPerChunkBuffer,
		numChunks, c_filtersBufferLength, c_batchSize, m_filtersGradientsBuffer);
	CudaAssert(cudaGetLastError());
}

/*
	Calculates input gradients on input data.
	Each thread calculates gradient for one input pixel of specified number of data per thread.

	Needs to be function with template parameters since loops must have constant parameters to be unrolled.
*/
template <uint c_blockWidth, uint c_dataPerThread, uint c_blockHeight, uint c_dataPerLoad, uint c_numChannels, uint c_blockImagePatchSize, bool c_lastBatch>
__global__ void CalculateDataInputGradients(float* preactivationGradients, const uint dataWidth, const uint dataHeight, const uint dataSize, const uint dataCount,
	const int paddingX, const int paddingY, float* filtersBuffer, const uint filterWidth, const uint filterHeight, const uint filterSize, const uint numFilters,
	const uint stride, const uint numPatchesX, const uint numPatchesY, float* inputGradients)
{
	const uint c_dataPerBlock = c_blockWidth * c_dataPerThread;

	// Since same filters are used across threads in same block row and same gradients in threads across same block collumn,
	// we will benefit from caching gradients and filters into shared memory.
	__shared__ float filtersCache[c_blockHeight * c_numChannels][c_blockWidth + 1]; // Adding 1 to avoid shared memory bank conflicts.
	__shared__ float gradientsCache[c_blockWidth][c_dataPerBlock];

	// Positioning preactivation gradients buffer.
	const uint c_dataOffset = blockIdx.x * c_dataPerBlock;
	const uint c_threadIndex = threadIdx.y * c_blockWidth + threadIdx.x;
	const uint c_gradientsCacheFilterIndex = c_threadIndex / c_dataPerLoad;
	const uint c_gradientsCacheDataIndex = c_threadIndex % c_dataPerLoad;
	const uint c_numPatches = numPatchesX * numPatchesY;
	preactivationGradients += c_gradientsCacheFilterIndex * c_numPatches * dataCount + c_dataOffset + c_gradientsCacheDataIndex;

	// Positioning filters buffer.
	filtersBuffer += threadIdx.x;

	// Positioning input gradients buffer.
	const uint c_numBlockImagePatchesX = (dataWidth + c_blockImagePatchSize - 1) / c_blockImagePatchSize;
	const uint c_blockImagePatchY = blockIdx.y / c_numBlockImagePatchesX;
	const uint c_blockImagePatchX = blockIdx.y % c_numBlockImagePatchesX;
	const uint c_pixelOffsetY = c_blockImagePatchY * c_blockImagePatchSize;
	const uint c_pixelOffsetX = c_blockImagePatchX * c_blockImagePatchSize;
	const uint c_pixelY = c_pixelOffsetY + threadIdx.y / c_blockImagePatchSize;
	const uint c_pixelX = c_pixelOffsetX + threadIdx.y % c_blockImagePatchSize;
	const bool c_validPixel = c_pixelX < dataWidth && c_pixelY < dataHeight;
	inputGradients += (c_pixelY * dataWidth + c_pixelX) * dataCount + c_dataOffset + threadIdx.x;

	// Initializing buffer for this thread calculated gradients.
	float threadGradients[c_numChannels][c_dataPerThread];
	#pragma unroll
	for (uint channelIndex = 0; channelIndex < c_numChannels; ++channelIndex)
	{
		#pragma unroll
		for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
		{
			threadGradients[channelIndex][dataIndex] = 0.f;
		}
	}

	// Calculating this thread gradients.
	const uint c_firstPatchX = c_pixelOffsetX + paddingX < filterWidth ? 0 : (c_pixelOffsetX + paddingX - filterWidth) / stride + 1;
	const uint c_firstPatchY = c_pixelOffsetY + paddingY < filterHeight ? 0 : (c_pixelOffsetY + paddingY - filterHeight) / stride + 1;
	const uint c_lastPatchX = min(numPatchesX, (c_pixelOffsetX + c_blockImagePatchSize - 1 + paddingX) / stride + 1);
	const uint c_lastPatchY = min(numPatchesY, (c_pixelOffsetY + c_blockImagePatchSize - 1 + paddingY) / stride + 1);
	float* filtersCacheLoad = &filtersCache[threadIdx.y][threadIdx.x];
	float* gradientsCacheLoad = &gradientsCache[c_gradientsCacheFilterIndex][c_gradientsCacheDataIndex];
	for (uint currPatchY = c_firstPatchY; currPatchY < c_lastPatchY; ++currPatchY)
	{
		const int c_filterPixelY = (int)c_pixelY + paddingY - (int)(currPatchY * stride);
		for (uint currPatchX = c_firstPatchX; currPatchX < c_lastPatchX; ++currPatchX)
		{
			const int c_filterPixelX = (int)c_pixelX + paddingX - (int)(currPatchX * stride);
			const uint c_filterPixel = c_filterPixelY * filterWidth + c_filterPixelX;
			const uint c_currPatch = currPatchY * numPatchesX + currPatchX;
			const bool c_validFilterPixel = c_filterPixelX >= 0 && c_filterPixelX < filterWidth && c_filterPixelY >= 0 && c_filterPixelY < filterHeight;

			for (uint currFilter = 0; currFilter < numFilters; currFilter += c_blockWidth)
			{
				// Load gradients cache
				const float* preactivationGradientsBufferLoad = preactivationGradients + (currFilter * c_numPatches + c_currPatch) * dataCount;
				const uint c_dataLoadSlide = c_blockWidth * c_blockHeight / c_dataPerLoad;
				#pragma unroll
				for (uint dataToLoadIndex = 0; dataToLoadIndex < c_dataPerBlock; dataToLoadIndex += c_dataPerLoad)
				{
					if (!c_lastBatch || c_dataOffset + dataToLoadIndex + c_gradientsCacheDataIndex < dataCount)
					{
						#pragma unroll						
						for (uint filterToLoad = 0; filterToLoad < c_blockWidth; filterToLoad += c_dataLoadSlide)
						{
							gradientsCacheLoad[filterToLoad * c_dataPerBlock + dataToLoadIndex] = preactivationGradientsBufferLoad[filterToLoad * c_numPatches * dataCount];
						}
					}
					else
					{
						#pragma unroll
						for (uint filterToLoad = 0; filterToLoad < c_blockWidth; filterToLoad += c_dataLoadSlide)
						{
							gradientsCacheLoad[filterToLoad * c_dataPerBlock + dataToLoadIndex] = 0.f;
						}
					}
				}

				if (c_validPixel && c_validFilterPixel)
				{
					// Load filters cache.
					const float* filtersBufferLoad = filtersBuffer + c_filterPixel * numFilters + currFilter;
					#pragma unroll
					for (uint channelIndex = 0; channelIndex < c_numChannels; ++channelIndex)
					{
						filtersCacheLoad[channelIndex * c_blockHeight * (c_blockWidth + 1)] = filtersBufferLoad[channelIndex * filterSize * numFilters];
					}
				}

				__syncthreads();

				if (c_validPixel && c_validFilterPixel)
				{
					#pragma unroll
					for (uint channelIndex = 0; channelIndex < c_numChannels; ++channelIndex)
					{
						#pragma unroll
						for (uint filterIndex = 0; filterIndex < c_blockWidth; ++filterIndex)
						{
							#pragma unroll
							for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
							{
								threadGradients[channelIndex][dataIndex] += filtersCache[channelIndex * c_blockHeight + threadIdx.y][filterIndex] *
									gradientsCache[filterIndex][dataIndex * c_blockWidth + threadIdx.x];
							}
						}
					}
				}

				__syncthreads();
			}
		}
	}

	if (c_validPixel)
	{
		// Writing this thread calculated gradients into gradients buffer.
		#pragma unroll
		for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
		{
			if (!c_lastBatch || c_dataOffset + threadIdx.x + dataIndex * c_blockWidth < dataCount)
			{
				#pragma unroll
				for (uint channelIndex = 0; channelIndex < c_numChannels; ++channelIndex)
				{
					inputGradients[channelIndex * dataSize * dataCount + dataIndex * c_blockWidth] = threadGradients[channelIndex][dataIndex];
				}
			}
		}
	}
}

/*
	Calculates input gradients of filtered data (resulted from previously applied filters).
	Each thread calculates gradient for one input pixel, of specified number of data per thread and specified number of channels per thread.
	Grid is organized in a way that each row of blocks works on one pixel of data, and each column of blocks works on different data,
	or same data but on different channel.
	Columns are sorted first by channel than by data.

	Needs to be function with template parameters since loops must have constant parameters to be unrolled.
*/
template <uint c_blockWidth, uint c_dataPerThread, uint c_blockHeight, uint c_channelsPerThread, uint c_filtersCacheLength, uint c_gradientsCacheLength, bool c_lastBatch>
__global__ void CalculateFilteredInputGradients(float* preactivationGradients, const uint dataWidth, const uint dataHeight, const uint dataSize, const uint dataCount,
	const int paddingX, const int paddingY, float* filtersBuffer, const uint filterWidth, const uint filterHeight, const uint filterSize, const uint numFilters,
	const uint numFilterChannels, const uint stride, const uint numPatchesX, const uint numPatchesY, float* inputGradients)
{
	const uint c_dataPerBlock = c_blockWidth * c_dataPerThread;
	const uint c_channelsPerBlock = c_blockHeight * c_channelsPerThread;

	// Since same filters are used across threads in same block row and same gradients in threads across same block collumn,
	// we will benefit from caching gradients and filters into shared memory.
	__shared__ float filtersCache[c_channelsPerBlock][c_filtersCacheLength];
	__shared__ float gradientsCache[c_gradientsCacheLength][c_dataPerBlock];

	// Positioning preactivation gradients buffer.
	const uint c_blocksPerChannel = gridDim.x / (numFilterChannels / c_channelsPerBlock);
	const uint c_dataOffset = (blockIdx.x % c_blocksPerChannel) * c_dataPerBlock;
	const uint c_gradientsCacheFilterIndex = threadIdx.y;
	const uint c_gradientsCacheDataIndex = threadIdx.x;
	const uint c_numPatches = numPatchesX * numPatchesY;
	preactivationGradients += c_gradientsCacheFilterIndex * c_numPatches * dataCount + c_dataOffset + c_gradientsCacheDataIndex;
	
	// Positioning filters buffer, it will be loaded into cache, one window by window, where window has dimensions ChannelsPerBlock x CacheLength.	
	const uint c_dataChannelIndex = (blockIdx.x / c_blocksPerChannel) * c_channelsPerBlock;
	const uint c_threadIndex = threadIdx.y * c_blockWidth + threadIdx.x;
	const uint c_filtersCacheChannelIndex = c_threadIndex / c_filtersCacheLength;
	const uint c_filtersCachePosition = c_threadIndex % c_filtersCacheLength;
	filtersBuffer += (c_dataChannelIndex + c_filtersCacheChannelIndex) * numFilters * filterSize + c_filtersCachePosition;

	// Positioning input gradients buffer.
	inputGradients += ((c_dataChannelIndex + threadIdx.y) * dataSize + blockIdx.y) * dataCount + c_dataOffset + threadIdx.x;

	// Initializing buffer for this thread calculated gradients.
	float threadGradients[c_channelsPerThread][c_dataPerThread];
	#pragma unroll
	for (uint channelIndex = 0; channelIndex < c_channelsPerThread; ++channelIndex)
	{
		#pragma unroll
		for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
		{
			threadGradients[channelIndex][dataIndex] = 0.f;
		}
	}

	// Calculating this thread gradients.
	const uint c_pixelY = blockIdx.y / dataWidth;
	const uint c_pixelX = blockIdx.y % dataWidth;
	const uint c_firstPatchX = c_pixelX + paddingX < filterWidth ? 0 : (c_pixelX + paddingX - filterWidth) / stride + 1;
	const uint c_firstPatchY = c_pixelY + paddingY < filterHeight ? 0 : (c_pixelY + paddingY - filterHeight) / stride + 1;
	const uint c_lastPatchX = min(numPatchesX, (c_pixelX + paddingX) / stride + 1);
	const uint c_lastPatchY = min(numPatchesY, (c_pixelY + paddingY) / stride + 1);
	float* filtersCacheLoad = &filtersCache[c_filtersCacheChannelIndex][c_filtersCachePosition];
	float* gradientsCacheLoad = &gradientsCache[c_gradientsCacheFilterIndex][c_gradientsCacheDataIndex];
	for (uint currPatchY = c_firstPatchY; currPatchY < c_lastPatchY; ++currPatchY)
	{
		const uint c_filterPixelY = c_pixelY + paddingY - currPatchY * stride;
		for (uint currPatchX = c_firstPatchX; currPatchX < c_lastPatchX; ++currPatchX)
		{
			const uint c_filterPixelX = c_pixelX + paddingX - currPatchX * stride;
			const uint c_filterPixel = c_filterPixelY * filterWidth + c_filterPixelX;
			const uint c_currPatch = currPatchY * numPatchesX + currPatchX;

			for (uint currFilter = 0; currFilter < numFilters; currFilter += c_filtersCacheLength)
			{
				const float* filtersBufferLoad = filtersBuffer + c_filterPixel * numFilters + currFilter;

				// Load filters cache window.
				const uint channelToLoadSlide = c_blockWidth * c_blockHeight / c_filtersCacheLength;
				#pragma unroll				
				for (uint channelToLoad = 0; channelToLoad < c_channelsPerBlock; channelToLoad += channelToLoadSlide)
				{
					if (c_channelsPerBlock % channelToLoadSlide == 0 || channelToLoad + c_filtersCacheChannelIndex < c_channelsPerBlock)
					{
						filtersCacheLoad[channelToLoad * c_filtersCacheLength] = filtersBufferLoad[channelToLoad * filterSize * numFilters];
					}
				}

				for (uint currGradientFilter = currFilter; currGradientFilter < currFilter + c_filtersCacheLength; currGradientFilter += c_gradientsCacheLength)
				{
					// Load gradients cache window.
					const float* preactivationGradientsBufferLoad = preactivationGradients + (currGradientFilter * c_numPatches + c_currPatch) * dataCount;
					#pragma unroll
					for (uint filterToLoad = 0; filterToLoad < c_gradientsCacheLength; filterToLoad += c_blockHeight)
					{
						if (c_gradientsCacheLength % c_blockHeight == 0 || c_gradientsCacheFilterIndex + filterToLoad < c_gradientsCacheLength)
						{
							#pragma unroll
							for (uint dataIndex = 0; dataIndex < c_dataPerBlock; dataIndex += c_blockWidth)
							{
								if (!c_lastBatch || c_dataOffset + c_gradientsCacheDataIndex + dataIndex < dataCount)
								{
									gradientsCacheLoad[filterToLoad * c_dataPerBlock + dataIndex] = preactivationGradientsBufferLoad[filterToLoad * c_numPatches * dataCount + dataIndex];
								}
								else
								{
									gradientsCacheLoad[filterToLoad * c_dataPerBlock + dataIndex] = 0.f;
								}
							}
						}
					}

					__syncthreads();

					// Calculate gradients from cache.
					#pragma unroll
					for (uint filterIndex = 0; filterIndex < c_gradientsCacheLength; ++filterIndex)
					{
						#pragma unroll
						for (uint channelIndex = 0; channelIndex < c_channelsPerThread; ++channelIndex)
						{
							#pragma unroll
							for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
							{
								threadGradients[channelIndex][dataIndex] += filtersCache[channelIndex * c_blockHeight + threadIdx.y][currGradientFilter - currFilter + filterIndex] *
									gradientsCache[filterIndex][dataIndex * c_blockWidth + threadIdx.x];
							}
						}
					}

					__syncthreads();
				}
			}
		}
	}

	// Writing this thread calculated gradients into gradients buffer.
	#pragma unroll
	for (uint dataIndex = 0; dataIndex < c_dataPerThread; ++dataIndex)
	{
		if (!c_lastBatch || c_dataOffset + threadIdx.x + dataIndex * c_blockWidth < dataCount)
		{
			#pragma unroll
			for (uint channelIndex = 0; channelIndex < c_channelsPerThread; ++channelIndex)
			{
				inputGradients[channelIndex * c_blockHeight * dataSize * dataCount + dataIndex * c_blockWidth] = threadGradients[channelIndex][dataIndex];
			}
		}
	}
}

void ConvolutionalLayer::CalculateInputGradients()
{
	if (m_inputNumChannels < 3)
	{
		ShipAssert(false, "Currently not supported!");
	}
	else if (m_inputNumChannels == 3)
	{
		uint dataPerThread = m_inputDataCount % 128 == 0 ? 8 : (m_inputDataCount % 64 == 0 ? 4 : 2);
		uint blockWidth = 16;
		uint blockHeight = 16;
		// Block image patch size needs to be square root of block height!
		// Noted here as a constant to avoid sqrt computation, if we already hardcode blockHeight.
		uint blockImagePatchSize = 4;
		dim3 blockDimensions(blockWidth, blockHeight);
		dim3 gridDimensions(DivideUp(m_inputDataCount, blockWidth * dataPerThread), DivideUp(m_inputDataWidth, blockImagePatchSize) * DivideUp(m_inputDataHeight, blockImagePatchSize));
		bool lastBatch = m_inputDataCount % (blockWidth * dataPerThread) != 0;

		if (lastBatch)
		{
			LAUNCH_KERNEL_ASYNC((CalculateDataInputGradients<16, 2, 16, 32, 3, 4, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
				m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
				m_numFilters, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
		}
		else
		{
			if (m_inputDataCount % 128 == 0)
			{
				LAUNCH_KERNEL_ASYNC((CalculateDataInputGradients<16, 8, 16, 32, 3, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
					m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
					m_numFilters, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
			}
			else if (m_inputDataCount % 64 == 0)
			{
				LAUNCH_KERNEL_ASYNC((CalculateDataInputGradients<16, 4, 16, 32, 3, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
					m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
					m_numFilters, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
			}
			else
			{
				LAUNCH_KERNEL_ASYNC((CalculateDataInputGradients<16, 2, 16, 32, 3, 4, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
					m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
					m_numFilters, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
			}
		}
	}
	else if (m_inputNumChannels % 8 == 0)
	{
		uint dataPerThread = m_inputDataCount % 128 == 0 ? 4 : (m_inputDataCount % 64 == 0 ? 2 : 1);
		uint channelsPerThread = m_inputNumChannels % 64 == 0 ? 8 :
			(m_inputNumChannels % 48 == 0 ? 12 :
			(m_inputNumChannels % 32 == 0 ? 8 :
			(m_inputNumChannels % 16 == 0 ? 4 : 2)));
		uint blockWidth = 32;
		uint blockHeight = m_inputNumChannels % 64 == 0 ? 8 : 4;
		dim3 blockDimensions(blockWidth, blockHeight);
		dim3 gridDimensions(DivideUp(m_inputDataCount, blockWidth * dataPerThread) * (m_inputNumChannels / (blockHeight * channelsPerThread)),
			m_inputDataSize);
		bool lastBatch = m_inputDataCount % (blockWidth * dataPerThread) != 0;

		if (lastBatch)
		{
			if (m_inputNumChannels % 64 == 0)
			{
				if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 1, 8, 8, 32, 16, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
						m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 1, 8, 8, 16, 16, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
						m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
				}
			}
			else if (m_inputNumChannels % 48 == 0)
			{
				LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 1, 4, 12, 16, 16, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
					m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
					m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
			}
			else if (m_inputNumChannels % 32 == 0)
			{
				if (m_numFilters % 32 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 1, 4, 8, 32, 16, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
						m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 1, 4, 8, 16, 16, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
						m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
				}
			}
			else if (m_inputNumChannels % 16 == 0)
			{
				LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 1, 4, 4, 16, 16, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
					m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
					m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
			}
			else
			{
				LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 1, 4, 2, 16, 16, true>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
					m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
					m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
			}
		}
		else
		{
			if (m_inputNumChannels % 64 == 0)
			{
				if (m_numFilters % 32 == 0)
				{
					if (m_inputDataCount % 128 == 0)
					{
						LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 4, 8, 8, 32, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
							m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
							m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
					}
					else if (m_inputDataCount % 64 == 0)
					{
						LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 2, 8, 8, 32, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
							m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
							m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
					}
					else
					{
						LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 1, 8, 8, 32, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
							m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
							m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
					}
				}
				else
				{
					if (m_inputDataCount % 128 == 0)
					{
						LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 4, 8, 8, 16, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
							m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
							m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
					}
					else if (m_inputDataCount % 64 == 0)
					{
						LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 2, 8, 8, 16, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
							m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
							m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
					}
					else
					{
						LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 1, 8, 8, 16, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
							m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
							m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
					}
				}
			}
			else if (m_inputNumChannels % 48 == 0)
			{
				if (m_inputDataCount % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 4, 4, 12, 16, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
						m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
				}
				else if (m_inputDataCount % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 2, 4, 12, 16, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
						m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 1, 4, 12, 16, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
						m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
				}
			}
			else if (m_inputNumChannels % 32 == 0)
			{
				if (m_numFilters % 32 == 0)
				{
					if (m_inputDataCount % 128 == 0)
					{
						LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 4, 4, 8, 32, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
							m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
							m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
					}
					else if (m_inputDataCount % 64 == 0)
					{
						LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 2, 4, 8, 32, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
							m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
							m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
					}
					else
					{
						LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 1, 4, 8, 32, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
							m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
							m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
					}
				}
				else
				{
					if (m_inputDataCount % 128 == 0)
					{
						LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 4, 4, 8, 16, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
							m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
							m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
					}
					else if (m_inputDataCount % 64 == 0)
					{
						LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 2, 4, 8, 16, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
							m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
							m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
					}
					else
					{
						LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 1, 4, 8, 16, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
							m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
							m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
					}
				}
			}
			else if (m_inputNumChannels % 16 == 0)
			{
				if (m_inputDataCount % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 4, 4, 4, 16, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
						m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
				}
				else if (m_inputDataCount % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 2, 4, 4, 16, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
						m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 1, 4, 4, 16, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
						m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
				}
			}
			else
			{
				if (m_inputDataCount % 128 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 4, 4, 2, 16, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
						m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
				}
				else if (m_inputDataCount % 64 == 0)
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 2, 4, 2, 16, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
						m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
				}
				else
				{
					LAUNCH_KERNEL_ASYNC((CalculateFilteredInputGradients<32, 1, 4, 2, 16, 16, false>), gridDimensions, blockDimensions, m_deviceCalculationStream)(m_preactivationGradientsBuffer,
						m_inputDataWidth, m_inputDataHeight, m_inputDataSize, m_inputDataCount, m_paddingX, m_paddingY, m_filtersBuffer, m_filterWidth, m_filterHeight, m_filterSize,
						m_numFilters, m_numFilterChannels, m_stride, m_numPatchesX, m_numPatchesY, m_inputGradientsBuffer);
				}
			}
		}
	}
	else if (m_inputNumChannels % 4 == 0)
	{
		ShipAssert(false, "Currently not supported!");
	}

	CudaAssert(cudaGetLastError());
}

void ConvolutionalLayer::CalculatePreactivationsGradients()
{
	CalculatePreactivationGradients(m_activationType, m_activationGradientsBuffer, m_activationDataBuffer, (uint)(m_activationBufferSize / sizeof(float)),
		m_preactivationGradientsBuffer, m_deviceCalculationStream);
}

void ConvolutionalLayer::DoBackwardProp()
{
	CalculatePreactivationsGradients();
	CalculateInputGradients();
	CalculateWeightsGradients();
	CalculateBiasesGradients();
}

void ConvolutionalLayer::UpdateLayerParameters(float learningProgress)
{
	CommonUpdateLayerParameters(learningProgress, m_filtersBuffer, m_filtersGradientsBuffer, m_filtersUpdateBuffer, (uint)(m_filtersBufferSize / sizeof(float)),
		m_filtersUpdateMomentum, m_filtersUpdateLearningRateProgressStep, m_filtersUpdateStartingLearningRate, m_filtersUpdateLearningRateUpdateFactor,
		m_filtersUpdateDecay, m_biasesBuffer, m_biasesGradientsBuffer, m_biasesUpdateBuffer, (uint)(m_biasesBufferSize / sizeof(float)), m_biasesUpdateMomentum,
		m_biasesUpdateLearningRateProgressStep, m_biasesUpdateStartingLearningRate, m_biasesUpdateLearningRateUpdateFactor, m_biasesUpdateDecay);
}