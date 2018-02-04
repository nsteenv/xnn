// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network convolutional layer, used in tests.
// Created: 01/27/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/mockconvolutionallayer.cuh"

MockConvolutionalLayer::MockConvolutionalLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numFilters, uint filterWidth,
	uint filterHeight, uint numFilterChannels, float weightsDeviation, float biasesInitialValue, float filtersUpdateMomentum, float filtersUpdateDecay,
	float filtersUpdateLearningRateProgressStep, float filtersUpdateStartingLearningRate, float filtersUpdateLearningRateUpdateFactor, float biasesUpdateMomentum,
	float biasesUpdateDecay, float biasesUpdateLearningRateProgressStep, float biasesUpdateStartingLearningRate, float biasesUpdateLearningRateUpdateFactor,
	int paddingX, int paddingY, uint stride, ActivationType activationType)
{
	m_layerType = LayerType::Convolutional;
	m_indexInTier = 0;
	m_tierSize = 1;

	m_inputNumChannels = inputNumChannels;
	m_inputDataWidth = inputDataWidth;
	m_inputDataHeight = inputDataHeight;
	m_inputDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = true;
	m_activationType = activationType;

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
		CudaAssert(cudaMallocHost<float>(&m_inputDataBuffer, m_inputBufferSize));
	}

	// Allocating input gradients buffer.
	CudaAssert(cudaMallocHost<float>(&m_inputGradientsBuffer, m_inputBufferSize));

	// Allocating filters buffers.
	m_filtersBufferSize = m_numFilters * m_filterSize * m_numFilterChannels * sizeof(float);
	CudaAssert(cudaMallocHost<float>(&m_filtersBuffer, m_filtersBufferSize));
	CudaAssert(cudaMallocHost<float>(&m_filtersGradientsBuffer, m_filtersBufferSize));
	CudaAssert(cudaMallocHost<float>(&m_filtersUpdateBuffer, m_filtersBufferSize));

	// Initializing filter weights.
	InitializeFilterWeights(weightsDeviation);
	InitializeBuffer(m_filtersUpdateBuffer, m_filtersBufferSize, 0.f);

	// Allocating biases buffers.
	m_biasesBufferSize = m_numFilters * sizeof(float);
	CudaAssert(cudaMallocHost<float>(&m_biasesBuffer, m_biasesBufferSize));
	CudaAssert(cudaMallocHost<float>(&m_biasesGradientsBuffer, m_biasesBufferSize));
	CudaAssert(cudaMallocHost<float>(&m_biasesUpdateBuffer, m_biasesBufferSize));

	// Initializing biases.
	InitializeBuffer(m_biasesBuffer, m_biasesBufferSize, biasesInitialValue);
	InitializeBuffer(m_biasesUpdateBuffer, m_biasesBufferSize, 0.f);

	// Allocating preactivation and activation data buffers.
	m_activationBufferSize = m_numFilters * m_activationDataSize * m_inputDataCount * sizeof(float);
	CudaAssert(cudaMallocHost<float>(&m_preactivationDataBuffer, m_activationBufferSize));
	CudaAssert(cudaMallocHost<float>(&m_activationDataBuffer, m_activationBufferSize));

	// Allocating preactivation gradients buffer.
	CudaAssert(cudaMallocHost<float>(&m_preactivationGradientsBuffer, m_activationBufferSize));

	// Allocating activation gradients buffer.
	m_holdsActivationGradients = true;
	if (m_holdsActivationGradients)
	{
		CudaAssert(cudaMallocHost<float>(&m_activationGradientsBuffer, m_activationBufferSize));
	}
}

void MockConvolutionalLayer::InitializeFilterWeights(float weightsDeviation)
{
	default_random_engine generator((uint)chrono::system_clock::now().time_since_epoch().count());
	normal_distribution<float> distribution(0.f, weightsDeviation);

	size_t filtersBufferLength = m_filtersBufferSize / sizeof(float);
	for (size_t i = 0; i < filtersBufferLength; ++i)
	{
		m_filtersBuffer[i] = distribution(generator);
	}
}

void MockConvolutionalLayer::InitializeBuffer(float* buffer, size_t bufferSize, float initialValue)
{
	size_t bufferLength = bufferSize / sizeof(float);
	for (size_t i = 0; i < bufferLength; ++i)
	{
		buffer[i] = initialValue;
	}
}

MockConvolutionalLayer::~MockConvolutionalLayer()
{
	if (m_holdsInputData)
	{
		CudaAssert(cudaFreeHost(m_inputDataBuffer));
	}
	m_inputDataBuffer = NULL;
	CudaAssert(cudaFreeHost(m_inputGradientsBuffer));
	m_inputGradientsBuffer = NULL;

	CudaAssert(cudaFreeHost(m_filtersBuffer));
	CudaAssert(cudaFreeHost(m_filtersGradientsBuffer));
	CudaAssert(cudaFreeHost(m_filtersUpdateBuffer));

	CudaAssert(cudaFreeHost(m_biasesBuffer));
	CudaAssert(cudaFreeHost(m_biasesGradientsBuffer));
	CudaAssert(cudaFreeHost(m_biasesUpdateBuffer));

	CudaAssert(cudaFreeHost(m_preactivationDataBuffer));
	CudaAssert(cudaFreeHost(m_activationDataBuffer));
	m_activationDataBuffer = NULL;

	CudaAssert(cudaFreeHost(m_preactivationGradientsBuffer));
	if (m_holdsActivationGradients)
	{
		CudaAssert(cudaFreeHost(m_activationGradientsBuffer));
	}
	m_activationGradientsBuffer = NULL;
}

void MockConvolutionalLayer::LoadInputs()
{
	TestingAssert(m_prevLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_inputDataBuffer, m_prevLayers[0]->GetActivationDataBuffer(), m_inputBufferSize, cudaMemcpyDeviceToHost));
}

void MockConvolutionalLayer::LoadActivationGradients()
{
	TestingAssert(m_nextLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_activationGradientsBuffer, m_nextLayers[0]->GetInputGradientsBuffer(), m_activationBufferSize, cudaMemcpyDeviceToHost));
}

void MockConvolutionalLayer::CalculatePreactivations()
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint filterIndex = 0; filterIndex < m_numFilters; ++filterIndex)
		{
			const uint c_activationChannelOffset = m_activationDataSize * filterIndex * m_inputDataCount;
			for (uint channel = 0; channel < m_inputNumChannels; ++channel)
			{
				const uint c_filtersChannelOffset = channel * m_numFilters * m_filterSize;
				const uint c_dataChannelOffset = channel * m_inputDataCount * m_inputDataSize;
				int startY = -m_paddingY;
				for (uint patchY = 0; patchY < m_numPatchesY; ++patchY)
				{
					int startX = -m_paddingX;
					for (uint patchX = 0; patchX < m_numPatchesX; ++patchX)
					{
						const uint c_activationDataIndex = c_activationChannelOffset + (patchY * m_numPatchesX + patchX) * m_inputDataCount + dataIndex;
						if (channel == 0)
						{
							m_preactivationDataBuffer[c_activationDataIndex] = 0.0f;
						}
						for (int currY = startY; currY < startY + (int)m_filterHeight; ++currY)
						{
							for (int currX = startX; currX < startX + (int)m_filterWidth; ++currX)
							{
								if (currY >= 0 && currY < (int)m_inputDataHeight && currX >= 0 && currX < (int)m_inputDataWidth)
								{
									m_preactivationDataBuffer[c_activationDataIndex] +=
										m_filtersBuffer[c_filtersChannelOffset + ((currY - startY) * m_filterWidth + currX - startX) * m_numFilters + filterIndex] *
										m_inputDataBuffer[c_dataChannelOffset + (currY * m_inputDataWidth + currX) * m_inputDataCount + dataIndex];
								}
							}
						}
						startX += m_stride;
					}
					startY += m_stride;
				}
			}
		}
	}
}

void MockConvolutionalLayer::AddBiases()
{
	const uint c_width = m_inputDataCount * m_numPatchesY * m_numPatchesX;
	for (uint filterIndex = 0; filterIndex < m_numFilters; ++filterIndex)
	{
		for (uint i = 0; i < c_width; ++i)
		{
			m_preactivationDataBuffer[filterIndex * c_width + i] += m_biasesBuffer[filterIndex];
		}
	}
}

void MockConvolutionalLayer::CalculateActivations()
{
	for (uint i = 0; i < m_activationBufferSize / sizeof(float); ++i)
	{
		if (m_activationType == ActivationType::ReLu)
		{
			m_activationDataBuffer[i] = m_preactivationDataBuffer[i] < 0.0f ? 0.0f : m_preactivationDataBuffer[i];
		}
		else if (m_activationType == ActivationType::Sigmoid)
		{
			m_activationDataBuffer[i] = 1 / (1 + exp(-m_preactivationDataBuffer[i]));
		}
		else if (m_activationType == ActivationType::Tanh)
		{
			m_activationDataBuffer[i] = 1.0f - 2.0f / (exp(2.0f * m_preactivationDataBuffer[i]) + 1.0f);
		}
		else
		{
			TestingAssert(false, "Unknown activation type!");
		}
	}
}

void MockConvolutionalLayer::DoForwardProp(PropagationMode propagationMode)
{
	CalculatePreactivations();
	AddBiases();
	CalculateActivations();
}

void MockConvolutionalLayer::CalculateBiasesGradients()
{
	uint batchSize = m_parallelismMode == ParallelismMode::Model ? m_inputDataCount : m_tierSize * m_inputDataCount;
	const uint c_width = m_inputDataCount * m_numPatchesY * m_numPatchesX;
	for (uint filterIndex = 0; filterIndex < m_numFilters; ++filterIndex)
	{
		float biasGradient = 0.f;
		for (uint i = 0; i < c_width; ++i)
		{
			biasGradient += m_preactivationGradientsBuffer[filterIndex * c_width + i];
		}

		m_biasesGradientsBuffer[filterIndex] = biasGradient / (float)batchSize;
	}
}

void MockConvolutionalLayer::CalculateWeightsGradients()
{
	// Initializing gradients to zero.
	size_t filtersBufferLength = m_filtersBufferSize / sizeof(float);
	for (size_t i = 0; i < filtersBufferLength; ++i)
	{
		m_filtersGradientsBuffer[i] = 0.f;
	}

	// Calculating gradients.
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint filterIndex = 0; filterIndex < m_numFilters; ++filterIndex)
		{
			const uint c_activationChannelOffset = m_activationDataSize * filterIndex * m_inputDataCount;
			for (uint channel = 0; channel < m_inputNumChannels; ++channel)
			{
				const uint c_filtersChannelOffset = channel * m_numFilters * m_filterSize;
				const uint c_dataChannelOffset = channel * m_inputDataCount * m_inputDataSize;
				int startY = -m_paddingY;
				for (uint patchY = 0; patchY < m_numPatchesY; ++patchY)
				{
					int startX = -m_paddingX;
					for (uint patchX = 0; patchX < m_numPatchesX; ++patchX)
					{
						const uint c_activationDataIndex = c_activationChannelOffset + (patchY * m_numPatchesX + patchX) * m_inputDataCount + dataIndex;
						if (channel == 0)
						{
							m_preactivationDataBuffer[c_activationDataIndex] = 0.0f;
						}
						for (int currY = startY; currY < startY + (int)m_filterHeight; ++currY)
						{
							for (int currX = startX; currX < startX + (int)m_filterWidth; ++currX)
							{
								if (currY >= 0 && currY < (int)m_inputDataHeight && currX >= 0 && currX < (int)m_inputDataWidth)
								{
									m_filtersGradientsBuffer[c_filtersChannelOffset + ((currY - startY) * m_filterWidth + currX - startX) * m_numFilters + filterIndex] +=
										m_preactivationGradientsBuffer[c_activationDataIndex] *
										m_inputDataBuffer[c_dataChannelOffset + (currY * m_inputDataWidth + currX) * m_inputDataCount + dataIndex];
								}
							}
						}
						startX += (int)m_stride;
					}
					startY += (int)m_stride;
				}
			}
		}
	}

	// Scaling gradients with batch size.
	float batchSize = m_parallelismMode == ParallelismMode::Model ? (float)m_inputDataCount : (float)(m_tierSize * m_inputDataCount);
	for (size_t i = 0; i < filtersBufferLength; ++i)
	{
		m_filtersGradientsBuffer[i] /= batchSize;
	}
}

void MockConvolutionalLayer::CalculateInputGradients()
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint channel = 0; channel < m_inputNumChannels; ++channel)
		{
			for (uint pixelY = 0; pixelY < m_inputDataHeight; ++pixelY)
			{
				for (uint pixelX = 0; pixelX < m_inputDataWidth; ++pixelX)
				{
					const uint c_currPixel = pixelY * m_inputDataWidth + pixelX;
					const uint c_firstPatchX = pixelX + m_paddingX < m_filterWidth ? 0 : (pixelX + m_paddingX - m_filterWidth) / m_stride + 1;
					const uint c_firstPatchY = pixelY + m_paddingY < m_filterHeight ? 0 : (pixelY + m_paddingY - m_filterHeight) / m_stride + 1;
					const uint c_lastPatchX = min(m_numPatchesX, (pixelX + m_paddingX) / m_stride + 1);
					const uint c_lastPatchY = min(m_numPatchesY, (pixelY + m_paddingY) / m_stride + 1);

					float gradient = 0.0f;

					for (uint currPatchY = c_firstPatchY; currPatchY < c_lastPatchY; ++currPatchY)
					{
						const uint c_filterPixelY = pixelY + m_paddingY - currPatchY * m_stride;
						for (uint currPatchX = c_firstPatchX; currPatchX < c_lastPatchX; ++currPatchX)
						{
							const uint c_filterPixelX = pixelX + m_paddingX - currPatchX * m_stride;
							const uint c_filterPixel = c_filterPixelY * m_filterWidth + c_filterPixelX;
							const uint c_currPatch = currPatchY * m_numPatchesX + currPatchX;

							for (uint currFilter = 0; currFilter < m_numFilters; ++currFilter)
							{
								gradient += m_filtersBuffer[(channel * m_filterSize + c_filterPixel) * m_numFilters + currFilter] *
									m_preactivationGradientsBuffer[(currFilter * m_numPatchesX * m_numPatchesY + c_currPatch) * m_inputDataCount + dataIndex];
							}
						}
					}

					m_inputGradientsBuffer[(channel * m_inputDataSize + c_currPixel) * m_inputDataCount + dataIndex] = gradient;
				}
			}
		}
	}
}

void MockConvolutionalLayer::CalculatePreactivationsGradients()
{
	for (uint i = 0; i < m_activationBufferSize / sizeof(float); ++i)
	{
		if (m_activationType == ActivationType::ReLu)
		{
			m_preactivationGradientsBuffer[i] = m_activationGradientsBuffer[i] * (m_activationDataBuffer[i] > 0.0f ? 1.0f : 0.0f);
		}
		else if (m_activationType == ActivationType::Sigmoid)
		{
			m_preactivationGradientsBuffer[i] = m_activationGradientsBuffer[i] * m_activationDataBuffer[i] * (1.0f - m_activationDataBuffer[i]);
		}
		else if (m_activationType == ActivationType::Tanh)
		{
			m_preactivationGradientsBuffer[i] = m_activationGradientsBuffer[i] * (1.0f - m_activationDataBuffer[i] * m_activationDataBuffer[i]);
		}
		else
		{
			TestingAssert(false, "Unknown activation type!");
		}
	}
}

void MockConvolutionalLayer::DoBackwardProp()
{
	CalculatePreactivationsGradients();
	CalculateInputGradients();
	CalculateWeightsGradients();
	CalculateBiasesGradients();
}

void MockConvolutionalLayer::UpdateLayerParameters(float learningProgress)
{
	// Updating filters.
	float filtersUpdateProgressSteps = floorf(learningProgress / m_filtersUpdateLearningRateProgressStep);
	const float filtersLearningRate = m_filtersUpdateStartingLearningRate * powf(m_filtersUpdateLearningRateUpdateFactor, filtersUpdateProgressSteps);
	for (uint i = 0; i < m_filtersBufferSize / sizeof(float); ++i)
	{
		m_filtersUpdateBuffer[i] = m_filtersUpdateMomentum * m_filtersUpdateBuffer[i] + filtersLearningRate * (m_filtersGradientsBuffer[i] -
			m_filtersUpdateDecay * m_filtersBuffer[i]);
		m_filtersBuffer[i] += m_filtersUpdateBuffer[i];
	}

	// Updating biases.
	float biasesUpdateProgressSteps = floorf(learningProgress / m_biasesUpdateLearningRateProgressStep);
	const float biasesLearningRate = m_biasesUpdateStartingLearningRate * powf(m_biasesUpdateLearningRateUpdateFactor, biasesUpdateProgressSteps);
	for (uint i = 0; i < m_biasesBufferSize / sizeof(float); ++i)
	{
		m_biasesUpdateBuffer[i] = m_biasesUpdateMomentum * m_biasesUpdateBuffer[i] + biasesLearningRate * (m_biasesGradientsBuffer[i] -
			m_biasesUpdateDecay * m_biasesBuffer[i]);
		m_biasesBuffer[i] += m_biasesUpdateBuffer[i];
	}
}