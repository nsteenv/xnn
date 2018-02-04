// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network max pool layer, used in tests.
// Created: 02/07/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/mockmaxpoollayer.cuh"

MockMaxPoolLayer::MockMaxPoolLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint unitWidth,
	uint unitHeight, int paddingX, int paddingY, uint unitStride)
{
	m_layerType = LayerType::MaxPool;
	m_indexInTier = 0;
	m_tierSize = 1;

	m_inputNumChannels = inputNumChannels;
	m_inputDataWidth = inputDataWidth;
	m_inputDataHeight = inputDataHeight;
	m_inputDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = true;

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
		CudaAssert(cudaMallocHost<float>(&m_inputDataBuffer, m_inputBufferSize));
	}

	// Allocating input gradients buffer.
	CudaAssert(cudaMallocHost<float>(&m_inputGradientsBuffer, m_inputBufferSize));

	// Allocating activation data buffer.
	m_activationBufferSize = m_activationNumChannels * m_activationDataSize * m_inputDataCount * sizeof(float);
	CudaAssert(cudaMallocHost<float>(&m_activationDataBuffer, m_activationBufferSize));

	// Allocating activation gradients buffer.
	m_holdsActivationGradients = true;
	if (m_holdsActivationGradients)
	{
		CudaAssert(cudaMallocHost<float>(&m_activationGradientsBuffer, m_activationBufferSize));
	}
}

MockMaxPoolLayer::~MockMaxPoolLayer()
{
	if (m_holdsInputData)
	{
		CudaAssert(cudaFreeHost(m_inputDataBuffer));
	}
	m_inputDataBuffer = NULL;
	CudaAssert(cudaFreeHost(m_inputGradientsBuffer));
	m_inputGradientsBuffer = NULL;
	CudaAssert(cudaFreeHost(m_activationDataBuffer));
	m_activationDataBuffer = NULL;
	if (m_holdsActivationGradients)
	{
		CudaAssert(cudaFreeHost(m_activationGradientsBuffer));
	}
	m_activationGradientsBuffer = NULL;
}

void MockMaxPoolLayer::LoadInputs()
{
	TestingAssert(m_prevLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_inputDataBuffer, m_prevLayers[0]->GetActivationDataBuffer(), m_inputBufferSize, cudaMemcpyDeviceToHost));
}

void MockMaxPoolLayer::LoadActivationGradients()
{
	TestingAssert(m_nextLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_activationGradientsBuffer, m_nextLayers[0]->GetInputGradientsBuffer(), m_activationBufferSize, cudaMemcpyDeviceToHost));
}

void MockMaxPoolLayer::DoForwardProp(PropagationMode propagationMode)
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint channel = 0; channel < m_inputNumChannels; ++channel)
		{
			const uint c_activationChannelOffset = channel * m_activationDataSize * m_inputDataCount;
			const uint c_dataChannelOffset = channel * m_inputDataSize * m_inputDataCount;
			int startY = -m_paddingY;
			for (uint unitY = 0; unitY < m_numUnitsY; ++unitY)
			{
				int startX = -m_paddingX;
				for (uint unitX = 0; unitX < m_numUnitsX; ++unitX)
				{
					const uint c_activationDataIndex = c_activationChannelOffset + (unitY * m_numUnitsX + unitX) * m_inputDataCount + dataIndex;
					m_activationDataBuffer[c_activationDataIndex] = -FLT_MAX;
					for (int currY = startY; currY < startY + (int)m_unitHeight; ++currY)
					{
						for (int currX = startX; currX < startX + (int)m_unitWidth; ++currX)
						{
							if (currY >= 0 && currY < (int)m_inputDataHeight && currX >= 0 && currX < (int)m_inputDataWidth)
							{
								m_activationDataBuffer[c_activationDataIndex] = fmaxf(m_activationDataBuffer[c_activationDataIndex],
									m_inputDataBuffer[c_dataChannelOffset + (currY * m_inputDataWidth + currX) * m_inputDataCount + dataIndex]);
							}
						}
					}
					startX += m_unitStride;
				}
				startY += m_unitStride;
			}
		}
	}
}

void MockMaxPoolLayer::DoBackwardProp()
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint channel = 0; channel < m_inputNumChannels; ++channel)
		{
			const uint c_activationChannelOffset = channel * m_activationDataSize * m_inputDataCount + dataIndex;
			const uint c_dataChannelOffset = channel * m_inputDataSize * m_inputDataCount + dataIndex;
			for (uint pixelY = 0; pixelY < m_inputDataHeight; ++pixelY)			
			{
				for (uint pixelX = 0; pixelX < m_inputDataWidth; ++pixelX)
				{
					const uint c_dataPixelOffset = c_dataChannelOffset + (pixelY * m_inputDataWidth + pixelX) * m_inputDataCount;
					float data = m_inputDataBuffer[c_dataPixelOffset];

					// Calculating indexes of units whose activation this pixel affected.
					const uint c_firstUnitX = (uint)m_paddingX + pixelX < m_unitWidth ? 0 : ((uint)m_paddingX + pixelX - m_unitWidth) / m_unitStride + 1;
					const uint c_firstUnitY = (uint)m_paddingY + pixelY < m_unitHeight ? 0 : ((uint)m_paddingY + pixelY - m_unitHeight) / m_unitStride + 1;
					const uint c_lastUnitX = min(m_numUnitsX, ((uint)m_paddingX + pixelX) / m_unitStride + 1);
					const uint c_lastUnitY = min(m_numUnitsY, ((uint)m_paddingY + pixelY) / m_unitStride + 1);
					
					// Calculating pixel gradient.
					float calculatedGradient = 0.f;
					for (uint unitY = c_firstUnitY; unitY < c_lastUnitY; ++unitY)
					{
						for (uint unitX = c_firstUnitX; unitX < c_lastUnitX; ++unitX)
						{
							const uint c_unitOffset = c_activationChannelOffset + (unitY * m_numUnitsX + unitX) * m_inputDataCount;
							float activation = m_activationDataBuffer[c_unitOffset];
							float activationGradient = m_activationGradientsBuffer[c_unitOffset];

							calculatedGradient += fabs(data - activation) < 0.000000001f ? activationGradient : 0.f;
						}
					}

					m_inputGradientsBuffer[c_dataPixelOffset] = calculatedGradient;
				}
			}
		}
	}
}