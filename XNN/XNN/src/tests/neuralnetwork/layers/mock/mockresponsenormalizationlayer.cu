// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network response normalization layer, used in tests.
// Created: 02/09/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/mockresponsenormalizationlayer.cuh"

MockResponseNormalizationLayer::MockResponseNormalizationLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount,
	uint depth, float bias, float alphaCoeff, float betaCoeff)
{
	m_layerType = LayerType::ResponseNormalization;
	m_indexInTier = 0;
	m_tierSize = 1;

	m_inputNumChannels = m_activationNumChannels = inputNumChannels;
	m_inputDataWidth = m_activationDataWidth = inputDataWidth;
	m_inputDataHeight = m_activationDataHeight = inputDataHeight;
	m_inputDataSize = m_activationDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = true;

	m_depth = depth;
	m_bias = bias;
	// Adjusting alpha coefficient upfront, according to formula.
	m_alphaCoeff = alphaCoeff / (float)depth;
	m_betaCoeff = betaCoeff;

	// Allocating input data buffer.
	m_inputBufferSize = m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);
	if (m_holdsInputData)
	{
		CudaAssert(cudaMallocHost<float>(&m_inputDataBuffer, m_inputBufferSize));
	}

	// Allocating input gradients buffer.
	CudaAssert(cudaMallocHost<float>(&m_inputGradientsBuffer, m_inputBufferSize));

	// Allocating activation data buffer.
	m_activationBufferSize = m_inputBufferSize;
	CudaAssert(cudaMallocHost<float>(&m_activationDataBuffer, m_activationBufferSize));

	// Allocating activation gradients buffer.
	m_holdsActivationGradients = true;
	if (m_holdsActivationGradients)
	{
		CudaAssert(cudaMallocHost<float>(&m_activationGradientsBuffer, m_activationBufferSize));
	}
}

MockResponseNormalizationLayer::~MockResponseNormalizationLayer()
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

void MockResponseNormalizationLayer::LoadInputs()
{
	TestingAssert(m_prevLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_inputDataBuffer, m_prevLayers[0]->GetActivationDataBuffer(), m_inputBufferSize, cudaMemcpyDeviceToHost));
}

void MockResponseNormalizationLayer::LoadActivationGradients()
{
	TestingAssert(m_nextLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_activationGradientsBuffer, m_nextLayers[0]->GetInputGradientsBuffer(), m_activationBufferSize, cudaMemcpyDeviceToHost));
}

void MockResponseNormalizationLayer::DoForwardProp(PropagationMode propagationMode)
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint pixelY = 0; pixelY < m_inputDataHeight; ++pixelY)
		{
			for (uint pixelX = 0; pixelX < m_inputDataWidth; ++pixelX)
			{
				const uint c_pixelOffset = (pixelY * m_inputDataWidth + pixelX) * m_inputDataCount + dataIndex;
				float crossChannelSum = 0.f;
				for (uint channel = 0; channel < m_inputNumChannels; ++channel)
				{					
					const uint c_bufferOffset = channel * m_inputDataSize * m_inputDataCount + c_pixelOffset;
					if (channel == 0)
					{
						const int c_actualStartChannel = (int)channel - (int)m_depth / 2;
						const int c_startChannel = max(c_actualStartChannel, 0);
						const int c_endChannel = min(c_actualStartChannel + (int)m_depth, (int)m_inputNumChannels);
						for (int currChannel = c_startChannel; currChannel < c_endChannel; ++currChannel)
						{
							const float c_channelData = m_inputDataBuffer[currChannel * m_inputDataSize * m_inputDataCount + c_pixelOffset];
							crossChannelSum += c_channelData * c_channelData;
						}
					}
					else
					{
						const int c_channelToSubtract = (int)channel - (int)m_depth / 2 - 1;
						const int c_channelToAdd = c_channelToSubtract + (int)m_depth;
						if (c_channelToSubtract >= 0)
						{
							const float c_channelToSubtractData = m_inputDataBuffer[c_channelToSubtract * m_inputDataSize * m_inputDataCount + c_pixelOffset];
							crossChannelSum -= c_channelToSubtractData * c_channelToSubtractData;
						}
						if (c_channelToAdd < (int)m_inputNumChannels)
						{
							const float c_channelToAddData = m_inputDataBuffer[c_channelToAdd * m_inputDataSize * m_inputDataCount + c_pixelOffset];
							crossChannelSum += c_channelToAddData * c_channelToAddData;
						}
					}
					m_activationDataBuffer[c_bufferOffset] = m_inputDataBuffer[c_bufferOffset] * powf(m_bias + m_alphaCoeff * crossChannelSum, -m_betaCoeff);
				}
			}
		}
	}
}

void MockResponseNormalizationLayer::DoBackwardProp()
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint pixelY = 0; pixelY < m_inputDataHeight; ++pixelY)
		{
			for (uint pixelX = 0; pixelX < m_inputDataWidth; ++pixelX)
			{
				const uint c_pixelOffset = (pixelY * m_inputDataWidth + pixelX) * m_inputDataCount + dataIndex;
				float crossChannelSum = 0.f;
				for (uint channel = 0; channel < m_inputNumChannels; ++channel)
				{					
					if (channel == 0)
					{
						const int c_actualStartChannel = (int)channel - (int)m_depth + (int)m_depth / 2 + 1;
						const int c_startChannel = max(c_actualStartChannel, 0);
						const int c_endChannel = min(c_actualStartChannel + (int)m_depth, (int)m_inputNumChannels);
						for (int currChannel = c_startChannel; currChannel < c_endChannel; ++currChannel)
						{
							const uint c_position = currChannel * m_inputDataSize * m_inputDataCount + c_pixelOffset;
							float data = m_inputDataBuffer[c_position];
							float activation = m_activationDataBuffer[c_position];
							float activationGradient = m_activationGradientsBuffer[c_position];
							crossChannelSum += activationGradient * activation * (data == 0.f ? 0.f : powf(activation / data, 1.0f / m_betaCoeff));
						}
					}
					else
					{
						const int c_channelToSubtract = (int)channel - (int)m_depth + (int)m_depth / 2;
						const int c_channelToAdd = c_channelToSubtract + (int)m_depth;
						if (c_channelToSubtract >= 0)
						{
							const uint c_position = c_channelToSubtract * m_inputDataSize * m_inputDataCount + c_pixelOffset;
							float data = m_inputDataBuffer[c_position];
							float activation = m_activationDataBuffer[c_position];
							float activationGradient = m_activationGradientsBuffer[c_position];
							crossChannelSum -= activationGradient * activation * (data == 0.f ? 0.f : powf(activation / data, 1.0f / m_betaCoeff));
						}
						if (c_channelToAdd < (int)m_inputNumChannels)
						{
							const uint c_position = c_channelToAdd * m_inputDataSize * m_inputDataCount + c_pixelOffset;
							float data = m_inputDataBuffer[c_position];
							float activation = m_activationDataBuffer[c_position];
							float activationGradient = m_activationGradientsBuffer[c_position];
							crossChannelSum += activationGradient * activation * (data == 0.f ? 0.f : powf(activation / data, 1.0f / m_betaCoeff));
						}
					}

					const uint c_bufferOffset = channel * m_inputDataSize * m_inputDataCount + c_pixelOffset;
					float channelData = m_inputDataBuffer[c_bufferOffset];
					float channelActivation = m_activationDataBuffer[c_bufferOffset];
					float channelActivationGradient = m_activationGradientsBuffer[c_bufferOffset];
					m_inputGradientsBuffer[c_bufferOffset] = -2.0f * m_alphaCoeff * m_betaCoeff * channelData * crossChannelSum +
						channelActivationGradient * (channelData == 0.f ? 0.f : channelActivation / channelData);
				}
			}
		}
	}
}