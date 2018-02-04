// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network dropout layer, used in tests.
// Created: 02/16/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/mockdropoutlayer.cuh"

MockDropoutLayer::MockDropoutLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, float dropProbability)
{
	m_layerType = LayerType::Dropout;
	m_indexInTier = 0;
	m_tierSize = 1;

	m_inputNumChannels = m_activationNumChannels = inputNumChannels;
	m_inputDataWidth = m_activationDataWidth = inputDataWidth;
	m_inputDataHeight = m_activationDataHeight = inputDataHeight;
	m_inputDataSize = m_activationDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = true;

	m_dropProbability = dropProbability;

	// Allocating input data buffer.
	m_inputBufferSize = m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);
	if (m_holdsInputData)
	{
		CudaAssert(cudaMallocHost<float>(&m_inputDataBuffer, m_inputBufferSize));
	}

	// Allocating input gradients buffer.
	CudaAssert(cudaMallocHost<float>(&m_inputGradientsBuffer, m_inputBufferSize));

	// Allocating dropout filter buffer.
	m_dropoutFilterSize = m_inputBufferSize;
	CudaAssert(cudaMallocHost<float>(&m_dropoutFilter, m_dropoutFilterSize));

	// Allocating activation data buffers.
	m_activationBufferSize = m_inputBufferSize;
	CudaAssert(cudaMallocHost<float>(&m_activationDataBuffer, m_activationBufferSize));

	// Allocating activation gradients buffer.
	m_holdsActivationGradients = true;
	if (m_holdsActivationGradients)
	{
		CudaAssert(cudaMallocHost<float>(&m_activationGradientsBuffer, m_activationBufferSize));
	}
}

MockDropoutLayer::~MockDropoutLayer()
{
	if (m_holdsInputData)
	{
		CudaAssert(cudaFreeHost(m_inputDataBuffer));
	}
	m_inputDataBuffer = NULL;
	CudaAssert(cudaFreeHost(m_inputGradientsBuffer));
	m_inputGradientsBuffer = NULL;
	CudaAssert(cudaFreeHost(m_dropoutFilter));
	CudaAssert(cudaFreeHost(m_activationDataBuffer));
	m_activationDataBuffer = NULL;
	if (m_holdsActivationGradients)
	{
		CudaAssert(cudaFreeHost(m_activationGradientsBuffer));
	}
	m_activationGradientsBuffer = NULL;
}

void MockDropoutLayer::LoadInputs()
{
	TestingAssert(m_prevLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_inputDataBuffer, m_prevLayers[0]->GetActivationDataBuffer(), m_inputBufferSize, cudaMemcpyDeviceToHost));
}

void MockDropoutLayer::LoadActivationGradients()
{
	TestingAssert(m_nextLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");

	CudaAssert(cudaMemcpy(m_activationGradientsBuffer, m_nextLayers[0]->GetInputGradientsBuffer(), m_activationBufferSize, cudaMemcpyDeviceToHost));
}

void MockDropoutLayer::CreateDropoutFilter()
{
	// Filling dropout filter with random values.
	default_random_engine generator((uint)chrono::system_clock::now().time_since_epoch().count());
	uniform_real_distribution<float> distribution(0.f, 1.f);

	size_t dropoutFilterLength = m_dropoutFilterSize / sizeof(float);
	for (size_t i = 0; i < dropoutFilterLength; ++i)
	{
		m_dropoutFilter[i] = distribution(generator);
	}

	// Dropping filter values which are not above the drop probability.
	for (size_t i = 0; i < dropoutFilterLength; ++i)
	{
		m_dropoutFilter[i] = m_dropoutFilter[i] > m_dropProbability ? 1.0f : 0.0f;
	}
}

void MockDropoutLayer::ApplyDropoutFilter()
{
	size_t dropoutFilterLength = m_dropoutFilterSize / sizeof(float);
	for (size_t i = 0; i < dropoutFilterLength; ++i)
	{
		m_activationDataBuffer[i] = m_inputDataBuffer[i] * m_dropoutFilter[i];
	}
}

void MockDropoutLayer::DoForwardProp(PropagationMode propagationMode)
{
	CreateDropoutFilter();
	ApplyDropoutFilter();
}

void MockDropoutLayer::DoBackwardProp()
{
	size_t dropoutFilterLength = m_dropoutFilterSize / sizeof(float);
	for (size_t i = 0; i < dropoutFilterLength; ++i)
	{
		m_inputGradientsBuffer[i] = m_activationGradientsBuffer[i] * m_dropoutFilter[i];
	}
}