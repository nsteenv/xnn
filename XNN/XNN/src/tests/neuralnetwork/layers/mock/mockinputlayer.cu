// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network input layer, used in tests.
// Created: 01/24/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/mockinputlayer.cuh"

MockInputLayer::MockInputLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, float dataScale)
{
	// Hack to avoid casting to InputLayer inside layers CommonLoadInputs function.
	m_layerType = LayerType::Standard;
	m_indexInTier = 0;
	m_tierSize = 1;
	m_dataScale = dataScale;

	m_inputNumChannels = m_activationNumChannels = inputNumChannels;
	m_inputDataWidth = m_activationDataWidth = inputDataWidth;
	m_inputDataHeight = m_activationDataHeight = inputDataHeight;
	m_inputDataSize = m_activationDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = true;

	// Allocating input data buffer.
	m_inputBufferSize = m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);
	CudaAssert(cudaMallocHost<float>(&m_inputDataBuffer, m_inputBufferSize));

	// Allocating activation data buffer.
	m_activationBufferSize = m_inputBufferSize;
	CudaAssert(cudaMalloc<float>(&m_activationDataBuffer, m_activationBufferSize));

	m_holdsActivationGradients = false;
}

MockInputLayer::~MockInputLayer()
{
	CudaAssert(cudaFreeHost(m_inputDataBuffer));
	m_inputDataBuffer = NULL;
	m_inputGradientsBuffer = NULL;
	m_activationGradientsBuffer = NULL;
}

void MockInputLayer::LoadInputs()
{
	default_random_engine generator((uint)chrono::system_clock::now().time_since_epoch().count());
	uniform_int_distribution<int> distribution(0, 255);

	size_t inputBufferLength = m_inputBufferSize / sizeof(float);
	for (size_t i = 0; i < inputBufferLength; ++i)
	{
		m_inputDataBuffer[i] = m_dataScale * ((float)distribution(generator) - 128.f);
	}
}

void MockInputLayer::DoForwardProp(PropagationMode propagationMode)
{
	CudaAssert(cudaMemcpy(m_activationDataBuffer, m_inputDataBuffer, m_inputBufferSize, cudaMemcpyHostToDevice));
}

void MockInputLayer::DoBackwardProp()
{
	ShipAssert(false, "Shouldn't backpropagate on input layer!");
}