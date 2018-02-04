// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network standard layer, used in tests.
// Created: 02/13/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/mockstandardlayer.cuh"

MockStandardLayer::MockStandardLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numNeurons, float weightsDeviation,
	float biasesInitialValue, float weightsUpdateMomentum, float weightsUpdateDecay, float weightsUpdateLearningRateProgressStep, float weightsUpdateStartingLearningRate,
	float weightsUpdateLearningRateUpdateFactor, float biasesUpdateMomentum, float biasesUpdateDecay, float biasesUpdateLearningRateProgressStep,
	float biasesUpdateStartingLearningRate, float biasesUpdateLearningRateUpdateFactor, ActivationType activationType, bool outputToGpuMemory)
{
	m_layerType = LayerType::Standard;
	m_indexInTier = 0;
	m_tierSize = 1;

	m_inputNumChannels = inputNumChannels;
	m_inputDataWidth = inputDataWidth;
	m_inputDataHeight = inputDataHeight;
	m_inputDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = true;
	m_activationType = activationType;
	m_outputToGpuMemory = outputToGpuMemory;

	m_numNeurons = numNeurons;
	m_numWeightsPerNeuron = m_inputNumChannels * m_inputDataSize;

	m_weightsUpdateMomentum = weightsUpdateMomentum;
	m_weightsUpdateDecay = weightsUpdateDecay;
	m_weightsUpdateLearningRateProgressStep = weightsUpdateLearningRateProgressStep;
	m_weightsUpdateStartingLearningRate = weightsUpdateStartingLearningRate;
	m_weightsUpdateLearningRateUpdateFactor = weightsUpdateLearningRateUpdateFactor;

	m_biasesUpdateMomentum = biasesUpdateMomentum;
	m_biasesUpdateDecay = biasesUpdateDecay;
	m_biasesUpdateLearningRateProgressStep = biasesUpdateLearningRateProgressStep;
	m_biasesUpdateStartingLearningRate = biasesUpdateStartingLearningRate;
	m_biasesUpdateLearningRateUpdateFactor = biasesUpdateLearningRateUpdateFactor;

	m_activationNumChannels = 1;
	m_activationDataWidth = m_numNeurons;
	m_activationDataHeight = 1;
	m_activationDataSize = m_activationDataWidth * m_activationDataHeight;

	// Allocating input data buffer.
	m_inputBufferSize = m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);
	if (m_holdsInputData)
	{
		CudaAssert(cudaMallocHost<float>(&m_inputDataBuffer, m_inputBufferSize));
	}

	// Allocating input gradients buffer.
	CudaAssert(cudaMallocHost<float>(&m_inputGradientsBuffer, m_inputBufferSize));

	// Allocating weights buffer.
	m_weightsBufferSize = m_numNeurons * m_numWeightsPerNeuron * sizeof(float);
	CudaAssert(cudaMallocHost<float>(&m_weightsBuffer, m_weightsBufferSize));
	CudaAssert(cudaMallocHost<float>(&m_weightsGradientsBuffer, m_weightsBufferSize));
	CudaAssert(cudaMallocHost<float>(&m_weightsUpdateBuffer, m_weightsBufferSize));

	// Initializing weights.
	InitializeWeights(weightsDeviation);
	InitializeBuffer(m_weightsUpdateBuffer, m_weightsBufferSize, 0.f);

	// Allocating biases buffer.
	m_biasesBufferSize = m_numNeurons * sizeof(float);
	CudaAssert(cudaMallocHost<float>(&m_biasesBuffer, m_biasesBufferSize));
	CudaAssert(cudaMallocHost<float>(&m_biasesGradientsBuffer, m_biasesBufferSize));
	CudaAssert(cudaMallocHost<float>(&m_biasesUpdateBuffer, m_biasesBufferSize));

	// Initializing biases.
	InitializeBuffer(m_biasesBuffer, m_biasesBufferSize, biasesInitialValue);
	InitializeBuffer(m_biasesUpdateBuffer, m_biasesBufferSize, 0.f);

	// Allocating preactivation and activation data buffers.
	m_activationBufferSize = m_inputDataCount * m_activationDataSize * sizeof(float);
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

void MockStandardLayer::InitializeWeights(float weightsDeviation)
{
	default_random_engine generator((uint)chrono::system_clock::now().time_since_epoch().count());
	normal_distribution<float> distribution(0.f, weightsDeviation);

	size_t weightsBufferLength = m_weightsBufferSize / sizeof(float);
	for (size_t i = 0; i < weightsBufferLength; ++i)
	{
		m_weightsBuffer[i] = distribution(generator);
	}
}

void MockStandardLayer::InitializeBuffer(float* buffer, size_t bufferSize, float initialValue)
{
	size_t bufferLength = bufferSize / sizeof(float);
	for (size_t i = 0; i < bufferLength; ++i)
	{
		buffer[i] = initialValue;
	}
}

MockStandardLayer::~MockStandardLayer()
{
	if (m_holdsInputData)
	{
		CudaAssert(cudaFreeHost(m_inputDataBuffer));
	}
	m_inputDataBuffer = NULL;
	CudaAssert(cudaFreeHost(m_inputGradientsBuffer));
	m_inputGradientsBuffer = NULL;

	CudaAssert(cudaFreeHost(m_weightsBuffer));
	CudaAssert(cudaFreeHost(m_weightsGradientsBuffer));
	CudaAssert(cudaFreeHost(m_weightsUpdateBuffer));

	CudaAssert(cudaFreeHost(m_biasesBuffer));
	CudaAssert(cudaFreeHost(m_biasesGradientsBuffer));
	CudaAssert(cudaFreeHost(m_biasesUpdateBuffer));

	CudaAssert(cudaFreeHost(m_preactivationDataBuffer));
	if (m_outputToGpuMemory)
	{
		CudaAssert(cudaFree(m_activationDataBuffer));
	}
	else
	{
		CudaAssert(cudaFreeHost(m_activationDataBuffer));
	}
	m_activationDataBuffer = NULL;

	CudaAssert(cudaFreeHost(m_preactivationGradientsBuffer));
	if (m_holdsActivationGradients)
	{
		CudaAssert(cudaFreeHost(m_activationGradientsBuffer));
	}
	m_activationGradientsBuffer = NULL;
}

void MockStandardLayer::LoadInputs()
{
	TestingAssert(m_prevLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_inputDataBuffer, m_prevLayers[0]->GetActivationDataBuffer(), m_inputBufferSize, cudaMemcpyDeviceToHost));
}

void MockStandardLayer::LoadActivationGradients()
{
	TestingAssert(m_nextLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_activationGradientsBuffer, m_nextLayers[0]->GetInputGradientsBuffer(), m_activationBufferSize, cudaMemcpyDeviceToHost));
}

void MockStandardLayer::CalculatePreactivations()
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint neuronIndex = 0; neuronIndex < m_numNeurons; ++neuronIndex)
		{
			const uint preactivationDataBufferOffset = dataIndex + neuronIndex * m_inputDataCount;
			m_preactivationDataBuffer[preactivationDataBufferOffset] = 0.f;
			for (uint weightIndex = 0; weightIndex < m_numWeightsPerNeuron; ++weightIndex)
			{
				m_preactivationDataBuffer[preactivationDataBufferOffset] += m_inputDataBuffer[dataIndex + weightIndex * m_inputDataCount] *
					m_weightsBuffer[neuronIndex * m_numWeightsPerNeuron + weightIndex];
			}
		}
	}
}

void MockStandardLayer::AddBiases()
{
	for (uint neuronIndex = 0; neuronIndex < m_numNeurons; ++neuronIndex)
	{
		float biasValue = m_biasesBuffer[neuronIndex];
		for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
		{
			m_preactivationDataBuffer[neuronIndex * m_inputDataCount + dataIndex] += biasValue;
		}
	}
}

void MockStandardLayer::CalculateActivations()
{
	for (uint i = 0; i < m_activationBufferSize / sizeof(float); ++i)
	{
		if (m_activationType == ActivationType::Linear)
		{
			m_activationDataBuffer[i] = m_preactivationDataBuffer[i];
		}
		else if (m_activationType == ActivationType::ReLu)
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

void MockStandardLayer::DoForwardProp(PropagationMode propagationMode)
{
	CalculatePreactivations();
	AddBiases();
	CalculateActivations();

	if (m_outputToGpuMemory)
	{
		float* tmpBuffer;
		CudaAssert(cudaMalloc<float>(&tmpBuffer, m_activationBufferSize));
		CudaAssert(cudaMemcpy(tmpBuffer, m_activationDataBuffer, m_activationBufferSize, cudaMemcpyHostToDevice));
		CudaAssert(cudaFreeHost(m_activationDataBuffer));
		m_activationDataBuffer = tmpBuffer;
	}
}

void MockStandardLayer::CalculateBiasesGradients()
{
	float batchSize = m_parallelismMode == ParallelismMode::Model ? (float)m_inputDataCount : (float)(m_tierSize * m_inputDataCount);
	for (uint neuronIndex = 0; neuronIndex < m_numNeurons; ++neuronIndex)
	{
		float biasGradient = 0.f;
		uint neuronPreactivationsOffset = neuronIndex * m_inputDataCount;
		for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
		{
			biasGradient += m_preactivationGradientsBuffer[neuronPreactivationsOffset + dataIndex];
		}

		m_biasesGradientsBuffer[neuronIndex] = biasGradient / batchSize;
	}
}

void MockStandardLayer::CalculateWeightsGradients()
{
	float batchSize = m_parallelismMode == ParallelismMode::Model ? (float)m_inputDataCount : (float)(m_tierSize * m_inputDataCount);
	for (uint neuronIndex = 0; neuronIndex < m_numNeurons; ++neuronIndex)
	{
		for (uint weightIndex = 0; weightIndex < m_numWeightsPerNeuron; ++weightIndex)
		{
			const uint weightsGradientsBufferOffset = neuronIndex * m_numWeightsPerNeuron + weightIndex;
			m_weightsGradientsBuffer[weightsGradientsBufferOffset] = 0.f;
			for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
			{
				m_weightsGradientsBuffer[weightsGradientsBufferOffset] += m_inputDataBuffer[weightIndex * m_inputDataCount + dataIndex] *
					m_preactivationGradientsBuffer[neuronIndex * m_inputDataCount + dataIndex];
			}
			m_weightsGradientsBuffer[weightsGradientsBufferOffset] /= batchSize;
		}
	}
}

void MockStandardLayer::CalculateInputGradients()
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		for (uint weightIndex = 0; weightIndex < m_numWeightsPerNeuron; ++weightIndex)
		{
			const uint inputGradientsBufferOffset = dataIndex + weightIndex * m_inputDataCount;
			m_inputGradientsBuffer[inputGradientsBufferOffset] = 0.f;
			for (uint neuronIndex = 0; neuronIndex < m_numNeurons; ++neuronIndex)
			{
				m_inputGradientsBuffer[inputGradientsBufferOffset] += m_preactivationGradientsBuffer[neuronIndex * m_inputDataCount + dataIndex] *
					m_weightsBuffer[neuronIndex * m_numWeightsPerNeuron + weightIndex];
			}
		}
	}
}

void MockStandardLayer::CalculatePreactivationsGradients()
{
	for (uint i = 0; i < m_activationBufferSize / sizeof(float); ++i)
	{
		if (m_activationType == ActivationType::Linear)
		{
			m_preactivationGradientsBuffer[i] = m_activationGradientsBuffer[i];
		}
		else if (m_activationType == ActivationType::ReLu)
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

void MockStandardLayer::DoBackwardProp()
{
	CalculatePreactivationsGradients();
	CalculateInputGradients();
	CalculateWeightsGradients();
	CalculateBiasesGradients();
}

void MockStandardLayer::UpdateLayerParameters(float learningProgress)
{
	// Updating weights.
	float weightsUpdateProgressSteps = floorf(learningProgress / m_weightsUpdateLearningRateProgressStep);
	const float weightsLearningRate = m_weightsUpdateStartingLearningRate * powf(m_weightsUpdateLearningRateUpdateFactor, weightsUpdateProgressSteps);
	for (uint i = 0; i < m_weightsBufferSize / sizeof(float); ++i)
	{
		m_weightsUpdateBuffer[i] = m_weightsUpdateMomentum * m_weightsUpdateBuffer[i] + weightsLearningRate * (m_weightsGradientsBuffer[i] -
			m_weightsUpdateDecay * m_weightsBuffer[i]);
		m_weightsBuffer[i] += m_weightsUpdateBuffer[i];
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