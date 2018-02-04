// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network output layer, used in tests.
// Created: 02/21/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/mockoutputlayer.cuh"

MockOutputLayer::MockOutputLayer(uint inputDataSize, uint inputDataCount, LossFunctionType lossFunctionType, bool calculateMultipleGuessAccuracy, uint numGuesses,
	bool generateRandomInputGradients)
{
	m_layerType = LayerType::Output;
	m_indexInTier = 0;
	m_tierSize = 1;
	m_lossFunctionType = lossFunctionType;

	m_inputDataSize = m_activationDataSize = inputDataSize;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = true;

	// Allocating input data buffer.
	m_inputBufferSize = m_inputDataSize * m_inputDataCount * sizeof(float);
	if (m_holdsInputData)
	{
		CudaAssert(cudaMallocHost<float>(&m_inputDataBuffer, m_inputBufferSize));
	}

	m_labelsBufferSize = m_inputDataCount * sizeof(uint);
	m_lossBuffersSize = m_inputDataCount * sizeof(float);

	CudaAssert(cudaMallocHost<uint>(&m_dataLabels, m_labelsBufferSize));

	if (m_lossFunctionType == LossFunctionType::LogisticRegression)
	{
		CudaAssert(cudaMallocHost<float>(&m_logLikelihoods, m_lossBuffersSize));
	}

	CudaAssert(cudaMallocHost<float>(&m_scores, m_lossBuffersSize));

	m_calculateMultipleGuessAccuracy = calculateMultipleGuessAccuracy;
	if (m_calculateMultipleGuessAccuracy)
	{
		m_numGuesses = numGuesses;
		CudaAssert(cudaMallocHost<float>(&m_multipleGuessScores, m_lossBuffersSize));
	}

	m_generateRandomInputGradients = generateRandomInputGradients;
	if (m_generateRandomInputGradients)
	{
		CudaAssert(cudaMalloc<float>(&m_inputGradientsBuffer, m_inputBufferSize));
	}

	m_holdsActivationGradients = false;
}

void MockOutputLayer::LoadDataLabels(vector<uint> dataLabels)
{
	for (size_t i = 0; i < dataLabels.size(); ++i)
	{
		m_dataLabels[i] = dataLabels[i];
	}
}

MockOutputLayer::~MockOutputLayer()
{
	if (m_holdsInputData)
	{
		CudaAssert(cudaFreeHost(m_inputDataBuffer));
	}
	m_inputDataBuffer = NULL;
	m_activationDataBuffer = NULL;

	CudaAssert(cudaFreeHost(m_dataLabels));

	if (m_lossFunctionType == LossFunctionType::LogisticRegression)
	{
		CudaAssert(cudaFreeHost(m_logLikelihoods));
	}

	CudaAssert(cudaFreeHost(m_scores));

	if (m_calculateMultipleGuessAccuracy)
	{
		CudaAssert(cudaFreeHost(m_multipleGuessScores));
	}

	if (m_generateRandomInputGradients)
	{
		CudaAssert(cudaFree(m_inputGradientsBuffer));
	}
	m_inputGradientsBuffer = NULL;
}

void MockOutputLayer::LoadInputs()
{
	TestingAssert(m_prevLayers.size() == 1, "We do not support more than one previous layer in tests, for now.");
	CudaAssert(cudaMemcpy(m_inputDataBuffer, m_prevLayers[0]->GetActivationDataBuffer(), m_inputBufferSize, cudaMemcpyDeviceToHost));
}

void MockOutputLayer::CalculateLogisticRegressionLoss()
{
	for (uint dataIndex = 0; dataIndex < m_inputDataCount; ++dataIndex)
	{
		float predictedProbability = m_inputDataBuffer[m_dataLabels[dataIndex] * m_inputDataCount + dataIndex];
		m_logLikelihoods[dataIndex] = logf(predictedProbability);

		// Counting for how many incorrect labels we predicted higher or equal probability.
		uint predictedHigherOrEqualCnt = 0;
		for (size_t inputActivationIndex = 0; inputActivationIndex < m_inputDataSize; ++inputActivationIndex)
		{
			if (m_inputDataBuffer[inputActivationIndex * m_inputDataCount + dataIndex] >= predictedProbability)
			{
				++predictedHigherOrEqualCnt;
			}
		}

		m_scores[dataIndex] = predictedHigherOrEqualCnt > 1 ? 0.f : 1.0f;
		if (m_calculateMultipleGuessAccuracy)
		{
			m_multipleGuessScores[dataIndex] = predictedHigherOrEqualCnt > m_numGuesses ? 0.f : 1.f;
		}
	}
}

void MockOutputLayer::LogisticRegressionForwardProp()
{
	CalculateLogisticRegressionLoss();

	// Calculating loss.
	m_loss = 0.f;
	for (uint i = 0; i < m_inputDataCount; ++i)
	{
		m_loss -= m_logLikelihoods[i];
	}

	// Calculating accuracy.
	m_accuracy = 0.f;
	for (uint i = 0; i < m_inputDataCount; ++i)
	{
		m_accuracy += m_scores[i];
	}

	// Calculating multiple guess accuracy.
	m_multipleGuessAccuracy = 0.f;
	for (uint i = 0; i < m_inputDataCount; ++i)
	{
		m_multipleGuessAccuracy += m_multipleGuessScores[i];
	}
}

void MockOutputLayer::DoForwardProp(PropagationMode propagationMode)
{
	m_activationDataBuffer = m_inputDataBuffer;

	if (m_lossFunctionType == LossFunctionType::LogisticRegression)
	{
		LogisticRegressionForwardProp();
	}
}

void MockOutputLayer::DoBackwardProp()
{
	if (m_generateRandomInputGradients)
	{
		float* tempBuffer;
		CudaAssert(cudaMallocHost<float>(&tempBuffer, m_inputBufferSize));

		default_random_engine generator((uint)chrono::system_clock::now().time_since_epoch().count());
		uniform_int_distribution<int> distribution(0, 255);

		size_t inputBufferLength = m_inputBufferSize / sizeof(float);
		for (size_t i = 0; i < inputBufferLength; ++i)
		{
			tempBuffer[i] = 0.1f * (float)distribution(generator) / 255.0f;
		}

		CudaAssert(cudaMemcpy(m_inputGradientsBuffer, tempBuffer, m_inputBufferSize, cudaMemcpyHostToDevice));
		CudaAssert(cudaFreeHost(tempBuffer));
	}
	else
	{
		if (m_lossFunctionType == LossFunctionType::LogisticRegression)
		{
			if (m_prevLayers[0]->GetLayerType() == LayerType::SoftMax)
			{
				// If previous layer is SoftMax we are letting it handle the gradient computation, for numerical stability.
				m_inputGradientsBuffer = NULL;
			}
			else
			{
				TestingAssert(false, "Currently not supported!");
			}
		}
	}
}