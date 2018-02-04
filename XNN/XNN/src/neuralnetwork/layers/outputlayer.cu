// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network output layer.
// Created: 02/20/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/outputlayer.cuh"

OutputLayer::OutputLayer(cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, uint inputDataSize, uint inputDataCount,
	uint labelsCount, LossFunctionType lossFunctionType, bool calculateMultipleGuessAccuracy, uint numGuesses, uint numTestPasses)
{
	m_layerType = LayerType::Output;
	m_parallelismMode = ParallelismMode::Model;
	m_deviceCalculationStream = deviceCalculationStream;
	m_deviceMemoryStream = deviceMemoryStream;
	m_indexInTier = 0;
	m_tierSize = 1;
	m_lossFunctionType = lossFunctionType;

	m_numTestPasses = numTestPasses;
	m_testPassCounter = 0;

	m_inputDataSize = m_activationDataSize = inputDataSize;
	m_inputDataCount = inputDataCount;
	m_holdsInputData = false;
	m_inputBufferSize = m_inputDataSize * m_inputDataCount * sizeof(float);
	m_testAverageInputsBuffer = NULL;

	m_labelsCount = labelsCount;
	m_labelsOffset = 0;
	m_labelsBufferSize = m_labelsCount * sizeof(uint);
	CudaAssert(cudaMalloc<uint>(&m_dataLabels, m_labelsBufferSize));
	CudaAssert(cudaMallocHost<uint>(&m_hostLabelsBuffer, m_labelsBufferSize));

	m_lossBuffersSize = m_inputDataCount * sizeof(float);
	m_loss = 0.f;
	m_accuracy = 0.f;
	m_multipleGuessAccuracy = 0.f;

	if (m_lossFunctionType == LossFunctionType::LogisticRegression)
	{
		CudaAssert(cudaMalloc<float>(&m_logLikelihoods, m_lossBuffersSize));
		CudaAssert(cudaMallocHost<float>(&m_hostLogLikelihoods, m_lossBuffersSize));
	}

	CudaAssert(cudaMalloc<float>(&m_scores, m_lossBuffersSize));
	CudaAssert(cudaMallocHost<float>(&m_hostScores, m_lossBuffersSize));

	m_calculateMultipleGuessAccuracy = calculateMultipleGuessAccuracy;
	if (m_calculateMultipleGuessAccuracy)
	{
		m_numGuesses = numGuesses;
		CudaAssert(cudaMalloc<float>(&m_multipleGuessScores, m_lossBuffersSize));
		CudaAssert(cudaMallocHost<float>(&m_hostMultipleGuessScores, m_lossBuffersSize));
	}

	m_holdsActivationGradients = false;
	m_activationGradientsBuffer = NULL;
}

void OutputLayer::Reinitialize(uint newInputDataCount)
{
	m_inputDataCount = newInputDataCount;
	m_labelsBufferSize = m_inputDataCount * sizeof(uint);
	m_lossBuffersSize = m_inputDataCount * sizeof(float);
}

void OutputLayer::LoadDataLabels(vector<uint> dataLabels)
{
	m_labelsOffset = 0;

	for (size_t i = 0; i < dataLabels.size(); ++i)
	{
		m_hostLabelsBuffer[i] = dataLabels[i];
	}

	CudaAssert(cudaMemcpyAsync(m_dataLabels, m_hostLabelsBuffer, m_labelsBufferSize, cudaMemcpyHostToDevice, m_deviceMemoryStream));

	SynchronizeMemoryOperations();
}

OutputLayer::~OutputLayer()
{
	m_inputDataBuffer = NULL;
	m_activationDataBuffer = NULL;

	CudaAssert(cudaFree(m_dataLabels));
	CudaAssert(cudaFreeHost(m_hostLabelsBuffer));
	
	if (m_lossFunctionType == LossFunctionType::LogisticRegression)
	{
		CudaAssert(cudaFree(m_logLikelihoods));
		CudaAssert(cudaFreeHost(m_hostLogLikelihoods));
	}

	CudaAssert(cudaFree(m_scores));
	CudaAssert(cudaFreeHost(m_hostScores));

	if (m_calculateMultipleGuessAccuracy)
	{
		CudaAssert(cudaFree(m_multipleGuessScores));
		CudaAssert(cudaFreeHost(m_hostMultipleGuessScores));
	}

	if (m_testAverageInputsBuffer != NULL)
	{
		CudaAssert(cudaFree(m_testAverageInputsBuffer));
	}
}

void OutputLayer::LoadInputs()
{
	CommonLoadInputs();
}

/*
	Calculates loss and scores for logistic regression loss function.
*/
__global__ void CalculateLogisticRegressionLoss(float* inputActivations, uint* dataLabels, const uint numInputSamples, const uint numInputActivations,
	float* logLikelihoods, float* scores, bool calculateMultipleGuessAccuracy, uint numGuesses, float* multipleGuessScores)
{
	const uint c_dataIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (c_dataIndex < numInputSamples)
	{
		float predictedProbability = inputActivations[dataLabels[c_dataIndex] * numInputSamples + c_dataIndex];
		logLikelihoods[c_dataIndex] = predictedProbability > 0.f ? __logf(predictedProbability) : -50.0f;

		// Counting for how many incorrect labels we predicted higher or equal probability.
		uint predictedHigherOrEqualCnt = 0;
		for (size_t inputActivationIndex = 0; inputActivationIndex < numInputActivations; ++inputActivationIndex)
		{
			if (inputActivations[inputActivationIndex * numInputSamples + c_dataIndex] >= predictedProbability)
			{
				++predictedHigherOrEqualCnt;
			}
		}

		scores[c_dataIndex] = predictedHigherOrEqualCnt > 1 ? 0.f : 1.0f;
		if (calculateMultipleGuessAccuracy)
		{
			multipleGuessScores[c_dataIndex] = predictedHigherOrEqualCnt > numGuesses ? 0.f : 1.f;
		}
	}
}

void OutputLayer::LogisticRegressionForwardProp(PropagationMode propagationMode)
{
	// Calculating log likelihoods and scores.
	const uint c_numThreadsPerBlock = 128;
	const uint c_numBlocks = DivideUp(m_inputDataCount, c_numThreadsPerBlock);
	float* inputBuffer = propagationMode == PropagationMode::Train ? m_inputDataBuffer : m_testAverageInputsBuffer;
	LAUNCH_KERNEL_ASYNC(CalculateLogisticRegressionLoss, dim3(c_numBlocks), dim3(c_numThreadsPerBlock), m_deviceCalculationStream)(inputBuffer, m_dataLabels + m_labelsOffset,
		m_inputDataCount, m_activationDataSize, m_logLikelihoods, m_scores, m_calculateMultipleGuessAccuracy, m_numGuesses, m_multipleGuessScores);
	CudaAssert(cudaGetLastError());
	SynchronizeCalculations();

	// Copying loss and score buffers to host memory.
	CudaAssert(cudaMemcpyAsync(m_hostLogLikelihoods, m_logLikelihoods, m_lossBuffersSize, cudaMemcpyDeviceToHost, m_deviceMemoryStream));
	CudaAssert(cudaMemcpyAsync(m_hostScores, m_scores, m_lossBuffersSize, cudaMemcpyDeviceToHost, m_deviceMemoryStream));
	if (m_calculateMultipleGuessAccuracy)
	{
		CudaAssert(cudaMemcpyAsync(m_hostMultipleGuessScores, m_multipleGuessScores, m_lossBuffersSize, cudaMemcpyDeviceToHost, m_deviceMemoryStream));
	}
	SynchronizeMemoryOperations();

	// Calculating loss.
	m_loss = 0.f;
	for (uint i = 0; i < m_inputDataCount; ++i)
	{
		m_loss -= m_hostLogLikelihoods[i];
		
		// It is bad sign if this happens! If previous layer was SoftMax, it means that some input activation in softmax layer
		// was so large or small, that it produced zero exponential on normalized smallest input activation.
		if (m_hostLogLikelihoods[i] == -50.f)
		{
			EmitWarning("Encountered zero probability in output layer!");
		}
	}

	// Calculating accuracy.
	m_accuracy = 0.f;
	for (uint i = 0; i < m_inputDataCount; ++i)
	{
		m_accuracy += m_hostScores[i];
	}

	if (m_calculateMultipleGuessAccuracy)
	{
		// Calculating multiple guess accuracy.
		m_multipleGuessAccuracy = 0.f;
		for (uint i = 0; i < m_inputDataCount; ++i)
		{
			m_multipleGuessAccuracy += m_hostMultipleGuessScores[i];
		}
	}
}

/*
	Averaging summed inputs from each test pass.
*/
__global__ void AverageSummedInputs(float* testAverageInputsBuffer, float numTestPasses, const uint bufferLength)
{
	for (uint bufferIndex = blockIdx.x * blockDim.x + threadIdx.x; bufferIndex < bufferLength; bufferIndex += gridDim.x * blockDim.x)
	{
		testAverageInputsBuffer[bufferIndex] /= numTestPasses;
	}
}

void OutputLayer::DoForwardProp(PropagationMode propagationMode)
{
	m_activationDataBuffer = m_inputDataBuffer;

	bool lastTestPass = false;
	if (propagationMode == PropagationMode::Test)
	{
		++m_testPassCounter;

		if (m_testPassCounter == 1)
		{
			// Allocate test average inputs buffer first time we do a test pass.
			if (m_testAverageInputsBuffer == NULL)
			{
				CudaAssert(cudaMalloc<float>(&m_testAverageInputsBuffer, m_inputBufferSize));
			}
			
			// Using device calculation stream on purpose, to avoid sync between streams.
			CudaAssert(cudaMemcpyAsync(m_testAverageInputsBuffer, m_inputDataBuffer, m_inputBufferSize, cudaMemcpyDeviceToDevice, m_deviceCalculationStream));
		}
		else
		{
			// Adding input from this pass.
			CalculateElementWiseSum(m_testAverageInputsBuffer, m_inputDataBuffer, (uint)(m_inputBufferSize / sizeof(float)), m_testAverageInputsBuffer,
				m_deviceCalculationStream);
		}

		if (m_testPassCounter == m_numTestPasses)
		{
			lastTestPass = true;
			
			// Averaging summed inputs from each test pass.
			CalculateElementWiseScale(m_testAverageInputsBuffer, (float)m_numTestPasses, (uint)(m_inputBufferSize / sizeof(float)), m_testAverageInputsBuffer,
				m_deviceCalculationStream);
		}
	}

	if (propagationMode == PropagationMode::Train || lastTestPass)
	{
		if (m_lossFunctionType == LossFunctionType::LogisticRegression)
		{
			LogisticRegressionForwardProp(propagationMode);
		}
	}

	if (lastTestPass)
	{
		m_testPassCounter = 0;
	}
}

void OutputLayer::DoBackwardProp()
{
	if (m_lossFunctionType == LossFunctionType::LogisticRegression)
	{
		if (m_prevLayers[0]->GetLayerType() == LayerType::SoftMax)
		{
			// If previous layer is SoftMax we are letting it handle the gradient computation, for numerical stability.
		}
		else
		{
			ShipAssert(false, "Currently not supported!");
		}
	}
}