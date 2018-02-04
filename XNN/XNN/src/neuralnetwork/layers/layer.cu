// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Abstract neural network layer.
// Created: 12/27/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/inputlayer.cuh"

Layer::Layer()
{
	m_inputLayerIndexInTier = -1;
	m_inputGradientsBuffer = NULL;
	m_activationGradientsHelpBuffer = NULL;
}

Layer::~Layer()
{
	if (m_holdsInputData && m_inputDataBuffer != NULL)
	{
		CudaAssert(cudaFree(m_inputDataBuffer));
	}

	if (m_inputGradientsBuffer != NULL)
	{
		CudaAssert(cudaFree(m_inputGradientsBuffer));
	}

	if (m_activationDataBuffer != NULL)
	{
		CudaAssert(cudaFree(m_activationDataBuffer));
	}

	if (m_holdsActivationGradients && m_activationGradientsBuffer != NULL)
	{
		CudaAssert(cudaFree(m_activationGradientsBuffer));
	}

	if (m_holdsActivationGradients && m_activationGradientsHelpBuffer != NULL)
	{
		CudaAssert(cudaFree(m_activationGradientsHelpBuffer));
	}
}

void Layer::Reinitialize(uint newInputDataCount)
{
	m_inputDataCount = newInputDataCount;
	m_inputBufferSize = m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);
	m_activationBufferSize = m_inputBufferSize;
}

void Layer::CommonLoadInputs()
{
	// If it holds the input data then it means it is connected with some layer trained on different GPU.
	if (m_holdsInputData)
	{
		if (m_prevLayers[0]->GetParallelismMode() == ParallelismMode::Model)
		{
			if (m_parallelismMode == ParallelismMode::Model)
			{
				if (m_prevLayers[0]->GetActivationDataCount() != m_inputDataCount)
				{
					Reinitialize(m_prevLayers[0]->GetActivationDataCount());
				}

				size_t prevBufferSize = m_prevLayers[0]->GetActivationBufferSize();
				size_t prevBufferLength = prevBufferSize / sizeof(float);
				for (size_t prevLayerIndex = 0; prevLayerIndex < m_prevLayers.size(); ++prevLayerIndex)
				{
					if (m_prevLayers[prevLayerIndex]->GetIndexInTier() == m_indexInTier)
					{
						CudaAssert(cudaMemcpyAsync(m_inputDataBuffer + prevLayerIndex * prevBufferLength, m_prevLayers[prevLayerIndex]->GetActivationDataBuffer(),
							prevBufferSize, cudaMemcpyDeviceToDevice, m_deviceMemoryStream));
					}
					else
					{
						CudaAssert(cudaMemcpyPeerAsync(m_inputDataBuffer + prevLayerIndex * prevBufferLength, m_indexInTier, m_prevLayers[prevLayerIndex]->GetActivationDataBuffer(),
							m_prevLayers[prevLayerIndex]->GetIndexInTier(), prevBufferSize, m_deviceMemoryStream));
					}
				}
			}
			else if (m_parallelismMode == ParallelismMode::Data)
			{
				ShipAssert(false, "Currently not supported!");
			}
		}
		else if (m_prevLayers[0]->GetParallelismMode() == ParallelismMode::Data)
		{
			if (m_parallelismMode == ParallelismMode::Model)
			{
				uint inputLayerIndexInTier = (uint)(m_inputLayerIndexInTier + 1);

				if (m_prevLayers[inputLayerIndexInTier]->GetActivationDataCount() != m_inputDataCount)
				{
					Reinitialize(m_prevLayers[inputLayerIndexInTier]->GetActivationDataCount());
				}

				if (inputLayerIndexInTier == m_indexInTier)
				{
					m_inputDataBuffer = m_prevLayers[inputLayerIndexInTier]->GetActivationDataBuffer();
				}
				else
				{
					CudaAssert(cudaMemcpyPeerAsync(m_inputDataBuffer, m_indexInTier, m_prevLayers[inputLayerIndexInTier]->GetActivationDataBuffer(),
						m_prevLayers[inputLayerIndexInTier]->GetIndexInTier(), m_inputBufferSize, m_deviceMemoryStream));
				}
			}
			else if (m_parallelismMode == ParallelismMode::Data)
			{
				ShipAssert(false, "Currently not supported!");
			}
		}
	}
	else
	{
		if (m_prevLayers[0]->GetLayerType() == LayerType::Input)
		{
			InputLayer* inputLayer = static_cast<InputLayer*>(m_prevLayers[0]);

			if (inputLayer->GetActivationDataCount(m_indexInTier) != m_inputDataCount)
			{
				Reinitialize(inputLayer->GetActivationDataCount(m_indexInTier));
			}

			m_inputDataBuffer = inputLayer->GetActivationDataBuffer(m_indexInTier);
		}
		else
		{
			if (m_prevLayers[0]->GetActivationDataCount() != m_inputDataCount)
			{
				Reinitialize(m_prevLayers[0]->GetActivationDataCount());
			}

			m_inputDataBuffer = m_prevLayers[0]->GetActivationDataBuffer();
		}
	}
}

void Layer::LoadActivationGradients()
{
	if (m_layerType == LayerType::Input)
	{
		ShipAssert(false, "Shouldn't load gradients to input layer!");
	}
	else if (m_layerType == LayerType::Output)
	{
		// Nothing to load for output layer.
		return;
	}

	// If it holds the activation gradients then it means it is connected with some layer trained on different GPU.
	if (m_holdsActivationGradients)
	{
		if (m_nextLayers[0]->GetParallelismMode() == ParallelismMode::Model)
		{
			if (m_parallelismMode == ParallelismMode::Model || m_parallelismMode == ParallelismMode::Data)
			{
				size_t activationBufferLength = m_activationBufferSize / sizeof(float);

				// Copy over gradients from first of next layers.
				float* inputGradientsBuffer = m_parallelismMode == ParallelismMode::Data ? m_nextLayers[0]->GetInputGradientsBuffer() :
					m_nextLayers[0]->GetInputGradientsBuffer() + m_indexInTier * activationBufferLength;
				if (m_nextLayers[0]->GetIndexInTier() == m_indexInTier)
				{
					CudaAssert(cudaMemcpyAsync(m_activationGradientsBuffer, inputGradientsBuffer, m_activationBufferSize, cudaMemcpyDeviceToDevice, m_deviceMemoryStream));
				}
				else
				{
					CudaAssert(cudaMemcpyPeerAsync(m_activationGradientsBuffer, m_indexInTier, inputGradientsBuffer, m_nextLayers[0]->GetIndexInTier(),
						m_activationBufferSize, m_deviceMemoryStream));
				}

				// Add up gradients from rest of next layers.
				if (m_nextLayers.size() > 1)
				{
					if (m_activationGradientsHelpBuffer == NULL)
					{
						CudaAssert(cudaMalloc<float>(&m_activationGradientsHelpBuffer, m_activationBufferSize));
					}
					
					for (size_t nextLayerIndex = 1; nextLayerIndex < m_nextLayers.size(); ++nextLayerIndex)
					{
						// Copy gradients to temp buffer.
						inputGradientsBuffer = m_parallelismMode == ParallelismMode::Data ? m_nextLayers[nextLayerIndex]->GetInputGradientsBuffer() :
							m_nextLayers[nextLayerIndex]->GetInputGradientsBuffer() + m_indexInTier * activationBufferLength;
						if (m_nextLayers[nextLayerIndex]->GetIndexInTier() == m_indexInTier)
						{
							CudaAssert(cudaMemcpyAsync(m_activationGradientsHelpBuffer, inputGradientsBuffer, m_activationBufferSize, cudaMemcpyDeviceToDevice, m_deviceMemoryStream));
						}
						else
						{
							CudaAssert(cudaMemcpyPeerAsync(m_activationGradientsHelpBuffer, m_indexInTier, inputGradientsBuffer, m_nextLayers[nextLayerIndex]->GetIndexInTier(),
								m_activationBufferSize, m_deviceMemoryStream));
						}

						// Add gradients from temp buffer to gradients buffer.
						CalculateElementWiseSum(m_activationGradientsBuffer, m_activationGradientsHelpBuffer, (uint)activationBufferLength, m_activationGradientsBuffer, m_deviceMemoryStream);
					}
				}
			}
		}
		else if (m_nextLayers[0]->GetParallelismMode() == ParallelismMode::Data)
		{
			if (m_parallelismMode == ParallelismMode::Model)
			{
				ShipAssert(false, "Currently not supported!");
			}
			else if (m_parallelismMode == ParallelismMode::Data)
			{
				ShipAssert(false, "Currently not supported!");
			}
		}
	}
	else
	{
		m_activationGradientsBuffer = m_nextLayers[0]->GetInputGradientsBuffer();
	}
}

/*
	Usually done once at the beginning of training, so we can afford doing work on host.
*/
void Layer::InitializeParamsFromDistribution(float* paramsBuffer, size_t paramsBufferSize, float deviation)
{
	default_random_engine generator((uint)chrono::system_clock::now().time_since_epoch().count());
	normal_distribution<float> distribution(0.f, deviation);

	float* hostParamsBuffer;
	CudaAssert(cudaMallocHost<float>(&hostParamsBuffer, paramsBufferSize, cudaHostAllocPortable));
	size_t paramsBufferLength = paramsBufferSize / sizeof(float);
	for (size_t i = 0; i < paramsBufferLength; ++i)
	{
		hostParamsBuffer[i] = distribution(generator);
	}
	CudaAssert(cudaMemcpyAsync(paramsBuffer, hostParamsBuffer, paramsBufferSize, cudaMemcpyHostToDevice, m_deviceMemoryStream));
	SynchronizeMemoryOperations();
	CudaAssert(cudaFreeHost(hostParamsBuffer));
}

/*
	Usually done once at the beginning of training, so we can afford doing work on host.
*/
void Layer::InitializeParamsToValue(float* paramsBuffer, size_t paramsBufferSize, float initialValue)
{
	float* hostParamsBuffer;
	CudaAssert(cudaMallocHost<float>(&hostParamsBuffer, paramsBufferSize, cudaHostAllocPortable));
	size_t paramsBufferLength = paramsBufferSize / sizeof(float);
	for (size_t i = 0; i < paramsBufferLength; ++i)
	{
		hostParamsBuffer[i] = initialValue;
	}
	CudaAssert(cudaMemcpyAsync(paramsBuffer, hostParamsBuffer, paramsBufferSize, cudaMemcpyHostToDevice, m_deviceMemoryStream));
	SynchronizeMemoryOperations();
	CudaAssert(cudaFreeHost(hostParamsBuffer));
}

/*
	Updates layer parameters by applying momentum to last update, learning rate to gradients, and decay to parameters.
*/
__global__ void ApplyParamatersUpdate(float* paramsBuffer, float* gradientsBuffer, float* updatesBuffer, uint numElements,
	float updateMomentum, float learningRate, float updateDecay)
{
	for (uint elementIndex = blockIdx.x * blockDim.x + threadIdx.x; elementIndex < numElements; elementIndex += gridDim.x * blockDim.x)
	{
		updatesBuffer[elementIndex] = updateMomentum * updatesBuffer[elementIndex] + learningRate * (gradientsBuffer[elementIndex] -
			updateDecay * paramsBuffer[elementIndex]);
		paramsBuffer[elementIndex] += updatesBuffer[elementIndex];
	}
}

void Layer::CommonUpdateLayerParameters(float learningProgress, float* weightsBuffer, float* weightsGradientsBuffer, float* weightsUpdateBuffer,
	uint weightsGradientsBufferLength, float weightsUpdateMomentum, float weightsUpdateLearningRateProgressStep, float weightsUpdateStartingLearningRate,
	float weightsUpdateLearningRateUpdateFactor, float weightsUpdateDecay, float* biasesBuffer, float* biasesGradientsBuffer, float* biasesUpdateBuffer,
	uint biasesGradientsBufferLength, float biasesUpdateMomentum, float biasesUpdateLearningRateProgressStep, float biasesUpdateStartingLearningRate,
	float biasesUpdateLearningRateUpdateFactor, float biasesUpdateDecay)
{
	// Updating weights.
	const uint c_numBlocks = 128;
	const uint c_numThreadsPerBlock = 128;
	float weightsUpdateProgressSteps = floorf(learningProgress / weightsUpdateLearningRateProgressStep);
	const float c_weightsLearningRate = weightsUpdateStartingLearningRate * powf(weightsUpdateLearningRateUpdateFactor, weightsUpdateProgressSteps);
	dim3 blockDimensions(c_numThreadsPerBlock);
	dim3 gridDimensions(min(c_numBlocks, DivideUp(weightsGradientsBufferLength, c_numThreadsPerBlock)));
	LAUNCH_KERNEL_ASYNC(ApplyParamatersUpdate, gridDimensions, blockDimensions, m_deviceCalculationStream)(weightsBuffer, weightsGradientsBuffer,
		weightsUpdateBuffer, weightsGradientsBufferLength, weightsUpdateMomentum, c_weightsLearningRate, weightsUpdateDecay);
	CudaAssert(cudaGetLastError());

	// Updating biases.
	float biasesUpdateProgressSteps = floorf(learningProgress / biasesUpdateLearningRateProgressStep);
	const float c_biasesLearningRate = biasesUpdateStartingLearningRate * powf(biasesUpdateLearningRateUpdateFactor, biasesUpdateProgressSteps);
	blockDimensions = dim3(c_numThreadsPerBlock);
	gridDimensions = dim3(min(c_numBlocks, DivideUp(biasesGradientsBufferLength, c_numThreadsPerBlock)));
	LAUNCH_KERNEL_ASYNC(ApplyParamatersUpdate, gridDimensions, blockDimensions, m_deviceCalculationStream)(biasesBuffer, biasesGradientsBuffer,
		biasesUpdateBuffer, biasesGradientsBufferLength, biasesUpdateMomentum, c_biasesLearningRate, biasesUpdateDecay);
	CudaAssert(cudaGetLastError());

	SynchronizeCalculations();
}

void Layer::SynchronizeCalculations()
{
	CudaAssert(cudaStreamSynchronize(m_deviceCalculationStream));
}

void Layer::SynchronizeMemoryOperations()
{
	CudaAssert(cudaStreamSynchronize(m_deviceMemoryStream));
}