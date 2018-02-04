// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network.
// Created: 12/27/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/neuralnet.cuh"

NeuralNet::NeuralNet(size_t maxNetworkTierSize)
{
	m_maxNetworkTierSize = maxNetworkTierSize;

	for (size_t tierLayer = 0; tierLayer < m_maxNetworkTierSize; ++tierLayer)
	{
		CudaAssert(cudaSetDevice((int)tierLayer));

		// Initialize calculation stream.
		cudaStream_t deviceCalculationStream;
		CudaAssert(cudaStreamCreateWithFlags(&deviceCalculationStream, cudaStreamNonBlocking));
		m_deviceCalculationStreams.push_back(deviceCalculationStream);

		// Initialize memory stream.
		cudaStream_t deviceMemoryStream;
		CudaAssert(cudaStreamCreateWithFlags(&deviceMemoryStream, cudaStreamNonBlocking));
		m_deviceMemoryStreams.push_back(deviceMemoryStream);

		// Initialize cuBLAS handles.
		cublasHandle_t cublasHandle;
		CudaCublasAssert(cublasCreate(&cublasHandle));
		m_cublasHandles.push_back(cublasHandle);

		// Initialize cuRAND state buffers.
		curandState* curandStateBuffer;
		CudaAssert(cudaMalloc<curandState>(&curandStateBuffer, DropoutLayer::c_numCurandBlocks * DropoutLayer::c_numCurandThreadsPerBlock * sizeof(curandState)));
		InitCurandStatesBuffer(curandStateBuffer, deviceCalculationStream);
		m_curandStatesBuffers.push_back(curandStateBuffer);

		// We need to sync whole device since cublas uses stream 0 to create handles,
		// but this is called once per network so we don't care.
		CudaAssert(cudaDeviceSynchronize());
	}

	// Reverting back to default device.
	CudaAssert(cudaSetDevice(0));
}

/*
	Initializes one cuRAND state per thread.
*/
__global__ void InitializeCurandStates(curandState* curandStatesBuffer, unsigned long long seedValue)
{
	const uint c_stateIndex = blockIdx.x * blockDim.x + threadIdx.x;

	// Initializing each state with different subsequence, to get more statistically uncorrelated sequences in different cuRAND states.
	curand_init(seedValue, c_stateIndex, 0, &curandStatesBuffer[c_stateIndex]);
}

void NeuralNet::InitCurandStatesBuffer(curandState* curandStatesBuffer, cudaStream_t deviceCalculationStream)
{
	dim3 blockDimensions(DropoutLayer::c_numCurandThreadsPerBlock);
	dim3 gridDimensions(DropoutLayer::c_numCurandBlocks);
	// Making it less likely for statistically correlated sequences of random values across different cuRAND state buffers,
	// since they are all initialized in approximately same time.
	unsigned long long seedValue = 2 * chrono::system_clock::now().time_since_epoch().count() + 1;
	LAUNCH_KERNEL_ASYNC(InitializeCurandStates, gridDimensions, blockDimensions, deviceCalculationStream)(curandStatesBuffer, seedValue);
	CudaAssert(cudaGetLastError());
}

NeuralNet::~NeuralNet()
{
	// Delete layers.
	for (size_t tier = 0; tier < m_layersTiers.size(); ++tier)
	{
		for (size_t layerIndex = 0; layerIndex < m_layersTiers[tier].size(); ++layerIndex)
		{
			delete m_layersTiers[tier][layerIndex];
		}
	}

	// Destroy streams.
	for (size_t stream = 0; stream < m_deviceCalculationStreams.size(); ++stream)
	{
		CudaAssert(cudaStreamDestroy(m_deviceCalculationStreams[stream]));
		CudaAssert(cudaStreamDestroy(m_deviceMemoryStreams[stream]));
	}

	// Destroy cuBLAS handles.
	for (size_t handle = 0; handle < m_cublasHandles.size(); ++handle)
	{
		CudaCublasAssert(cublasDestroy(m_cublasHandles[handle]));
	}

	// Destroy cuRAND state buffers.
	for (size_t buffer = 0; buffer < m_curandStatesBuffers.size(); ++buffer)
	{
		CudaAssert(cudaFree(m_curandStatesBuffers[buffer]));
	}
}

void NeuralNet::SaveModel(string modelFile, bool saveUpdateBuffers)
{
	ofstream modelStream(modelFile, ofstream::out | ofstream::binary);

	for (size_t tier = 0; tier < m_layersTiers.size(); ++tier)
	{
		if (m_layersTiers[tier][0]->GetLayerType() == LayerType::Convolutional)
		{
			vector<ConvolutionalLayer*> layers;
			if (m_layersTiers[tier][0]->GetParallelismMode() == ParallelismMode::Data)
			{
				layers.push_back(static_cast<ConvolutionalLayer*>(m_layersTiers[tier][0]));
			}
			else
			{
				for (Layer* layer : m_layersTiers[tier])
				{
					layers.push_back(static_cast<ConvolutionalLayer*>(layer));
				}
			}

			// Writing filters.
			float* tempFiltersBuffer;
			CudaAssert(cudaMallocHost<float>(&tempFiltersBuffer, layers[0]->GetFiltersBufferSize()));
			for (ConvolutionalLayer* convLayer : layers)
			{
				CudaAssert(cudaSetDevice(convLayer->GetIndexInTier()));
				CudaAssert(cudaMemcpy(tempFiltersBuffer, convLayer->GetFiltersBuffer(), convLayer->GetFiltersBufferSize(), cudaMemcpyDeviceToHost));
				// Need to do synchronize since cudaMemcpy is asynchronous for memory copy under 64kb.
				CudaAssert(cudaDeviceSynchronize());
				modelStream.write(reinterpret_cast<const char*>(tempFiltersBuffer), convLayer->GetFiltersBufferSize());
			}
			CudaAssert(cudaFreeHost(tempFiltersBuffer));

			// Writing biases.
			float* tempBiasesBuffer;
			CudaAssert(cudaMallocHost<float>(&tempBiasesBuffer, layers[0]->GetBiasesBufferSize()));
			for (ConvolutionalLayer* convLayer : layers)
			{
				CudaAssert(cudaSetDevice(convLayer->GetIndexInTier()));
				CudaAssert(cudaMemcpy(tempBiasesBuffer, convLayer->GetBiasesBuffer(), convLayer->GetBiasesBufferSize(), cudaMemcpyDeviceToHost));
				// Need to do synchronize since cudaMemcpy is asynchronous for memory copy under 64kb.
				CudaAssert(cudaDeviceSynchronize());
				modelStream.write(reinterpret_cast<const char*>(tempBiasesBuffer), convLayer->GetBiasesBufferSize());
			}
			CudaAssert(cudaFreeHost(tempBiasesBuffer));

			if (saveUpdateBuffers)
			{
				// Writing filters update buffers.
				float* tempFiltersUpdatesBuffer;
				CudaAssert(cudaMallocHost<float>(&tempFiltersUpdatesBuffer, layers[0]->GetFiltersBufferSize()));
				for (ConvolutionalLayer* convLayer : layers)
				{
					CudaAssert(cudaSetDevice(convLayer->GetIndexInTier()));
					CudaAssert(cudaMemcpy(tempFiltersUpdatesBuffer, convLayer->GetFiltersUpdateBuffer(), convLayer->GetFiltersBufferSize(), cudaMemcpyDeviceToHost));
					// Need to do synchronize since cudaMemcpy is asynchronous for memory copy under 64kb.
					CudaAssert(cudaDeviceSynchronize());
					modelStream.write(reinterpret_cast<const char*>(tempFiltersUpdatesBuffer), convLayer->GetFiltersBufferSize());
				}
				CudaAssert(cudaFreeHost(tempFiltersUpdatesBuffer));

				// Writing biases update buffers.
				float* tempBiasesUpdatesBuffer;
				CudaAssert(cudaMallocHost<float>(&tempBiasesUpdatesBuffer, layers[0]->GetBiasesBufferSize()));
				for (ConvolutionalLayer* convLayer : layers)
				{
					CudaAssert(cudaSetDevice(convLayer->GetIndexInTier()));
					CudaAssert(cudaMemcpy(tempBiasesUpdatesBuffer, convLayer->GetBiasesUpdateBuffer(), convLayer->GetBiasesBufferSize(), cudaMemcpyDeviceToHost));
					// Need to do synchronize since cudaMemcpy is asynchronous for memory copy under 64kb.
					CudaAssert(cudaDeviceSynchronize());
					modelStream.write(reinterpret_cast<const char*>(tempBiasesUpdatesBuffer), convLayer->GetBiasesBufferSize());
				}
				CudaAssert(cudaFreeHost(tempBiasesUpdatesBuffer));
			}
		}
		else if (m_layersTiers[tier][0]->GetLayerType() == LayerType::Standard)
		{
			vector<StandardLayer*> layers;
			if (m_layersTiers[tier][0]->GetParallelismMode() == ParallelismMode::Data)
			{
				layers.push_back(static_cast<StandardLayer*>(m_layersTiers[tier][0]));
			}
			else
			{
				for (Layer* layer : m_layersTiers[tier])
				{
					layers.push_back(static_cast<StandardLayer*>(layer));
				}
			}

			// Writing weights.
			float* tempWeightsBuffer;
			CudaAssert(cudaMallocHost<float>(&tempWeightsBuffer, layers[0]->GetWeightsBufferSize()));
			for (StandardLayer* standardLayer : layers)
			{
				CudaAssert(cudaSetDevice(standardLayer->GetIndexInTier()));
				CudaAssert(cudaMemcpy(tempWeightsBuffer, standardLayer->GetWeightsBuffer(), standardLayer->GetWeightsBufferSize(), cudaMemcpyDeviceToHost));
				// Need to do synchronize since cudaMemcpy is asynchronous for memory copy under 64kb.
				CudaAssert(cudaDeviceSynchronize());
				modelStream.write(reinterpret_cast<const char*>(tempWeightsBuffer), standardLayer->GetWeightsBufferSize());
			}
			CudaAssert(cudaFreeHost(tempWeightsBuffer));

			// Writing biases.
			float* tempBiasesBuffer;
			CudaAssert(cudaMallocHost<float>(&tempBiasesBuffer, layers[0]->GetBiasesBufferSize()));
			for (StandardLayer* standardLayer : layers)
			{
				CudaAssert(cudaSetDevice(standardLayer->GetIndexInTier()));
				CudaAssert(cudaMemcpy(tempBiasesBuffer, standardLayer->GetBiasesBuffer(), standardLayer->GetBiasesBufferSize(), cudaMemcpyDeviceToHost));
				// Need to do synchronize since cudaMemcpy is asynchronous for memory copy under 64kb.
				CudaAssert(cudaDeviceSynchronize());
				modelStream.write(reinterpret_cast<const char*>(tempBiasesBuffer), standardLayer->GetBiasesBufferSize());
			}
			CudaAssert(cudaFreeHost(tempBiasesBuffer));

			if (saveUpdateBuffers)
			{
				// Writing weights update buffers.
				float* tempWeightsUpdatesBuffer;
				CudaAssert(cudaMallocHost<float>(&tempWeightsUpdatesBuffer, layers[0]->GetWeightsBufferSize()));
				for (StandardLayer* standardLayer : layers)
				{
					CudaAssert(cudaSetDevice(standardLayer->GetIndexInTier()));
					CudaAssert(cudaMemcpy(tempWeightsUpdatesBuffer, standardLayer->GetWeightsUpdateBuffer(), standardLayer->GetWeightsBufferSize(), cudaMemcpyDeviceToHost));
					// Need to do synchronize since cudaMemcpy is asynchronous for memory copy under 64kb.
					CudaAssert(cudaDeviceSynchronize());
					modelStream.write(reinterpret_cast<const char*>(tempWeightsUpdatesBuffer), standardLayer->GetWeightsBufferSize());
				}
				CudaAssert(cudaFreeHost(tempWeightsUpdatesBuffer));

				// Writing biases update buffers.
				float* tempBiasesUpdatesBuffer;
				CudaAssert(cudaMallocHost<float>(&tempBiasesUpdatesBuffer, layers[0]->GetBiasesBufferSize()));
				for (StandardLayer* standardLayer : layers)
				{
					CudaAssert(cudaSetDevice(standardLayer->GetIndexInTier()));
					CudaAssert(cudaMemcpy(tempBiasesUpdatesBuffer, standardLayer->GetBiasesUpdateBuffer(), standardLayer->GetBiasesBufferSize(), cudaMemcpyDeviceToHost));
					// Need to do synchronize since cudaMemcpy is asynchronous for memory copy under 64kb.
					CudaAssert(cudaDeviceSynchronize());
					modelStream.write(reinterpret_cast<const char*>(tempBiasesUpdatesBuffer), standardLayer->GetBiasesBufferSize());
				}
				CudaAssert(cudaFreeHost(tempBiasesUpdatesBuffer));
			}
		}
	}

	CudaAssert(cudaSetDevice(0));

	modelStream.close();
}

void NeuralNet::SaveModelCheckpoint(string modelFile)
{
	SaveModel(modelFile, true);
}

void NeuralNet::SaveModelForPrediction(string modelFile)
{
	SaveModel(modelFile, false);
}

void NeuralNet::LoadModelCheckpoint(string modelFile)
{
	ifstream modelStream(modelFile, ifstream::in | ifstream::binary);

	for (size_t tier = 0; tier < m_layersTiers.size(); ++tier)
	{
		if (m_layersTiers[tier][0]->GetLayerType() == LayerType::Convolutional)
		{
			vector<ConvolutionalLayer*> layers;
			for (Layer* layer : m_layersTiers[tier])
			{
				layers.push_back(static_cast<ConvolutionalLayer*>(layer));
			}

			// Reading filters.
			float* tempFiltersBuffer;
			CudaAssert(cudaMallocHost<float>(&tempFiltersBuffer, layers[0]->GetFiltersBufferSize()));
			if (layers[0]->GetParallelismMode() == ParallelismMode::Data)
			{
				modelStream.read(reinterpret_cast<char*>(tempFiltersBuffer), layers[0]->GetFiltersBufferSize());
				for (ConvolutionalLayer* convLayer : layers)
				{
					CudaAssert(cudaSetDevice(convLayer->GetIndexInTier()));
					convLayer->CopyFiltersFromHost(tempFiltersBuffer);
				}
			}
			else
			{
				for (ConvolutionalLayer* convLayer : layers)
				{
					CudaAssert(cudaSetDevice(convLayer->GetIndexInTier()));
					modelStream.read(reinterpret_cast<char*>(tempFiltersBuffer), convLayer->GetFiltersBufferSize());
					convLayer->CopyFiltersFromHost(tempFiltersBuffer);
				}
			}
			CudaAssert(cudaFreeHost(tempFiltersBuffer));

			// Reading biases.
			float* tempBiasesBuffer;
			CudaAssert(cudaMallocHost<float>(&tempBiasesBuffer, layers[0]->GetBiasesBufferSize()));
			if (layers[0]->GetParallelismMode() == ParallelismMode::Data)
			{
				modelStream.read(reinterpret_cast<char*>(tempBiasesBuffer), layers[0]->GetBiasesBufferSize());
				for (ConvolutionalLayer* convLayer : layers)
				{
					CudaAssert(cudaSetDevice(convLayer->GetIndexInTier()));
					convLayer->CopyBiasesFromHost(tempBiasesBuffer);
				}
			}
			else
			{
				for (ConvolutionalLayer* convLayer : layers)
				{
					CudaAssert(cudaSetDevice(convLayer->GetIndexInTier()));
					modelStream.read(reinterpret_cast<char*>(tempBiasesBuffer), convLayer->GetBiasesBufferSize());
					convLayer->CopyBiasesFromHost(tempBiasesBuffer);
				}
			}
			CudaAssert(cudaFreeHost(tempBiasesBuffer));

			// Reading filters update buffer.
			float* tempFiltersUpdateBuffer;
			CudaAssert(cudaMallocHost<float>(&tempFiltersUpdateBuffer, layers[0]->GetFiltersBufferSize()));
			if (layers[0]->GetParallelismMode() == ParallelismMode::Data)
			{
				modelStream.read(reinterpret_cast<char*>(tempFiltersUpdateBuffer), layers[0]->GetFiltersBufferSize());
				for (ConvolutionalLayer* convLayer : layers)
				{
					CudaAssert(cudaSetDevice(convLayer->GetIndexInTier()));
					convLayer->CopyFiltersUpdateFromHost(tempFiltersUpdateBuffer);
				}
			}
			else
			{
				for (ConvolutionalLayer* convLayer : layers)
				{
					CudaAssert(cudaSetDevice(convLayer->GetIndexInTier()));
					modelStream.read(reinterpret_cast<char*>(tempFiltersUpdateBuffer), convLayer->GetFiltersBufferSize());
					convLayer->CopyFiltersUpdateFromHost(tempFiltersUpdateBuffer);
				}
			}
			CudaAssert(cudaFreeHost(tempFiltersUpdateBuffer));

			// Reading biases update buffer.
			float* tempBiasesUpdateBuffer;
			CudaAssert(cudaMallocHost<float>(&tempBiasesUpdateBuffer, layers[0]->GetBiasesBufferSize()));
			if (layers[0]->GetParallelismMode() == ParallelismMode::Data)
			{
				modelStream.read(reinterpret_cast<char*>(tempBiasesUpdateBuffer), layers[0]->GetBiasesBufferSize());
				for (ConvolutionalLayer* convLayer : layers)
				{
					CudaAssert(cudaSetDevice(convLayer->GetIndexInTier()));
					convLayer->CopyBiasesUpdateFromHost(tempBiasesUpdateBuffer);
				}
			}
			else
			{
				for (ConvolutionalLayer* convLayer : layers)
				{
					CudaAssert(cudaSetDevice(convLayer->GetIndexInTier()));
					modelStream.read(reinterpret_cast<char*>(tempBiasesUpdateBuffer), convLayer->GetBiasesBufferSize());
					convLayer->CopyBiasesUpdateFromHost(tempBiasesUpdateBuffer);
				}
			}
			CudaAssert(cudaFreeHost(tempBiasesUpdateBuffer));
		}
		else if (m_layersTiers[tier][0]->GetLayerType() == LayerType::Standard)
		{
			vector<StandardLayer*> layers;
			for (Layer* layer : m_layersTiers[tier])
			{
				layers.push_back(static_cast<StandardLayer*>(layer));
			}

			// Reading weights.
			float* tempWeightsBuffer;
			CudaAssert(cudaMallocHost<float>(&tempWeightsBuffer, layers[0]->GetWeightsBufferSize()));
			if (layers[0]->GetParallelismMode() == ParallelismMode::Data)
			{
				modelStream.read(reinterpret_cast<char*>(tempWeightsBuffer), layers[0]->GetWeightsBufferSize());
				for (StandardLayer* standardLayer : layers)
				{
					CudaAssert(cudaSetDevice(standardLayer->GetIndexInTier()));
					standardLayer->CopyWeightsFromHost(tempWeightsBuffer);
				}
			}
			else
			{
				for (StandardLayer* standardLayer : layers)
				{
					CudaAssert(cudaSetDevice(standardLayer->GetIndexInTier()));
					modelStream.read(reinterpret_cast<char*>(tempWeightsBuffer), standardLayer->GetWeightsBufferSize());
					standardLayer->CopyWeightsFromHost(tempWeightsBuffer);
				}
			}
			CudaAssert(cudaFreeHost(tempWeightsBuffer));

			// Reading biases.
			float* tempBiasesBuffer;
			CudaAssert(cudaMallocHost<float>(&tempBiasesBuffer, layers[0]->GetBiasesBufferSize()));
			if (layers[0]->GetParallelismMode() == ParallelismMode::Data)
			{
				modelStream.read(reinterpret_cast<char*>(tempBiasesBuffer), layers[0]->GetBiasesBufferSize());
				for (StandardLayer* standardLayer : layers)
				{
					CudaAssert(cudaSetDevice(standardLayer->GetIndexInTier()));
					standardLayer->CopyBiasesFromHost(tempBiasesBuffer);
				}
			}
			else
			{
				for (StandardLayer* standardLayer : layers)
				{
					CudaAssert(cudaSetDevice(standardLayer->GetIndexInTier()));
					modelStream.read(reinterpret_cast<char*>(tempBiasesBuffer), standardLayer->GetBiasesBufferSize());
					standardLayer->CopyBiasesFromHost(tempBiasesBuffer);
				}
			}
			CudaAssert(cudaFreeHost(tempBiasesBuffer));

			// Reading weights update buffer.
			float* tempWeightsUpdateBuffer;
			CudaAssert(cudaMallocHost<float>(&tempWeightsUpdateBuffer, layers[0]->GetWeightsBufferSize()));
			if (layers[0]->GetParallelismMode() == ParallelismMode::Data)
			{
				modelStream.read(reinterpret_cast<char*>(tempWeightsUpdateBuffer), layers[0]->GetWeightsBufferSize());
				for (StandardLayer* standardLayer : layers)
				{
					CudaAssert(cudaSetDevice(standardLayer->GetIndexInTier()));
					standardLayer->CopyWeightsUpdateFromHost(tempWeightsUpdateBuffer);
				}
			}
			else
			{
				for (StandardLayer* standardLayer : layers)
				{
					CudaAssert(cudaSetDevice(standardLayer->GetIndexInTier()));
					modelStream.read(reinterpret_cast<char*>(tempWeightsUpdateBuffer), standardLayer->GetWeightsBufferSize());
					standardLayer->CopyWeightsUpdateFromHost(tempWeightsUpdateBuffer);
				}
			}
			CudaAssert(cudaFreeHost(tempWeightsUpdateBuffer));

			// Reading biases update buffer.
			float* tempBiasesUpdateBuffer;
			CudaAssert(cudaMallocHost<float>(&tempBiasesUpdateBuffer, layers[0]->GetBiasesBufferSize()));
			if (layers[0]->GetParallelismMode() == ParallelismMode::Data)
			{
				modelStream.read(reinterpret_cast<char*>(tempBiasesUpdateBuffer), layers[0]->GetBiasesBufferSize());
				for (StandardLayer* standardLayer : layers)
				{
					CudaAssert(cudaSetDevice(standardLayer->GetIndexInTier()));
					standardLayer->CopyBiasesUpdateFromHost(tempBiasesUpdateBuffer);
				}
			}
			else
			{
				for (StandardLayer* standardLayer : layers)
				{
					CudaAssert(cudaSetDevice(standardLayer->GetIndexInTier()));
					modelStream.read(reinterpret_cast<char*>(tempBiasesUpdateBuffer), standardLayer->GetBiasesBufferSize());
					standardLayer->CopyBiasesUpdateFromHost(tempBiasesUpdateBuffer);
				}
			}
			CudaAssert(cudaFreeHost(tempBiasesUpdateBuffer));
		}
	}

	CudaAssert(cudaSetDevice(0));

	modelStream.close();
}

void NeuralNet::LoadModelForPrediction(string modelFile)
{
	ifstream modelStream(modelFile, ifstream::in | ifstream::binary);

	for (size_t tier = 0; tier < m_layersTiers.size(); ++tier)
	{
		if (m_layersTiers[tier][0]->GetLayerType() == LayerType::Convolutional)
		{
			ConvolutionalLayer* convLayer = static_cast<ConvolutionalLayer*>(m_layersTiers[tier][0]);

			// Reading filters.
			float* tempFiltersBuffer;
			CudaAssert(cudaMallocHost<float>(&tempFiltersBuffer, convLayer->GetFiltersBufferSize()));
			modelStream.read(reinterpret_cast<char*>(tempFiltersBuffer), convLayer->GetFiltersBufferSize());
			convLayer->CopyFiltersFromHost(tempFiltersBuffer);
			CudaAssert(cudaFreeHost(tempFiltersBuffer));

			// Reading biases.
			float* tempBiasesBuffer;
			CudaAssert(cudaMallocHost<float>(&tempBiasesBuffer, convLayer->GetBiasesBufferSize()));
			modelStream.read(reinterpret_cast<char*>(tempBiasesBuffer), convLayer->GetBiasesBufferSize());
			convLayer->CopyBiasesFromHost(tempBiasesBuffer);
			CudaAssert(cudaFreeHost(tempBiasesBuffer));
		}
		else if (m_layersTiers[tier][0]->GetLayerType() == LayerType::Standard)
		{
			StandardLayer* standardLayer = static_cast<StandardLayer*>(m_layersTiers[tier][0]);

			// Reading weights.
			float* tempWeightsBuffer;
			CudaAssert(cudaMallocHost<float>(&tempWeightsBuffer, standardLayer->GetWeightsBufferSize()));
			modelStream.read(reinterpret_cast<char*>(tempWeightsBuffer), standardLayer->GetWeightsBufferSize());
			standardLayer->CopyWeightsFromHost(tempWeightsBuffer);
			CudaAssert(cudaFreeHost(tempWeightsBuffer));

			// Reading biases.
			float* tempBiasesBuffer;
			CudaAssert(cudaMallocHost<float>(&tempBiasesBuffer, standardLayer->GetBiasesBufferSize()));
			modelStream.read(reinterpret_cast<char*>(tempBiasesBuffer), standardLayer->GetBiasesBufferSize());
			standardLayer->CopyBiasesFromHost(tempBiasesBuffer);
			CudaAssert(cudaFreeHost(tempBiasesBuffer));
		}
	}

	modelStream.close();
}