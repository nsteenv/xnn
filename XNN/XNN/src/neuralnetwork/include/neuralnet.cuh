// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network.
// Created: 12/27/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <fstream>

#include "../layers/include/convolutionallayer.cuh"
#include "../layers/include/dropoutlayer.cuh"
#include "../layers/include/inputlayer.cuh"
#include "../layers/include/maxpoollayer.cuh"
#include "../layers/include/outputlayer.cuh"
#include "../layers/include/responsenormalizationlayer.cuh"
#include "../layers/include/softmaxlayer.cuh"
#include "../layers/include/standardlayer.cuh"

using namespace std;

class NeuralNet
{
private:
	// Layers are organized into tiers.
	// Tier contains layers of same type which work in parallel.
	vector<vector<Layer*> > m_layersTiers;

	// Size of network tier with maximal size.
	size_t m_maxNetworkTierSize;

	// Streams this network uses for device calculations.
	vector<cudaStream_t> m_deviceCalculationStreams;

	// Streams this network uses for device memory operations.
	vector<cudaStream_t> m_deviceMemoryStreams;

	// Handles this network uses for cuBLAS operations.
	vector<cublasHandle_t> m_cublasHandles;

	// Buffers for cuRAND states this network uses for cuRAND operations.
	vector<curandState*> m_curandStatesBuffers;

	// Initializes states in cuRAND buffer.
	void InitCurandStatesBuffer(curandState* curandStatesBuffer, cudaStream_t deviceCalculationStream);

	// Saves trained network model to disk.
	void SaveModel(string modelFile, bool saveUpdateBuffers);

public:
	// Constructs network with specified capacity.
	NeuralNet(size_t maxNetworkTierSize);

	// Destructor.
	~NeuralNet();

	// Gets input layer of the network.
	InputLayer* GetInputLayer() const { return static_cast<InputLayer*>(m_layersTiers[0][0]); }

	// Gets output layer of the network.
	OutputLayer* GetOutputLayer() const { return static_cast<OutputLayer*>(m_layersTiers.back()[0]); }

	// Gets layer tiers.
	vector<vector<Layer*> >& GetLayerTiers() { return m_layersTiers; }

	// Gets streams this network uses for device calculations.
	vector<cudaStream_t>& GetDeviceCalculationStreams() { return m_deviceCalculationStreams; }

	// Gets streams this network uses for device memory operations.
	vector<cudaStream_t>& GetDeviceMemoryStreams() { return m_deviceMemoryStreams; }

	// Gets handles this network uses for cuBLAS operations.
	vector<cublasHandle_t>& GetCublasHandles() { return m_cublasHandles; }

	// Gets buffers for cuRAND states this network uses for cuRAND operations.
	vector<curandState*>& GetCurandStatesBuffers() { return m_curandStatesBuffers; }

	// Adds layer tier.
	void AddLayersTier(const vector<Layer*>& layerTier) { m_layersTiers.push_back(layerTier); }

	// Gets size of network tier with maximal size.
	size_t GetMaxNetworkTierSize() const { return m_maxNetworkTierSize; }

	// Saves trained network model checkpoint to disk.
	void SaveModelCheckpoint(string modelFile);

	// Saves trained network model for prediction to disk.
	void SaveModelForPrediction(string modelFile);

	// Loads saved network model checkpoint from disk.
	void LoadModelCheckpoint(string modelFile);

	// Loads saved network model for prediction from disk.
	void LoadModelForPrediction(string modelFile);
};