// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Abstract neural network layer.
// Created: 12/27/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <chrono>
#include <random>
#include <vector>

#include "../../include/matrixoperations.cuh"
#include "../../../utils/include/utils.cuh"

using namespace std;

// Layer types.
enum class LayerType
{
	Input,
	Convolutional,
	ResponseNormalization,
	MaxPool,
	Standard,
	Dropout,
	SoftMax,
	Output
};

// Parallelism modes.
enum class ParallelismMode
{
	Data,
	Model
};

// Propagation mode.
enum class PropagationMode
{
	Train,
	Test,
	Featurization
};

/*
	Base definition of layer.
*/
class Layer
{
protected:
	// Previous layers.
	vector<Layer*> m_prevLayers;

	// Next layers.
	vector<Layer*> m_nextLayers;

	// Layer type.
	LayerType m_layerType;

	// Layer parallelism mode.
	ParallelismMode m_parallelismMode;

	// Stream this layer uses for device calculations.
	cudaStream_t m_deviceCalculationStream;

	// Stream this layer uses for device memory operations.
	cudaStream_t m_deviceMemoryStream;

	// Index of this layer in his respective tier.
	uint m_indexInTier;

	// Size of this layer's respective tier.
	uint m_tierSize;

	// Index in his respective tier of previous layer whose activations we are working on,
	// used only in layers with model parallelism when they are preceded by layers with data parallelism.
	int m_inputLayerIndexInTier;

	// Number of input data channels.
	uint m_inputNumChannels;

	// Input data width.
	uint m_inputDataWidth;

	// Input data height.
	uint m_inputDataHeight;

	// Input data size.
	uint m_inputDataSize;

	// Count of input data.
	uint m_inputDataCount;

	// Input data buffer.
	float* m_inputDataBuffer;

	// Input gradients buffer.
	float* m_inputGradientsBuffer;

	// Input buffer size.
	size_t m_inputBufferSize;

	// Whether layer holds input data, or just points to buffer in another layer.
	bool m_holdsInputData;

	// Number of activation data channels.
	uint m_activationNumChannels;

	// Activation data width.
	uint m_activationDataWidth;

	// Activation data height.
	uint m_activationDataHeight;

	// Activation data size.
	uint m_activationDataSize;

	// Activations buffer.
	float* m_activationDataBuffer;

	// Activations gradient buffer.
	float* m_activationGradientsBuffer;

	// Helper buffer for collecting activations gradients, used only in layers connected with layers with different index in tier.
	float* m_activationGradientsHelpBuffer;

	// Whether layer holds activation gradients, or just points to buffer in another layer.
	bool m_holdsActivationGradients;

	// Activations buffer size.
	size_t m_activationBufferSize;

	// Does the common load of inputs.
	void CommonLoadInputs();

	// Initializes parameters with values from standard distribution with specified deviation.
	void InitializeParamsFromDistribution(float* paramsBuffer, size_t paramsBufferSize, float deviation);

	// Initializes parameters with specified initial values.
	void InitializeParamsToValue(float* paramsBuffer, size_t paramsBufferSize, float initialValue);

	// Does the common update of layers parameters.
	void CommonUpdateLayerParameters(float learningProgress, float* weightsBuffer, float* weightsGradientsBuffer, float* weightsUpdateBuffer,
		uint weightsGradientsBufferLength, float weightsUpdateMomentum, float weightsUpdateLearningRateProgressStep, float weightsUpdateStartingLearningRate,
		float weightsUpdateLearningRateUpdateFactor, float weightsUpdateDecay, float* biasesBuffer, float* biasesGradientsBuffer, float* biasesUpdateBuffer,
		uint biasesGradientsBufferLength, float biasesUpdateMomentum, float biasesUpdateLearningRateProgressStep, float biasesUpdateStartingLearningRate,
		float biasesUpdateLearningRateUpdateFactor, float biasesUpdateDecay);
	
	// Reinitializes layer when input data count changes.
	virtual void Reinitialize(uint newInputDataCount);

public:
	// Base constructor.
	Layer();

	// Base destructor.
	virtual ~Layer();

	// Gets previous layers.
	vector<Layer*>& GetPrevLayers() { return m_prevLayers; }

	// Gets next layers.
	vector<Layer*>& GetNextLayers() { return m_nextLayers; }

	// Adds previous layer.
	void AddPrevLayer(Layer* prevLayer) { m_prevLayers.push_back(prevLayer); }

	// Adds next layer.
	void AddNextLayer(Layer* nextLayer) { m_nextLayers.push_back(nextLayer); }

	// Gets layer type.
	LayerType GetLayerType() const { return m_layerType; }

	// Gets layer parallelism mode.
	ParallelismMode GetParallelismMode() const { return m_parallelismMode; }

	// Gets layer index in tier.
	uint GetIndexInTier() const { return m_indexInTier; }

	// Gets layer tier size.
	uint GetTierSize() const { return m_tierSize; }

	// Returns whether this layer holds input data.
	bool HoldsInputData() const { return m_holdsInputData; }

	// Gets index in tier of previous layer whose activations we are working on, in case of data -> model parallelism.
	int GetInputLayerIndexInTier() const { return m_inputLayerIndexInTier; }

	// Increases index in tier of previous layer whose activations we are working on, in case of data -> model parallelism.
	void IncreaseInputLayerIndexInTier() { ++m_inputLayerIndexInTier; }

	// Resets index in tier of previous layer whose activations we are working on, in case of data -> model parallelism.
	void ResetInputLayerIndexInTier() { m_inputLayerIndexInTier = -1; }

	// Gets input gradients buffer.
	float* GetInputGradientsBuffer() const { return m_inputGradientsBuffer; }

	// Gets number of activation data channels.
	uint GetActivationNumChannels() const { return m_activationNumChannels; }

	// Gets activation data width.
	uint GetActivationDataWidth() const { return m_activationDataWidth; }

	// Gets activation data height.
	uint GetActivationDataHeight() const { return m_activationDataHeight; }

	// Gets activation data size.
	uint GetActivationDataSize() const { return m_activationDataSize; }

	// Gets activation data buffer.
	float* GetActivationDataBuffer() const { return m_activationDataBuffer; }

	// Gets count of activation data (it is same as input data count, what gets into the layer gets out of the layer).
	uint GetActivationDataCount() const { return m_inputDataCount; }

	// Gets the activation buffer size.
	size_t GetActivationBufferSize() const { return m_activationBufferSize; }

	// Returns whether this layer holds activation gradients, or just points to buffer in another layer.
	bool HoldsActivationGradients() const { return m_holdsActivationGradients; }

	// Synchronizes operations in device calculation stream.
	void SynchronizeCalculations();

	// Synchronizes operations in device memory stream.
	void SynchronizeMemoryOperations();

	// Loads inputs to layer.
	virtual void LoadInputs() = 0;

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode) = 0;

	// Loads activation gradients to layer.
	void LoadActivationGradients();

	// Does backward propagation through layer.
	virtual void DoBackwardProp() = 0;

	// Updates layer's parameters (weights, biases, etc.)
	// Does nothing by default, implement in appropriate layers.
	virtual void UpdateLayerParameters(float learningProgress) {}
};