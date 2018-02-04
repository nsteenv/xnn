// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network convolutional layer, used in tests.
// Created: 01/27/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../../include/testingutils.cuh"
#include "../../../../../neuralnetwork/include/activationfunctions.cuh"
#include "../../../../../neuralnetwork/layers/include/layer.cuh"

using namespace std;

class MockConvolutionalLayer : public Layer
{
private:
	// Number of convolutional filters.
	uint m_numFilters;

	// Width of a filter.
	uint m_filterWidth;

	// Height of a filter.
	uint m_filterHeight;

	// Size of a filter.
	uint m_filterSize;

	// Number of channels per filter.
	uint m_numFilterChannels;

	// Filters buffer.
	float* m_filtersBuffer;

	// Filters buffer size.
	size_t m_filtersBufferSize;

	// Filters gradients buffer.
	float* m_filtersGradientsBuffer;

	// Filters update buffer.
	float* m_filtersUpdateBuffer;

	// Filters update momentum.
	float m_filtersUpdateMomentum;

	// Filters update decay.
	float m_filtersUpdateDecay;

	// Filters update learning rate progress step.
	float m_filtersUpdateLearningRateProgressStep;

	// Filters update starting learning rate.
	float m_filtersUpdateStartingLearningRate;

	// Filters update learning rate update factor.
	float m_filtersUpdateLearningRateUpdateFactor;

	// Biases buffer.
	float* m_biasesBuffer;

	// Biases buffer size.
	size_t m_biasesBufferSize;

	// Biases gradients buffer.
	float* m_biasesGradientsBuffer;

	// Biases update buffer.
	float* m_biasesUpdateBuffer;

	// Biases update momentum.
	float m_biasesUpdateMomentum;

	// Biases update decay.
	float m_biasesUpdateDecay;

	// Biases update learning rate progress step.
	float m_biasesUpdateLearningRateProgressStep;

	// Biases update starting learning rate.
	float m_biasesUpdateStartingLearningRate;

	// Biases update learning rate update factor.
	float m_biasesUpdateLearningRateUpdateFactor;

	// Padding in dimension X.
	int m_paddingX;

	// Padding in dimension Y.
	int m_paddingY;

	// Stride for patching.
	uint m_stride;

	// Number of patches to apply filters on in dimension X.
	uint m_numPatchesX;

	// Number of patches to apply filters on in dimension Y.
	uint m_numPatchesY;

	// Activation type of this layer.
	ActivationType m_activationType;

	// Preactivations buffer.
	float* m_preactivationDataBuffer;

	// Preactivations gradients buffer.
	float* m_preactivationGradientsBuffer;

	// Initializes filters weigths.
	void InitializeFilterWeights(float weightsDeviation);

	// Initializes buffer with initial values.
	void InitializeBuffer(float* buffer, size_t bufferSize, float initialValue);

	// Calculates preactivations.
	void CalculatePreactivations();

	// Adds biases to preactivations.
	void AddBiases();

	// Calculates activations.
	void CalculateActivations();

	// Calculates gradients of biases.
	void CalculateBiasesGradients();

	// Calculates gradients of weights.
	void CalculateWeightsGradients();

	// Calculates gradients of inputs.
	void CalculateInputGradients();

	// Calculates gradients of preactivations.
	void CalculatePreactivationsGradients();

public:
	// Constructor.
	MockConvolutionalLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numFilters, uint filterWidth,
		uint filterHeight, uint numFilterChannels, float weightsDeviation, float biasesInitialValue, float filtersUpdateMomentum, float filtersUpdateDecay,
		float filtersUpdateLearningRateProgressStep, float filtersUpdateStartingLearningRate, float filtersUpdateLearningRateUpdateFactor,
		float biasesUpdateMomentum, float biasesUpdateDecay, float biasesUpdateLearningRateProgressStep, float biasesUpdateStartingLearningRate,
		float biasesUpdateLearningRateUpdateFactor, int paddingX, int paddingY, uint stride, ActivationType activationType);

	// Gets filters buffer.
	float* GetFiltersBuffer() const { return m_filtersBuffer; }

	// Gets filters buffer size.
	size_t GetFiltersBufferSize() { return m_filtersBufferSize; }

	// Gets biases buffer.
	float* GetBiasesBuffer() const { return m_biasesBuffer; }

	// Gets biases buffer size.
	size_t GetBiasesBufferSize() { return m_biasesBufferSize; }

	// Gets filters gradients buffer.
	float* GetFiltersGradientsBuffer() const { return m_filtersGradientsBuffer; }

	// Gets biases gradients buffer.
	float* GetBiasesGradientsBuffer() const { return m_biasesGradientsBuffer; }

	// Gets input data buffer.
	float* GetInputDataBuffer() const { return m_inputDataBuffer; }

	// Destructor.
	virtual ~MockConvolutionalLayer();

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Loads activation gradients to layer.
	void LoadActivationGradients();

	// Does backward propagation through layer.
	virtual void DoBackwardProp();

	// Updates layer's parameters (filters, biases, etc.)
	virtual void UpdateLayerParameters(float learningProgress);
};