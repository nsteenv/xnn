// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network convolutional layer.
// Created: 01/03/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "layer.cuh"
#include "../../include/activationfunctions.cuh"
#include "../../../utils/include/config.cuh"

using namespace std;

/*
	From Wikipedia:

	The Convolutional layer is the core building block of a CNN. The layer's parameters consist of a set of learnable filters (or kernels),
	which have a small receptive field, but extend through the full depth of the input volume. During the forward pass, each filter is convolved across
	the width and height of the input volume, computing the dot product between the entries of the filter and the input and producing a 2-dimensional
	activation map of that filter. As a result, the network learns filters that activate when they see some specific type of feature at some spatial
	position in the input.
*/
class ConvolutionalLayer : public Layer
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

	// Helper buffer for calculating filters gradients per chunk of preactivation gradients.
	float* m_filtersGradientsPerChunkBuffer;

	// Size of helper buffer for calculating filters gradients per chunk of preactivation gradients.
	size_t m_filtersGradientsPerChunkBufferSize;

	// How many preactivation gradients are per chunk width for calculation of filters gradients per chunk.
	uint m_preactivationGradientsPerChunkWidth;

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

	// Buffer for holding partial sums for calculating biases gradients.
	float* m_biasesGradientsPartialSumsBuffer;

	// How many summations will be done per thread for partial sums for calculating biases gradients.
	static const uint c_biasesGradientsSumsPerThread;

	// How many threads will be used pre block to calculate partial sums for one filter bias gradient.
	static const uint c_biasesGradientsPartialSumThreadsPerBlock;

	// How many blocks will be used to calculate partial sums for one filter bias gradient.
	uint m_biasesGradientsPartialSumBlocks;

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

	// Calculates preactivations.
	void CalculatePreactivations();

	// Adds biases to preactivations.
	void AddBiases();

	// Calculates activations.
	void CalculateActivations();

	// Calculates gradients of biases.
	void CalculateBiasesGradients();

	// Calculates gradients of filters' weights.
	void CalculateWeightsGradients();

	// Calculates gradients of inputs.
	void CalculateInputGradients();

	// Calculates gradients of preactivations.
	void CalculatePreactivationsGradients();

	// Reinitializes layer when input data count changes.
	virtual void Reinitialize(uint newInputDataCount);

public:
	// Constructor.
	ConvolutionalLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, uint indexInTier, uint tierSize,
		uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, bool holdsInputData, uint numFilters, uint filterWidth,
		uint filterHeight, uint numFilterChannels, bool initializeWeights, float weightsDeviation, bool initializeBiases, float biasesInitialValue,
		float filtersUpdateMomentum, float filtersUpdateDecay, float filtersUpdateLearningRateProgressStep, float filtersUpdateStartingLearningRate,
		float filtersUpdateLearningRateUpdateFactor, float biasesUpdateMomentum, float biasesUpdateDecay, float biasesUpdateLearningRateProgressStep,
		float biasesUpdateStartingLearningRate, float biasesUpdateLearningRateUpdateFactor, int paddingX, int paddingY, uint stride,
		ActivationType activationType, bool holdsActivationGradients);

	// Copies filters from host buffer.
	void CopyFiltersFromHost(float* hostFiltersBuffer);

	// Copies filters update buffer from host buffer.
	void CopyFiltersUpdateFromHost(float* hostFiltersUpdateBuffer);

	// Copies biases from host buffer.
	void CopyBiasesFromHost(float* hostBiasesBuffer);

	// Copies biases update buffer from host buffer.
	void CopyBiasesUpdateFromHost(float* hostBiasesUpdateBuffer);

	// Gets filters buffer.
	float* GetFiltersBuffer() const { return m_filtersBuffer; }

	// Gets filters update buffer.
	float* GetFiltersUpdateBuffer() const { return m_filtersUpdateBuffer; }

	// Gets filters buffer size.
	size_t GetFiltersBufferSize() { return m_filtersBufferSize; }

	// Gets biases buffer.
	float* GetBiasesBuffer() const { return m_biasesBuffer; }

	// Gets biases update buffer.
	float* GetBiasesUpdateBuffer() const { return m_biasesUpdateBuffer; }

	// Gets biases buffer size.
	size_t GetBiasesBufferSize() { return m_biasesBufferSize; }

	// Gets filters gradients buffer.
	float* GetFiltersGradientsBuffer() const { return m_filtersGradientsBuffer; }

	// Gets biases gradients buffer.
	float* GetBiasesGradientsBuffer() const { return m_biasesGradientsBuffer; }

	// Destructor.
	virtual ~ConvolutionalLayer();

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Does backward propagation through layer.
	virtual void DoBackwardProp();

	// Updates layer's parameters (filters' weights, biases, etc.)
	virtual void UpdateLayerParameters(float learningProgress);
};