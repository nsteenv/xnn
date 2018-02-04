// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network standard layer, used in tests.
// Created: 02/13/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../../include/testingutils.cuh"
#include "../../../../../neuralnetwork/include/activationfunctions.cuh"
#include "../../../../../neuralnetwork/layers/include/layer.cuh"

using namespace std;

class MockStandardLayer : public Layer
{
private:
	// Should we output activations to GPU memory.
	bool m_outputToGpuMemory;

	// Number of neurons in standard layer.
	uint m_numNeurons;

	// Number of weights per neuron.
	uint m_numWeightsPerNeuron;

	// Weights buffer.
	float* m_weightsBuffer;

	// Weights buffer size.
	size_t m_weightsBufferSize;

	// Weights gradients buffer.
	float* m_weightsGradientsBuffer;

	// Weights update buffer.
	float* m_weightsUpdateBuffer;

	// Weights update momentum.
	float m_weightsUpdateMomentum;

	// Weights update decay.
	float m_weightsUpdateDecay;

	// Weights update learning rate progress step.
	float m_weightsUpdateLearningRateProgressStep;

	// Weights update starting learning rate.
	float m_weightsUpdateStartingLearningRate;

	// Weights update learning rate update factor.
	float m_weightsUpdateLearningRateUpdateFactor;

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

	// Activation type of this layer.
	ActivationType m_activationType;

	// Preactivations buffer.
	float* m_preactivationDataBuffer;

	// Preactivations gradients buffer.
	float* m_preactivationGradientsBuffer;

	// Initializes weigths.
	void InitializeWeights(float weightsDeviation);

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
	MockStandardLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numNeurons, float weightsDeviation,
		float biasesInitialValue, float weightsUpdateMomentum, float weightsUpdateDecay, float weightsUpdateLearningRateProgressStep,
		float weightsUpdateStartingLearningRate, float weightsUpdateLearningRateUpdateFactor, float biasesUpdateMomentum, float biasesUpdateDecay,
		float biasesUpdateLearningRateProgressStep,float biasesUpdateStartingLearningRate, float biasesUpdateLearningRateUpdateFactor, ActivationType activationType,
		bool outputToGpuMemory = false);

	// Gets weights buffer.
	float* GetWeightsBuffer() { return m_weightsBuffer; }

	// Gets weights buffer size.
	size_t GetWeightsBufferSize() { return m_weightsBufferSize; }

	// Gets biases buffer.
	float* GetBiasesBuffer() { return m_biasesBuffer; }

	// Gets biases buffer size.
	size_t GetBiasesBufferSize() { return m_biasesBufferSize; }

	// Gets weights gradients buffer.
	float* GetWeightsGradientsBuffer() const { return m_weightsGradientsBuffer; }

	// Gets biases gradients buffer.
	float* GetBiasesGradientsBuffer() const { return m_biasesGradientsBuffer; }

	// Destructor.
	virtual ~MockStandardLayer();

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Loads activation gradients to layer.
	void LoadActivationGradients();

	// Does backward propagation through layer.
	virtual void DoBackwardProp();

	// Updates layer's parameters (weights, biases, etc.)
	virtual void UpdateLayerParameters(float learningProgress);
};