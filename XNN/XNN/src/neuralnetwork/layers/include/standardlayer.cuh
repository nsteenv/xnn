// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network standard layer.
// Created: 01/17/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "layer.cuh"
#include "../../include/activationfunctions.cuh"
#include "../../../utils/include/config.cuh"

using namespace std;

/*
	Standard neural network layer, with neurons and weights.
*/
class StandardLayer : public Layer
{
private:
	// Handle for cuBLAS operations.
	cublasHandle_t m_cublasHandle;

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

	// Reinitializes layer when input data count changes.
	virtual void Reinitialize(uint newInputDataCount);

public:
	// Constructor.
	StandardLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, cublasHandle_t cublasHandle,
		uint indexInTier, uint tierSize, uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, bool holdsInputData, uint numNeurons,
		bool initializeWeights, float weightsDeviation, bool initializeBiases, float biasesInitialValue, float weightsUpdateMomentum, float weightsUpdateDecay,
		float weightsUpdateLearningRateProgressStep, float weightsUpdateStartingLearningRate, float weightsUpdateLearningRateUpdateFactor, float biasesUpdateMomentum,
		float biasesUpdateDecay, float biasesUpdateLearningRateProgressStep, float biasesUpdateStartingLearningRate, float biasesUpdateLearningRateUpdateFactor,
		ActivationType activationType, bool holdsActivationGradients);

	// Copies weights from host buffer.
	void CopyWeightsFromHost(float* hostWeightsBuffer);

	// Copies weights update buffer from host buffer.
	void CopyWeightsUpdateFromHost(float* hostWeightsUpdateBuffer);

	// Copies biases from host buffer.
	void CopyBiasesFromHost(float* hostBiasesBuffer);

	// Copies biases update buffer from host buffer.
	void CopyBiasesUpdateFromHost(float* hostBiasesUpdateBuffer);

	// Gets weights buffer.
	float* GetWeightsBuffer() { return m_weightsBuffer; }

	// Gets weights update buffer.
	float* GetWeightsUpdateBuffer() const { return m_weightsUpdateBuffer; }

	// Gets weights buffer size.
	size_t GetWeightsBufferSize() { return m_weightsBufferSize; }

	// Gets biases buffer.
	float* GetBiasesBuffer() { return m_biasesBuffer; }

	// Gets biases update buffer.
	float* GetBiasesUpdateBuffer() const { return m_biasesUpdateBuffer; }

	// Gets biases buffer size.
	size_t GetBiasesBufferSize() { return m_biasesBufferSize; }

	// Gets weights gradients buffer.
	float* GetWeightsGradientsBuffer() const { return m_weightsGradientsBuffer; }

	// Gets biases gradients buffer.
	float* GetBiasesGradientsBuffer() const { return m_biasesGradientsBuffer; }

	// Destructor.
	virtual ~StandardLayer();

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Does backward propagation through layer.
	virtual void DoBackwardProp();

	// Updates layer's parameters (weights, biases, etc.)
	virtual void UpdateLayerParameters(float learningProgress);
};