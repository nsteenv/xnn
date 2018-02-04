// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network output layer, used in tests.
// Created: 02/21/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../../include/testingutils.cuh"
#include "../../../../../neuralnetwork/layers/include/layer.cuh"
#include "../../../../../neuralnetwork/layers/include/outputlayer.cuh"

using namespace std;

class MockOutputLayer : public Layer
{
private:
	// Loss function type.
	LossFunctionType m_lossFunctionType;

	// Total value of loss function for current batch.
	float m_loss;

	// Total accuracy for current batch.
	float m_accuracy;

	// Labels buffer for input data samples.
	uint* m_dataLabels;

	// Labels buffer size.
	size_t m_labelsBufferSize;

	// Log-likelihood values, for logistic regression loss function.
	float* m_logLikelihoods;

	// Score values for each data sample.
	float* m_scores;

	// Size for loss and score buffers.
	size_t m_lossBuffersSize;

	// Should we calculate multiple guess accuracy.
	bool m_calculateMultipleGuessAccuracy;

	// Number of guesses for predicting correct output value, if we are calculating multiple guess accuracy.
	uint m_numGuesses;

	// Multiple guess score values for each data sample.
	float* m_multipleGuessScores;

	// Multiple guess accuracy for current batch.
	float m_multipleGuessAccuracy;

	// Should we generate random input gradients, for test purposes.
	bool m_generateRandomInputGradients;

	// Calculates loss and scores for logistic regression loss function.
	void CalculateLogisticRegressionLoss();

	// Forward prop for logistic regression loss function.
	void LogisticRegressionForwardProp();

public:
	// Constructor.
	MockOutputLayer(uint inputDataSize, uint inputDataCount, LossFunctionType lossFunctionType, bool calculateMultipleGuessAccuracy, uint numGuesses,
		bool generateRandomInputGradients = false);

	// Loads labels for input data samples.
	void LoadDataLabels(vector<uint> dataLabels);

	// Gets total value of loss function for current batch.
	float GetLoss() { return m_loss; }

	// Gets current batch total accuracy.
	float GetAccuracy() { return m_accuracy; }

	// Gets current batch multiple guess total accuracy.
	float GetMultipleGuessAccuracy() { return m_multipleGuessAccuracy; }

	// Gets log-likelihoods.
	float* GetLogLikelihoods() { return m_logLikelihoods; }

	// Gets scores.
	float* GetScores() { return m_scores; }

	// Gets multiple guess scores.
	float* GetMultipleGuessScores() { return m_multipleGuessScores; }

	// Destructor.
	virtual ~MockOutputLayer();

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};