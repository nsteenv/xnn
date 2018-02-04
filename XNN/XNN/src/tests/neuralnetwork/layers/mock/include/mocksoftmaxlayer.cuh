// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network softmax layer, used in tests.
// Created: 02/20/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../../include/testingutils.cuh"
#include "../../../../../neuralnetwork/layers/include/layer.cuh"
#include "../../../../../neuralnetwork/layers/include/outputlayer.cuh"

using namespace std;

class MockSoftMaxLayer : public Layer
{
private:
	// Buffer to store maximums of input activations for each input sample.
	float* m_inputActivationsMaxBuffer;

	// Buffer to store sums of exponentials of input activations for each input sample.
	float* m_exponentialsSumBuffer;

	// Stabilizes inputs to prevent overflow. We acomplish this by substracting maximum value of input activations (for each input sample)
	// from all the input activations, before computing the exponentials.
	void StabilizeInputs();

	// Calculates soft maximums.
	void CalculateSoftMaximums();

	// Does backward prop in case of logistic regression output layer.
	void LogisticRegressionBackwardProp(uint* dataLabels);

public:
	// Constructor.
	MockSoftMaxLayer(uint inputDataSize, uint inputDataCount);

	// Destructor.
	virtual ~MockSoftMaxLayer();

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Loads activation gradients to layer.
	void LoadActivationGradients();

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};