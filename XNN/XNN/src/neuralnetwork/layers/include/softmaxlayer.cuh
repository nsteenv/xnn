// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network softmax layer.
// Created: 02/20/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "layer.cuh"
#include "outputlayer.cuh"
#include "../../../utils/include/config.cuh"

using namespace std;

/*
	Soft Max layer calculates soft maximums of input activations, so they som to 1 and can be used as probabilities of prediction.
*/
class SoftMaxLayer : public Layer
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
	SoftMaxLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, uint inputDataSize,
		uint inputDataCount, bool holdsInputData);

	// Destructor.
	virtual ~SoftMaxLayer();

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};