// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network dropout layer, used in tests.
// Created: 02/16/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../../include/testingutils.cuh"
#include "../../../../../neuralnetwork/layers/include/layer.cuh"

using namespace std;

class MockDropoutLayer : public Layer
{
private:
	// Dropout filter.
	float* m_dropoutFilter;

	// Biases buffer size.
	size_t m_dropoutFilterSize;

	// Probability for dropping each activity.
	float m_dropProbability;

	// Creates dropout filter.
	void CreateDropoutFilter();

	// Applies dropout filter.
	void ApplyDropoutFilter();

public:
	// Constructor.
	MockDropoutLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, float dropProbability);

	// Gets dropout filter.
	float* GetDropoutFilter() { return m_dropoutFilter; }

	// Destructor.
	virtual ~MockDropoutLayer();

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Loads activation gradients to layer.
	void LoadActivationGradients();

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};