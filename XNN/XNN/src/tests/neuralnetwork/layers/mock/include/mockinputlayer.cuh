// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network input layer, used in tests.
// Created: 01/24/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../../../neuralnetwork/layers/include/layer.cuh"

using namespace std;

class MockInputLayer : public Layer
{
private:
	// Scale to apply to input data.
	float m_dataScale;

public:
	// Constructor.
	MockInputLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, float dataScale = 1.0f);

	// Destructor.
	virtual ~MockInputLayer();

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};