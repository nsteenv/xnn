// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network response normalization layer, used in tests.
// Created: 02/09/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../../include/testingutils.cuh"
#include "../../../../../neuralnetwork/layers/include/layer.cuh"

using namespace std;

class MockResponseNormalizationLayer : public Layer
{
private:
	// Depth of normalization.
	uint m_depth;

	// Normalization bias.
	float m_bias;

	// Normalization alpha coefficient (see the formula).
	float m_alphaCoeff;

	// Normalization beta coefficient (see the formula).
	float m_betaCoeff;

public:
	// Constructor.
	MockResponseNormalizationLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint depth, float bias,
		float alphaCoeff, float betaCoeff);

	// Destructor.
	virtual ~MockResponseNormalizationLayer();

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Loads activation gradients to layer.
	void LoadActivationGradients();

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};