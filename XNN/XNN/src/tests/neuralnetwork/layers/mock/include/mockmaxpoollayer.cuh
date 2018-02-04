// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Mocked neural network max pool layer, used in tests.
// Created: 02/07/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../../../include/testingutils.cuh"
#include "../../../../../neuralnetwork/layers/include/layer.cuh"

using namespace std;

class MockMaxPoolLayer : public Layer
{
private:
	// Width of the pooling unit.
	uint m_unitWidth;

	// Height of the pooling unit.
	uint m_unitHeight;

	// Padding in dimension X.
	int m_paddingX;

	// Padding in dimension Y.
	int m_paddingY;

	// Stride of the pooling unit.
	uint m_unitStride;

	// Number of pooling units in dimension X.
	uint m_numUnitsX;

	// Number of pooling units in dimension Y.
	uint m_numUnitsY;

public:
	// Constructor.
	MockMaxPoolLayer(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint unitWidth,
		uint unitHeight, int paddingX, int paddingY, uint unitStride);

	// Destructor.
	virtual ~MockMaxPoolLayer();

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Loads activation gradients to layer.
	void LoadActivationGradients();

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};