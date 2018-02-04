// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network max pool layer.
// Created: 02/06/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "layer.cuh"

using namespace std;

/*
	Max pool layer partitions the input image into a set of regions, and for each such region outputs the maximum value of input activity.
	It helps to reduce dimensionality, but also learns model to be more invariant to translation.
*/
class MaxPoolLayer : public Layer
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

	// Reinitializes layer when input data count changes.
	virtual void Reinitialize(uint newInputDataCount);

public:
	// Constructor.
	MaxPoolLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, uint indexInTier, uint tierSize, uint inputNumChannels,
		uint inputDataWidth, uint inputDataHeight, uint inputDataCount, bool holdsInputData, uint unitWidth, uint unitHeight, int paddingX, int paddingY,
		uint unitStride, bool holdsActivationGradients);

	// Destructor.
	virtual ~MaxPoolLayer() {}

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};