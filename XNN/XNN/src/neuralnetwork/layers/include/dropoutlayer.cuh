// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network dropout layer.
// Created: 02/16/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include "layer.cuh"

using namespace std;

/*
	Dropout layer provides efficient way to simulate combining multiple trained models to reduce test error and prevent overfitting.
	It works by dropping each neuron activity with certain probability, preventing complex coadaptations between neurons.
*/
class DropoutLayer : public Layer
{
private:
	// Dropout filter.
	float* m_dropoutFilter;

	// Biases buffer size.
	size_t m_dropoutFilterSize;

	// Buffer for cuRAND states.
	curandState* m_curandStatesBuffer;

	// Probability for dropping each activity.
	float m_dropProbability;

	// Should we use dropout filter from host.
	bool m_useHostDropoutFilter;

	// Creates dropout filter.
	void CreateDropoutFilter();

	// Applies dropout filter.
	void ApplyDropoutFilter();

	// Reinitializes layer when input data count changes.
	virtual void Reinitialize(uint newInputDataCount);

public:
	// Number of blocks to use for cuRAND operations.
	static const uint c_numCurandBlocks;

	// Number of threads to use for cuRAND operations.
	static const uint c_numCurandThreadsPerBlock;

	// Constructor.
	DropoutLayer(ParallelismMode parallelismMode, cudaStream_t deviceCalculationStream, cudaStream_t deviceMemoryStream, curandState* curandStatesBuffer,
		uint indexInTier, uint tierSize, uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, bool holdsInputData, float dropProbability,
		bool useHostDropoutFilter, bool holdsActivationGradients);

	// Copies dropout filter from host.
	void CopyDropoutFilterFromHost(float* hostDropoutFilter);

	// Destructor.
	virtual ~DropoutLayer();

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};