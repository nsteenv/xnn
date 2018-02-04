// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network activation functions.
// Created: 12/27/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../utils/include/utils.cuh"

using namespace std;

// Activation types.
enum class ActivationType
{
	Linear,
	ReLu,
	Sigmoid,
	Tanh
};

// Applies activation to preactivations.
void ApplyActivation(ActivationType activationType, float* preactivations, uint numPreactivations, float* activations, cudaStream_t deviceCalculationStream);

// Calculates gradients of activations to preactivations.
void CalculatePreactivationGradients(ActivationType activationType, float* activationGradients, float* activations, uint numActivations, float* preactivationGradients,
	cudaStream_t deviceCalculationStream);