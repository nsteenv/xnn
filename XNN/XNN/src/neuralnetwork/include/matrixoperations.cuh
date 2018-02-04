// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network matrix operations functions.
// Created: 03/06/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../utils/include/utils.cuh"

using namespace std;

// Calculates element-wise product of two input matrices, and stores result into output matrix.
void CalculateElementWiseProduct(float* inputMatrixA, float* inputMatrixB, uint numElements, float* outputMatrix, cudaStream_t deviceCalculationStream);

// Calculates element-wise sum of two input matrices, and stores result into output matrix.
void CalculateElementWiseSum(float* inputMatrixA, float* inputMatrixB, uint numElements, float* outputMatrix, cudaStream_t deviceCalculationStream);

// Calculates element-wise scaled matrix of input matrix, and stores result into output matrix.
void CalculateElementWiseScale(float* inputMatrix, float scale, uint numElements, float* outputMatrix, cudaStream_t deviceCalculationStream);