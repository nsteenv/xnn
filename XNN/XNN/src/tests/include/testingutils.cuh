// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Utility functions for testing.
// Created: 12/12/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../../utils/include/utils.cuh"

// Macro to check testing asserts.
#define TestingAssert(condition, message) _TestingAssert(condition, message, __FILE__, __FUNCTION__, __LINE__)

using namespace std;

// Tests resources folder.
const string c_testResourcesFolder = "C:\\Users\\marko\\Documents\\Visual Studio 2013\\Projects\\AI\\XNN\\XNN\\resources\\";

// Asserts test operation if condition is not met.
// Should never be called directly, use macro!
void _TestingAssert(bool condition, string message, const char* file, const char* function, int line);

// Compares two buffers with results and returns whether we got correct result, or if we didn't returns first difference etc.
void CompareBuffers(const float* regularBuffer, const float* mockBuffer, size_t buffersLength, float maxDiff, float maxDiffPercentage, float maxDiffPercentageThreshold,
	bool& correctResult, size_t& numDifferences, float& firstDifference, float& firstDifferentMock, float& firstDifferentReg, bool& foundDifferentFromZeroMock,
	bool& foundDifferentFromZeroReg);