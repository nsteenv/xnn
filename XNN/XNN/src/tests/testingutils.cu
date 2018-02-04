// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Utility functions for testing.
// Created: 12/12/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/testingutils.cuh"

void _TestingAssert(bool condition, string message, const char* file, const char* function, int line)
{
	if (!condition)
	{
		{
			lock_guard<mutex> lock(s_consoleMutex);
			s_consoleHelper.SetConsoleForeground(ConsoleForeground::RED);
			cout << endl << "Error in runing tests: " << message << endl << endl;
			cout << "Operation info: " << function << " (\"" << GetFileName(file) << "\" [line: " << line << "])" << endl << endl;
		}
		exit(EXIT_FAILURE);
	}
}

void CompareBuffers(const float* regularBuffer, const float* mockBuffer, size_t buffersLength, float maxDiff, float maxDiffPercentage, float maxDiffPercentageThreshold,
	bool& correctResult, size_t& numDifferences, float& firstDifference, float& firstDifferentMock, float& firstDifferentReg, bool& foundDifferentFromZeroMock,
	bool& foundDifferentFromZeroReg)
{
	numDifferences = 0;
	for (size_t i = 0; i < buffersLength; ++i)
	{
		float diff = fabs(mockBuffer[i] - regularBuffer[i]);
		if (diff > maxDiff || (diff > maxDiffPercentageThreshold && diff > maxDiffPercentage * max(abs(mockBuffer[i]), abs(regularBuffer[i]))))
		{
			++numDifferences;
			if (correctResult)
			{
				correctResult = false;
				firstDifference = mockBuffer[i] - regularBuffer[i];
				firstDifferentMock = mockBuffer[i];
				firstDifferentReg = regularBuffer[i];
			}
		}
		if (mockBuffer[i] != 0.0f && mockBuffer[i] != FLT_MIN)
		{
			foundDifferentFromZeroMock = true;
		}
		if (regularBuffer[i] != 0.0f && regularBuffer[i] != FLT_MIN)
		{
			foundDifferentFromZeroReg = true;
		}
	}
}