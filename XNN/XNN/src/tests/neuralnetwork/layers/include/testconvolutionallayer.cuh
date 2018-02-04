// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for convolutional layer.
// Created: 01/24/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <map>

#include "../mock/include/mockconvolutionallayer.cuh"
#include "../mock/include/mockinputlayer.cuh"
#include "../mock/include/mockoutputlayer.cuh"
#include "../../../include/abstracttester.cuh"
#include "../../../include/testingutils.cuh"
#include "../../../../neuralnetwork/layers/include/convolutionallayer.cuh"

using namespace std;

class TestConvolutionalLayer : public AbstractTester
{
private:
	// Mapping from test name to test function.
	map<std::string, void(TestConvolutionalLayer::*)()> m_convolutionalLayerTests;

	// Helper functions.
	void TestDoForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numFilters, uint filterWidth,
		uint filterHeight, int paddingX, int paddingY, uint stride);
	void TestDoBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint numFilters, uint filterWidth,
		uint filterHeight, int paddingX, int paddingY, uint stride);

	// Prints out computation info for debug purposes.
	void PrintComputationInfo(size_t activationDifferentPixelIndex, uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount,
		uint numFilters, uint filterWidth, uint filterHeight, int paddingX, int paddingY, uint stride, float* inputDataBuffer, float* filtersBuffer,
		float differentActivationPixelMock, float differentActivationPixelRegular);

	// Tests.
	void TestDoForwardProp();
	void TestDoBackwardProp();

public:
	// Constructor.
	TestConvolutionalLayer(string outputFolder);

	// Checks if tester has specific test registered.
	virtual bool HasTest(string testName);

	// Runs specific test.
	virtual void RunTest(string testName);

	// Runs all tests.
	virtual void RunAllTests();
};