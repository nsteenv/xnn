// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for response normalization layer.
// Created: 02/11/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <map>

#include "../mock/include/mockinputlayer.cuh"
#include "../mock/include/mockresponsenormalizationlayer.cuh"
#include "../mock/include/mockoutputlayer.cuh"
#include "../../../include/abstracttester.cuh"
#include "../../../include/testingutils.cuh"
#include "../../../../neuralnetwork/layers/include/responsenormalizationlayer.cuh"

using namespace std;

class TestResponseNormalizationLayer : public AbstractTester
{
private:
	// Mapping from test name to test function.
	map<std::string, void(TestResponseNormalizationLayer::*)()> m_responseNormalizationLayerTests;

	// Helper functions.
	void TestDoForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint depth, float bias,
		float alphaCoeff, float betaCoeff);
	void TestDoBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint depth, float bias,
		float alphaCoeff, float betaCoeff);

	// Tests.
	void TestDoForwardProp();
	void TestDoBackwardProp();

public:
	// Constructor.
	TestResponseNormalizationLayer(string outputFolder);

	// Checks if tester has specific test registered.
	virtual bool HasTest(string testName);

	// Runs specific test.
	virtual void RunTest(string testName);

	// Runs all tests.
	virtual void RunAllTests();
};