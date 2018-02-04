// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for standard layer.
// Created: 02/13/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <map>

#include "../mock/include/mockstandardlayer.cuh"
#include "../mock/include/mockinputlayer.cuh"
#include "../mock/include/mockoutputlayer.cuh"
#include "../../../include/abstracttester.cuh"
#include "../../../include/testingutils.cuh"
#include "../../../../neuralnetwork/layers/include/standardlayer.cuh"

using namespace std;
using namespace std::chrono;

class TestStandardLayer : public AbstractTester
{
private:
	// Mapping from test name to test function.
	map<std::string, void(TestStandardLayer::*)()> m_standardLayerTests;

	// Helper functions.
	void TestDoForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, cublasHandle_t cublasHandle,
		uint numNeurons);
	void TestDoBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, cublasHandle_t cublasHandle,
		uint numNeurons);

	// Tests.
	void TestDoForwardProp();
	void TestForwardPropSpeed();
	void TestDoBackwardProp();

public:
	// Constructor.
	TestStandardLayer(string outputFolder);

	// Checks if tester has specific test registered.
	virtual bool HasTest(string testName);

	// Runs specific test.
	virtual void RunTest(string testName);

	// Runs all tests.
	virtual void RunAllTests();
};