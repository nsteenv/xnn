// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for output layer.
// Created: 02/21/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <map>

#include "../mock/include/mockinputlayer.cuh"
#include "../mock/include/mockoutputlayer.cuh"
#include "../mock/include/mockstandardlayer.cuh"
#include "../../../include/abstracttester.cuh"
#include "../../../include/testingutils.cuh"
#include "../../../../neuralnetwork/layers/include/outputlayer.cuh"
#include "../../../../neuralnetwork/layers/include/softmaxlayer.cuh"

using namespace std;
using namespace std::chrono;

class TestOutputLayer : public AbstractTester
{
private:
	// Mapping from test name to test function.
	map<std::string, void(TestOutputLayer::*)()> m_outputLayerTests;

	// Helper functions.
	void TestDoForwardProp(uint inputDataSize, uint inputDataCount);

	// Tests.
	void TestDoForwardProp();
	//void TestForwardPropSpeed();

public:
	// Constructor.
	TestOutputLayer(string outputFolder);

	// Checks if tester has specific test registered.
	virtual bool HasTest(string testName);

	// Runs specific test.
	virtual void RunTest(string testName);

	// Runs all tests.
	virtual void RunAllTests();
};