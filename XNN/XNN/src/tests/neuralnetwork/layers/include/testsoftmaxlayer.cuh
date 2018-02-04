// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for softmax layer.
// Created: 02/20/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <map>

#include "../mock/include/mockinputlayer.cuh"
#include "../mock/include/mocksoftmaxlayer.cuh"
#include "../../../include/abstracttester.cuh"
#include "../../../include/testingutils.cuh"
#include "../../../../neuralnetwork/layers/include/outputlayer.cuh"
#include "../../../../neuralnetwork/layers/include/softmaxlayer.cuh"

using namespace std;
using namespace std::chrono;

class TestSoftMaxLayer : public AbstractTester
{
private:
	// Mapping from test name to test function.
	map<std::string, void(TestSoftMaxLayer::*)()> m_softMaxLayerTests;

	// Helper functions.
	void TestDoForwardProp(uint inputDataSize, uint inputDataCount);
	void TestDoBackwardProp(uint inputDataSize, uint inputDataCount);

	// Tests.
	void TestDoForwardProp();
	//void TestForwardPropSpeed();
	void TestDoBackwardProp();

public:
	// Constructor.
	TestSoftMaxLayer(string outputFolder);

	// Checks if tester has specific test registered.
	virtual bool HasTest(string testName);

	// Runs specific test.
	virtual void RunTest(string testName);

	// Runs all tests.
	virtual void RunAllTests();
};