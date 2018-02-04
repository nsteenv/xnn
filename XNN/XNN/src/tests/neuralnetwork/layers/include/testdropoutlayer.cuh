// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for dropout layer.
// Created: 02/16/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <map>

#include "../mock/include/mockdropoutlayer.cuh"
#include "../mock/include/mockinputlayer.cuh"
#include "../mock/include/mockoutputlayer.cuh"
#include "../../../include/abstracttester.cuh"
#include "../../../include/testingutils.cuh"
#include "../../../../neuralnetwork/layers/include/dropoutlayer.cuh"

using namespace std;
using namespace std::chrono;

class TestDropoutLayer : public AbstractTester
{
private:
	// Mapping from test name to test function.
	map<std::string, void(TestDropoutLayer::*)()> m_dropoutLayerTests;

	// Helper functions.
	void TestDoForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, float dropProbability);
	void TestDoBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, float dropProbability);

	// Tests.
	void TestDoForwardProp();
	//void TestForwardPropSpeed();
	void TestDoBackwardProp();

public:
	// Constructor.
	TestDropoutLayer(string outputFolder);

	// Checks if tester has specific test registered.
	virtual bool HasTest(string testName);

	// Runs specific test.
	virtual void RunTest(string testName);

	// Runs all tests.
	virtual void RunAllTests();
};