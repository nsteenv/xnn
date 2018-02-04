// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for max pool layer.
// Created: 02/07/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <map>

#include "../mock/include/mockinputlayer.cuh"
#include "../mock/include/mockmaxpoollayer.cuh"
#include "../mock/include/mockoutputlayer.cuh"
#include "../../../include/abstracttester.cuh"
#include "../../../include/testingutils.cuh"
#include "../../../../neuralnetwork/layers/include/maxpoollayer.cuh"

using namespace std;

class TestMaxPoolLayer : public AbstractTester
{
private:
	// Mapping from test name to test function.
	map<std::string, void(TestMaxPoolLayer::*)()> m_maxPoolLayerTests;

	// Helper functions.
	void TestDoForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint unitWidth,
		uint unitHeight, int paddingLeft, int paddingTop, uint unitStride);
	void TestDoBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint unitWidth,
		uint unitHeight, int paddingLeft, int paddingTop, uint unitStride);

	// Tests.
	void TestDoForwardProp();
	void TestDoBackwardProp();

public:
	// Constructor.
	TestMaxPoolLayer(string outputFolder);

	// Checks if tester has specific test registered.
	virtual bool HasTest(string testName);

	// Runs specific test.
	virtual void RunTest(string testName);

	// Runs all tests.
	virtual void RunAllTests();
};