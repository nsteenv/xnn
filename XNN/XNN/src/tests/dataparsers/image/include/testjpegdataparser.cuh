// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for jpeg data parser.
// Created: 12/12/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <map>
#include <vector>

#include "../../../include/abstracttester.cuh"
#include "../../../include/testingutils.cuh"
#include "../../../../dataparsers/image/include/jpegdataparser.cuh"

using namespace std;

class TestJpegDataParser : public AbstractTester
{
private:
	// Mapping from test name to test function.
	map<std::string, void(TestJpegDataParser::*)()> m_jpegDataParserTests;

	// Tests.
	void TestResizeImageCu();
	void TestCropImage();
	void TestResizeImageWithCropCu();

public:
	// Constructor.
	TestJpegDataParser(string outputFolder);

	// Checks if tester has specific test registered.
	virtual bool HasTest(string testName);

	// Runs specific test.
	virtual void RunTest(string testName);

	// Runs all tests.
	virtual void RunAllTests();
};