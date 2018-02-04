// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Abstract tester interface.
// Created: 12/12/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <string>

using namespace std;

class AbstractTester
{
protected:
	// Output folder for tests.
	string m_outputFolder;

public:
	// Checks if tester has specific test registered.
	virtual bool HasTest(string testName) = 0;

	// Runs specific test.
	virtual void RunTest(string testName) = 0;

	// Runs all tests.
	virtual void RunAllTests() = 0;
};