// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Driver for all tests.
// Created: 12/12/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "../dataparsers/image/include/testjpegdataparser.cuh"
#include "../neuralnetwork/layers/include/testconvolutionallayer.cuh"
#include "../neuralnetwork/layers/include/testdropoutlayer.cuh"
#include "../neuralnetwork/layers/include/testmaxpoollayer.cuh"
#include "../neuralnetwork/layers/include/testoutputlayer.cuh"
#include "../neuralnetwork/layers/include/testresponsenormalizationlayer.cuh"
#include "../neuralnetwork/layers/include/testsoftmaxlayer.cuh"
#include "../neuralnetwork/layers/include/teststandardlayer.cuh"

using namespace std;

class TestsDriver
{
private:
	// Output folder for tests.
	string m_outputFolder;

	// Component to test.
	string m_componentToRun;

	// Specific test name to test.
	string m_testToRun;

	// Map of all testers by component name.
	map<string, AbstractTester*> m_testers;

	// Registers testers.
	void RegisterTesters();
public:
	static const string c_outputFolderSignature;
	static const string c_componentToRunSignature;
	static const string c_testToRunSignature;

	// Desctructor,
	~TestsDriver();

	// Parses arguments for testing.
	bool ParseArguments(int argc, char *argv[]);

	// Runs tests.
	void RunTests();
};