// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for max pool layer.
// Created: 02/07/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/testmaxpoollayer.cuh"

TestMaxPoolLayer::TestMaxPoolLayer(string outputFolder)
{
	m_outputFolder = outputFolder;

	// Registering tests.
	m_maxPoolLayerTests["doforwardprop"] = &TestMaxPoolLayer::TestDoForwardProp;
	m_maxPoolLayerTests["dobackwardprop"] = &TestMaxPoolLayer::TestDoBackwardProp;
}


bool TestMaxPoolLayer::HasTest(string testName)
{
	auto test = m_maxPoolLayerTests.find(testName);
	return test != m_maxPoolLayerTests.end();
}

void TestMaxPoolLayer::RunTest(string testName)
{
	auto test = m_maxPoolLayerTests.find(testName);
	TestingAssert(test != m_maxPoolLayerTests.end(), "Test not found!");

	((*this).*(test->second))();
}

void TestMaxPoolLayer::RunAllTests()
{
	for (auto test = m_maxPoolLayerTests.begin(); test != m_maxPoolLayerTests.end(); ++test)
	{
		((*this).*(test->second))();
		s_consoleHelper.SetConsoleForeground(ConsoleForeground::GREEN);
		cout << "Test " << test->first << " passed!" << endl << endl;
		s_consoleHelper.RevertConsoleForeground();
	}
}

//******************************************************************************************************
// Helper functions
//******************************************************************************************************

void TestMaxPoolLayer::TestDoForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint unitWidth,
	uint unitHeight, int paddingLeft, int paddingTop, uint unitStride)
{
	// Creating layers.
	MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount);
	MockMaxPoolLayer mockMaxPoolLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, unitWidth, unitHeight, paddingLeft, paddingTop, unitStride);
	mockMaxPoolLayer.AddPrevLayer(&mockInputLayer);
	MaxPoolLayer maxPoolLayer(ParallelismMode::Data, 0, 0, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false, unitWidth, unitHeight,
		paddingLeft, paddingTop, unitStride, false);
	maxPoolLayer.AddPrevLayer(&mockInputLayer);

	// Doing forward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	mockInputLayer.LoadInputs();
	mockInputLayer.DoForwardProp(propagationMode);
	mockMaxPoolLayer.LoadInputs();
	maxPoolLayer.LoadInputs();
	maxPoolLayer.DoForwardProp(propagationMode);
	mockMaxPoolLayer.DoForwardProp(propagationMode);
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	size_t activationsBufferSize = mockMaxPoolLayer.GetActivationBufferSize();
	float* maxPoolLayerActivationBuffer;
	CudaAssert(cudaMallocHost<float>(&maxPoolLayerActivationBuffer, activationsBufferSize));
	CudaAssert(cudaMemcpy(maxPoolLayerActivationBuffer, maxPoolLayer.GetActivationDataBuffer(), activationsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult = true;
	size_t numDifferences = 0;
	float firstDifference = 0.f;
	float firstDifferentMock = 0.f;
	float firstDifferentReg = 0.f;
	bool foundDifferentFromZeroMock = false;
	bool foundDifferentFromZeroReg = false;
	size_t activationsBufferLength = activationsBufferSize / sizeof(float);
	const float* mockMaxPoolLayerActivationBuffer = mockMaxPoolLayer.GetActivationDataBuffer();
	const float maxDiff = 0.0001f;
	const float maxDiffPercentage = 0.001f;
	const float maxDiffPercentageThreshold = 0.000001f;
	CompareBuffers(maxPoolLayerActivationBuffer, mockMaxPoolLayerActivationBuffer, activationsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(maxPoolLayerActivationBuffer));

	TestingAssert(foundDifferentFromZeroMock, "All mock max pool activations are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
		to_string(inputDataCount) + "; Unit width: " + to_string(unitWidth) + "; Padding left: " + to_string(paddingLeft) + "; Unit stride: " + to_string(unitStride));
	TestingAssert(foundDifferentFromZeroReg, "All max pool activations are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
		to_string(inputDataCount) + "; Unit width: " + to_string(unitWidth) + "; Padding left: " + to_string(paddingLeft) + "; Unit stride: " + to_string(unitStride));
	TestingAssert(correctResult, "Incorrect forward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
		"; First different mock activation: " + to_string(firstDifferentMock) + "; First different regular activation: " + to_string(firstDifferentReg) +
		"; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) + "; Unit width: " + to_string(unitWidth) +
		"; Padding left: " + to_string(paddingLeft) + "; Unit stride: " + to_string(unitStride));

	cout << "Forward prop passed. Input num channels: " << inputNumChannels << "; Input data count: " << inputDataCount << "; Unit width: " << unitWidth <<
		"; Padding left: " << paddingLeft << "; Unit stride: " << unitStride << endl;
}

void TestMaxPoolLayer::TestDoBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint unitWidth,
	uint unitHeight, int paddingLeft, int paddingTop, uint unitStride)
{
	// Creating layers.
	MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount);
	MockMaxPoolLayer mockMaxPoolLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, unitWidth, unitHeight, paddingLeft, paddingTop, unitStride);
	mockMaxPoolLayer.AddPrevLayer(&mockInputLayer);
	MaxPoolLayer maxPoolLayer(ParallelismMode::Data, 0, 0, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false, unitWidth, unitHeight,
		paddingLeft, paddingTop, unitStride, false);
	maxPoolLayer.AddPrevLayer(&mockInputLayer);
	MockOutputLayer mockOutputLayer(maxPoolLayer.GetActivationDataSize() * maxPoolLayer.GetActivationNumChannels(), inputDataCount, LossFunctionType::LogisticRegression, false, 0, true);
	mockMaxPoolLayer.AddNextLayer(&mockOutputLayer);
	maxPoolLayer.AddNextLayer(&mockOutputLayer);

	// Doing forward and backward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	mockInputLayer.LoadInputs();
	mockInputLayer.DoForwardProp(propagationMode);
	mockMaxPoolLayer.LoadInputs();
	maxPoolLayer.LoadInputs();
	maxPoolLayer.DoForwardProp(propagationMode);
	mockMaxPoolLayer.DoForwardProp(propagationMode);
	mockOutputLayer.DoBackwardProp();
	maxPoolLayer.LoadActivationGradients();
	maxPoolLayer.DoBackwardProp();
	mockMaxPoolLayer.LoadActivationGradients();
	mockMaxPoolLayer.DoBackwardProp();
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	size_t inputGradientsBufferSize = mockInputLayer.GetActivationBufferSize();
	float* maxPoolLayerInputGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&maxPoolLayerInputGradientsBuffer, inputGradientsBufferSize));
	CudaAssert(cudaMemcpy(maxPoolLayerInputGradientsBuffer, maxPoolLayer.GetInputGradientsBuffer(), inputGradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult = true;
	size_t numDifferences = 0;
	float firstDifference = 0.f;
	float firstDifferentMock = 0.f;
	float firstDifferentReg = 0.f;
	bool foundDifferentFromZeroMock = false;
	bool foundDifferentFromZeroReg = false;
	size_t inputGradientsBufferLength = inputGradientsBufferSize / sizeof(float);
	const float* mockMaxPoolLayerInputGradientsBuffer = mockMaxPoolLayer.GetInputGradientsBuffer();
	const float maxDiff = 0.0001f;
	const float maxDiffPercentage = 0.001f;
	const float maxDiffPercentageThreshold = 0.000001f;
	CompareBuffers(maxPoolLayerInputGradientsBuffer, mockMaxPoolLayerInputGradientsBuffer, inputGradientsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	for (size_t i = 0; i < inputGradientsBufferLength; ++i)
	{
		if (abs(maxPoolLayerInputGradientsBuffer[i] - mockMaxPoolLayerInputGradientsBuffer[i]) > 0.00001f)
		{
			cout << "obican: " << maxPoolLayerInputGradientsBuffer[i] << "    ,     " << "mock: " << mockMaxPoolLayerInputGradientsBuffer[i] << endl;
			cout << i << endl;
		}
	}

	CudaAssert(cudaFreeHost(maxPoolLayerInputGradientsBuffer));

	TestingAssert(foundDifferentFromZeroMock, "All mock max pool input gradients are zeros! Input num channels: " + to_string(inputNumChannels) +
		"; Input data count: " + to_string(inputDataCount) + "; Unit width: " + to_string(unitWidth) + "; Padding left: " + to_string(paddingLeft) +
		"; Unit stride: " + to_string(unitStride));
	TestingAssert(foundDifferentFromZeroReg, "All max pool input gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
		to_string(inputDataCount) + "; Unit width: " + to_string(unitWidth) + "; Padding left: " + to_string(paddingLeft) + "; Unit stride: " + to_string(unitStride));
	TestingAssert(correctResult, "Incorrect backward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
		"; First different mock input gradient: " + to_string(firstDifferentMock) + "; First different regular input gradient: " + to_string(firstDifferentReg) +
		"; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) + "; Unit width: " + to_string(unitWidth) +
		"; Padding left: " + to_string(paddingLeft) + "; Unit stride: " + to_string(unitStride));

	cout << "Backward prop passed. Input num channels: " << inputNumChannels << "; Input data count: " << inputDataCount << "; Unit width: " << unitWidth <<
		"; Padding left: " << paddingLeft << "; Unit stride: " << unitStride << endl;
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

void TestMaxPoolLayer::TestDoForwardProp()
{
	// lastBatch == true

	// m_inputNumChannels % 16 == 0
	TestDoForwardProp(64 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 33 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/);
	TestDoForwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 33 /*inputDataCount*/, 4 /*unitWidth*/, 4 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 2 /*unitStride*/);
	// m_inputNumChannels % 16 != 0
	TestDoForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 57 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/);
	TestDoForwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 17 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 1 /*unitStride*/);

	
	// lastBatch == false

	// m_inputDataCount % 128 == 0

	// m_inputNumChannels % 16 == 0
	TestDoForwardProp(64 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 128 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/);
	TestDoForwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 4 /*unitWidth*/, 4 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 2 /*unitStride*/);
	// m_inputNumChannels % 16 != 0
	TestDoForwardProp(3 /*inputNumChannels*/, 77 /*inputDataWidth*/, 77 /*inputDataHeight*/, 128 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/);
	TestDoForwardProp(3 /*inputNumChannels*/, 77 /*inputDataWidth*/, 77 /*inputDataHeight*/, 128 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 1 /*unitStride*/);

	// m_inputDataCount % 64 == 0

	// m_inputNumChannels % 16 == 0
	TestDoForwardProp(128 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 64 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/);
	TestDoForwardProp(256 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 4 /*unitWidth*/, 4 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 2 /*unitStride*/);
	// m_inputNumChannels % 16 != 0
	TestDoForwardProp(3 /*inputNumChannels*/, 150 /*inputDataWidth*/, 150 /*inputDataHeight*/, 64 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/);
	TestDoForwardProp(3 /*inputNumChannels*/, 150 /*inputDataWidth*/, 150 /*inputDataHeight*/, 64 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 1 /*unitStride*/);

	// m_inputDataCount % 32 == 0

	// m_inputNumChannels % 16 == 0
	TestDoForwardProp(128 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 32 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/);
	TestDoForwardProp(256 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 4 /*unitWidth*/, 4 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 2 /*unitStride*/);
	// m_inputNumChannels % 16 != 0
	TestDoForwardProp(3 /*inputNumChannels*/, 201 /*inputDataWidth*/, 201 /*inputDataHeight*/, 32 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/);
	TestDoForwardProp(3 /*inputNumChannels*/, 201 /*inputDataWidth*/, 201 /*inputDataHeight*/, 32 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 1 /*unitStride*/);
}

void TestMaxPoolLayer::TestDoBackwardProp()
{
	// lastBatch == true

	// m_inputNumChannels % 16 == 0
	TestDoBackwardProp(64 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 33 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/);
	TestDoBackwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 33 /*inputDataCount*/, 4 /*unitWidth*/, 4 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 2 /*unitStride*/);
	// m_inputNumChannels % 16 != 0
	// TODO: Currently unsupported, uncomment here and below if you support this one day.
	//TestDoBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 57 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
	//	0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/);
	//TestDoBackwardProp(3 /*inputNumChannels*/, 224 /*inputDataWidth*/, 224 /*inputDataHeight*/, 17 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
	//	1 /*paddingLeft*/, 1 /*paddingTop*/, 1 /*unitStride*/);


	// lastBatch == false

	// m_inputDataCount % 128 == 0

	// m_inputNumChannels % 16 == 0
	TestDoBackwardProp(64 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 128 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/);
	TestDoBackwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 4 /*unitWidth*/, 4 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 2 /*unitStride*/);
	// m_inputNumChannels % 16 != 0
	//TestDoBackwardProp(3 /*inputNumChannels*/, 77 /*inputDataWidth*/, 77 /*inputDataHeight*/, 128 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
	//	0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/);
	//TestDoBackwardProp(3 /*inputNumChannels*/, 77 /*inputDataWidth*/, 77 /*inputDataHeight*/, 128 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
	//	1 /*paddingLeft*/, 1 /*paddingTop*/, 1 /*unitStride*/);

	// m_inputDataCount % 64 == 0

	// m_inputNumChannels % 16 == 0
	TestDoBackwardProp(128 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 64 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/);
	TestDoBackwardProp(256 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 64 /*inputDataCount*/, 4 /*unitWidth*/, 4 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 2 /*unitStride*/);
	// m_inputNumChannels % 16 != 0
	//TestDoBackwardProp(3 /*inputNumChannels*/, 150 /*inputDataWidth*/, 150 /*inputDataHeight*/, 64 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
	//	0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/);
	//TestDoBackwardProp(3 /*inputNumChannels*/, 150 /*inputDataWidth*/, 150 /*inputDataHeight*/, 64 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
	//	1 /*paddingLeft*/, 1 /*paddingTop*/, 1 /*unitStride*/);

	// m_inputDataCount % 32 == 0

	// m_inputNumChannels % 16 == 0
	TestDoBackwardProp(128 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 32 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
		0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/);
	TestDoBackwardProp(256 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 32 /*inputDataCount*/, 4 /*unitWidth*/, 4 /*unitHeight*/,
		1 /*paddingLeft*/, 1 /*paddingTop*/, 2 /*unitStride*/);
	// m_inputNumChannels % 16 != 0
	//TestDoBackwardProp(3 /*inputNumChannels*/, 201 /*inputDataWidth*/, 201 /*inputDataHeight*/, 32 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
	//	0 /*paddingLeft*/, 0 /*paddingTop*/, 2 /*unitStride*/);
	//TestDoBackwardProp(3 /*inputNumChannels*/, 201 /*inputDataWidth*/, 201 /*inputDataHeight*/, 32 /*inputDataCount*/, 3 /*unitWidth*/, 3 /*unitHeight*/,
	//	1 /*paddingLeft*/, 1 /*paddingTop*/, 1 /*unitStride*/);
}