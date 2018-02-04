// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for softmax layer.
// Created: 02/20/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/testsoftmaxlayer.cuh"

TestSoftMaxLayer::TestSoftMaxLayer(string outputFolder)
{
	m_outputFolder = outputFolder;

	// Registering tests.
	m_softMaxLayerTests["doforwardprop"] = &TestSoftMaxLayer::TestDoForwardProp;
	//m_softMaxLayerTests["forwardpropspeed"] = &TestSoftMaxLayer::TestForwardPropSpeed;
	m_softMaxLayerTests["dobackwardprop"] = &TestSoftMaxLayer::TestDoBackwardProp;
}


bool TestSoftMaxLayer::HasTest(string testName)
{
	auto test = m_softMaxLayerTests.find(testName);
	return test != m_softMaxLayerTests.end();
}

void TestSoftMaxLayer::RunTest(string testName)
{
	auto test = m_softMaxLayerTests.find(testName);
	TestingAssert(test != m_softMaxLayerTests.end(), "Test not found!");

	((*this).*(test->second))();
}

void TestSoftMaxLayer::RunAllTests()
{
	for (auto test = m_softMaxLayerTests.begin(); test != m_softMaxLayerTests.end(); ++test)
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

void TestSoftMaxLayer::TestDoForwardProp(uint inputDataSize, uint inputDataCount)
{
	// Creating layers.
	MockInputLayer inputLayer(1, inputDataSize, 1, inputDataCount);
	MockSoftMaxLayer mockSoftMaxLayer(inputDataSize, inputDataCount);
	mockSoftMaxLayer.AddPrevLayer(&inputLayer);
	SoftMaxLayer softMaxLayer(ParallelismMode::Model, 0, 0, inputDataSize, inputDataCount, false);
	softMaxLayer.AddPrevLayer(&inputLayer);

	// Doing forward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	inputLayer.LoadInputs();
	inputLayer.DoForwardProp(propagationMode);
	mockSoftMaxLayer.LoadInputs();
	softMaxLayer.LoadInputs();
	softMaxLayer.DoForwardProp(propagationMode);
	mockSoftMaxLayer.DoForwardProp(propagationMode);
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	size_t activationsBufferSize = mockSoftMaxLayer.GetActivationBufferSize();
	float* softMaxLayerActivationBuffer;
	CudaAssert(cudaMallocHost<float>(&softMaxLayerActivationBuffer, activationsBufferSize));
	CudaAssert(cudaMemcpy(softMaxLayerActivationBuffer, softMaxLayer.GetActivationDataBuffer(), activationsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult = true;
	size_t numDifferences = 0;
	float firstDifference = 0.f;
	float firstDifferentMock = 0.f;
	float firstDifferentReg = 0.f;
	bool foundDifferentFromZeroMock = false;
	bool foundDifferentFromZeroReg = false;
	size_t activationsBufferLength = activationsBufferSize / sizeof(float);
	const float* mockSoftMaxLayerActivationBuffer = mockSoftMaxLayer.GetActivationDataBuffer();
	const float maxDiff = 0.000001f;
	const float maxDiffPercentage = 0.001f;
	const float maxDiffPercentageThreshold = 0.0000001f;
	CompareBuffers(softMaxLayerActivationBuffer, mockSoftMaxLayerActivationBuffer, activationsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	bool foundIrregularMock = false;
	float irregularMock = 0.0f;
	bool foundIrregularReg = false;
	float irregularReg = 0.0f;
	for (size_t i = 0; i < activationsBufferLength; ++i)
	{
		if (mockSoftMaxLayerActivationBuffer[i] < 0.0f || mockSoftMaxLayerActivationBuffer[i] > 1.0f)
		{
			foundIrregularMock = true;
			irregularMock = mockSoftMaxLayerActivationBuffer[i];
		}
		if (softMaxLayerActivationBuffer[i] < 0.0f || softMaxLayerActivationBuffer[i] > 1.0f)
		{
			foundIrregularReg = true;
			irregularReg = softMaxLayerActivationBuffer[i];
		}
	}


	CudaAssert(cudaFreeHost(softMaxLayerActivationBuffer));

	TestingAssert(foundDifferentFromZeroMock, "All mock softmax layer activations are zeros! Input data size: " + to_string(inputDataSize) +
		"; Input data count: " + to_string(inputDataCount));
	TestingAssert(foundDifferentFromZeroReg, "All softmax layer activations are zeros! Input data size: " + to_string(inputDataSize) +
		"; Input data count: " + to_string(inputDataCount));
	TestingAssert(!foundIrregularMock, "Found irregular mock softmax layer activation! Input data size: " + to_string(inputDataSize) +
		"; Input data count: " + to_string(inputDataCount) + "; Irregular value: " + to_string(irregularMock));
	TestingAssert(!foundIrregularReg, "Found irregular softmax layer activation! Input data size: " + to_string(inputDataSize) +
		"; Input data count: " + to_string(inputDataCount) + "; Irregular value: " + to_string(irregularReg));
	TestingAssert(correctResult, "Incorrect forward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
		"; First different mock activation: " + to_string(firstDifferentMock) + "; First different regular activation: " + to_string(firstDifferentReg) +
		"; Input data size: " + to_string(inputDataSize) + "; Input data count: " + to_string(inputDataCount));
	

	cout << "Forward prop passed. Input data size: " << inputDataSize << "; Input data count: " << inputDataCount << endl;
}

void TestSoftMaxLayer::TestDoBackwardProp(uint inputDataSize, uint inputDataCount)
{
	// Creating layers.
	MockInputLayer inputLayer(1, inputDataSize, 1, inputDataCount);
	MockSoftMaxLayer mockSoftMaxLayer(inputDataSize, inputDataCount);
	mockSoftMaxLayer.AddPrevLayer(&inputLayer);
	SoftMaxLayer softMaxLayer(ParallelismMode::Model, 0, 0, inputDataSize, inputDataCount, false);
	softMaxLayer.AddPrevLayer(&inputLayer);
	OutputLayer outputLayer(0, 0, inputDataSize, inputDataCount, inputDataCount, LossFunctionType::LogisticRegression, true, 5, 0);
	mockSoftMaxLayer.AddNextLayer(&outputLayer);
	softMaxLayer.AddNextLayer(&outputLayer);

	// Creating random labels.
	vector<uint> labels;
	for (uint i = 0; i < inputDataCount; ++i)
	{
		labels.push_back((57 * i * i) % inputDataSize);
	}
	outputLayer.LoadDataLabels(labels);

	// Doing forward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	inputLayer.LoadInputs();
	inputLayer.DoForwardProp(propagationMode);
	mockSoftMaxLayer.LoadInputs();
	softMaxLayer.LoadInputs();
	softMaxLayer.DoForwardProp(propagationMode);
	mockSoftMaxLayer.DoForwardProp(propagationMode);
	CudaAssert(cudaDeviceSynchronize());

	// Doing backward prop.
	softMaxLayer.DoBackwardProp();
	mockSoftMaxLayer.DoBackwardProp();
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	size_t gradientsBufferSize = inputLayer.GetActivationBufferSize();
	float* softMaxLayerInputGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&softMaxLayerInputGradientsBuffer, gradientsBufferSize));
	CudaAssert(cudaMemcpy(softMaxLayerInputGradientsBuffer, softMaxLayer.GetInputGradientsBuffer(), gradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult = true;
	size_t numDifferences = 0;
	float firstDifference = 0.f;
	float firstDifferentMock = 0.f;
	float firstDifferentReg = 0.f;
	bool foundDifferentFromZeroMock = false;
	bool foundDifferentFromZeroReg = false;
	size_t gradientsBufferLength = gradientsBufferSize / sizeof(float);
	const float* mockSoftMaxLayerInputGradientsBuffer = mockSoftMaxLayer.GetInputGradientsBuffer();
	const float maxDiff = 0.000001f;
	const float maxDiffPercentage = 0.001f;
	const float maxDiffPercentageThreshold = 0.000001f;
	CompareBuffers(softMaxLayerInputGradientsBuffer, mockSoftMaxLayerInputGradientsBuffer, gradientsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(softMaxLayerInputGradientsBuffer));

	TestingAssert(foundDifferentFromZeroMock, "All mock softmax layer input gradients are zeros! Input data size: " + to_string(inputDataSize) +
		"; Input data count: " + to_string(inputDataCount));
	TestingAssert(foundDifferentFromZeroReg, "All softmax layer input gradients are zeros! Input data size: " + to_string(inputDataSize) +
		"; Input data count: " + to_string(inputDataCount));
	TestingAssert(correctResult, "Incorrect backward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
		"; First different mock input gradient: " + to_string(firstDifferentMock) + "; First different regular input gradient: " + to_string(firstDifferentReg) +
		"; Input data size: " + to_string(inputDataSize) + "; Input data count: " + to_string(inputDataCount));


	cout << "Backward prop passed. Input data size: " << inputDataSize << "; Input data count: " << inputDataCount << endl;
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

void TestSoftMaxLayer::TestDoForwardProp()
{
	// lastBatch == true
	TestDoForwardProp(2 /*inputDataSize*/, 119 /*inputDataCount*/);
	TestDoForwardProp(3 /*inputDataSize*/, 1 /*inputDataCount*/);
	TestDoForwardProp(5 /*inputDataSize*/, 17 /*inputDataCount*/);
	TestDoForwardProp(100 /*inputDataSize*/, 99 /*inputDataCount*/);
	TestDoForwardProp(1000 /*inputDataSize*/, 57 /*inputDataCount*/);
	TestDoForwardProp(10000 /*inputDataSize*/, 31 /*inputDataCount*/);

	// lastBatch == false
	TestDoForwardProp(2 /*inputDataSize*/, 512 /*inputDataCount*/);
	TestDoForwardProp(3 /*inputDataSize*/, 512 /*inputDataCount*/);
	TestDoForwardProp(5 /*inputDataSize*/, 512 /*inputDataCount*/);
	TestDoForwardProp(100 /*inputDataSize*/, 512 /*inputDataCount*/);
	TestDoForwardProp(1000 /*inputDataSize*/, 128 /*inputDataCount*/);
	TestDoForwardProp(10000 /*inputDataSize*/, 128 /*inputDataCount*/);
}

void TestSoftMaxLayer::TestDoBackwardProp()
{
	// lastBatch == true
	TestDoBackwardProp(2 /*inputDataSize*/, 119 /*inputDataCount*/);
	TestDoBackwardProp(3 /*inputDataSize*/, 1 /*inputDataCount*/);
	TestDoBackwardProp(5 /*inputDataSize*/, 17 /*inputDataCount*/);
	TestDoBackwardProp(100 /*inputDataSize*/, 99 /*inputDataCount*/);
	TestDoBackwardProp(1000 /*inputDataSize*/, 57 /*inputDataCount*/);
	TestDoBackwardProp(10000 /*inputDataSize*/, 31 /*inputDataCount*/);

	// lastBatch == false
	TestDoBackwardProp(2 /*inputDataSize*/, 512 /*inputDataCount*/);
	TestDoBackwardProp(3 /*inputDataSize*/, 512 /*inputDataCount*/);
	TestDoBackwardProp(5 /*inputDataSize*/, 512 /*inputDataCount*/);
	TestDoBackwardProp(100 /*inputDataSize*/, 512 /*inputDataCount*/);
	TestDoBackwardProp(1000 /*inputDataSize*/, 128 /*inputDataCount*/);
	TestDoBackwardProp(10000 /*inputDataSize*/, 128 /*inputDataCount*/);
}