// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for dropout layer.
// Created: 02/16/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/testdropoutlayer.cuh"

TestDropoutLayer::TestDropoutLayer(string outputFolder)
{
	m_outputFolder = outputFolder;

	// Registering tests.
	m_dropoutLayerTests["doforwardprop"] = &TestDropoutLayer::TestDoForwardProp;
	//m_dropoutLayerTests["forwardpropspeed"] = &TestDropoutLayer::TestForwardPropSpeed;
	m_dropoutLayerTests["dobackwardprop"] = &TestDropoutLayer::TestDoBackwardProp;
}


bool TestDropoutLayer::HasTest(string testName)
{
	auto test = m_dropoutLayerTests.find(testName);
	return test != m_dropoutLayerTests.end();
}

void TestDropoutLayer::RunTest(string testName)
{
	auto test = m_dropoutLayerTests.find(testName);
	TestingAssert(test != m_dropoutLayerTests.end(), "Test not found!");

	((*this).*(test->second))();
}

void TestDropoutLayer::RunAllTests()
{
	for (auto test = m_dropoutLayerTests.begin(); test != m_dropoutLayerTests.end(); ++test)
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

void TestDropoutLayer::TestDoForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, float dropProbability)
{
	// Creating layers.
	MockInputLayer inputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount);
	MockDropoutLayer mockDropoutLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, dropProbability);
	mockDropoutLayer.AddPrevLayer(&inputLayer);
	DropoutLayer dropoutLayer(ParallelismMode::Data, 0, 0, NULL, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false,
		dropProbability, true, false);
	dropoutLayer.AddPrevLayer(&inputLayer);

	// Doing forward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	inputLayer.LoadInputs();
	inputLayer.DoForwardProp(propagationMode);
	mockDropoutLayer.LoadInputs();
	mockDropoutLayer.DoForwardProp(propagationMode);
	dropoutLayer.CopyDropoutFilterFromHost(mockDropoutLayer.GetDropoutFilter());
	dropoutLayer.LoadInputs();
	dropoutLayer.DoForwardProp(propagationMode);
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	size_t activationsBufferSize = mockDropoutLayer.GetActivationBufferSize();
	float* dropoutLayerActivationBuffer;
	CudaAssert(cudaMallocHost<float>(&dropoutLayerActivationBuffer, activationsBufferSize));
	CudaAssert(cudaMemcpy(dropoutLayerActivationBuffer, dropoutLayer.GetActivationDataBuffer(), activationsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult = true;
	size_t numDifferences = 0;
	float firstDifference = 0.f;
	float firstDifferentMock = 0.f;
	float firstDifferentReg = 0.f;
	bool foundDifferentFromZeroMock = false;
	bool foundDifferentFromZeroReg = false;
	size_t activationsBufferLength = activationsBufferSize / sizeof(float);
	const float* mockDropoutLayerActivationBuffer = mockDropoutLayer.GetActivationDataBuffer();
	const float maxDiff = 0.000001f;
	const float maxDiffPercentage = 0.001f;
	const float maxDiffPercentageThreshold = 0.00000001f;
	CompareBuffers(dropoutLayerActivationBuffer, mockDropoutLayerActivationBuffer, activationsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(dropoutLayerActivationBuffer));

	TestingAssert(foundDifferentFromZeroMock, "All mock dropout layer activations are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data width: " +
		to_string(inputDataWidth) + "; Input data height: " + to_string(inputDataHeight) + "; Input data count: " + to_string(inputDataCount) + "; Drop probability: " +
		to_string(dropProbability));
	TestingAssert(foundDifferentFromZeroReg, "All dropout layer activations are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data width: " +
		to_string(inputDataWidth) + "; Input data height: " + to_string(inputDataHeight) + "; Input data count: " + to_string(inputDataCount) + "; Drop probability: " +
		to_string(dropProbability));
	TestingAssert(correctResult, "Incorrect forward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
		"; First different mock activation: " + to_string(firstDifferentMock) + "; First different regular activation: " + to_string(firstDifferentReg) +
		"; Input num channels: " + to_string(inputNumChannels) + "; Input data width: " + to_string(inputDataWidth) + "; Input data height: " +
		to_string(inputDataHeight) + "; Input data count: " + to_string(inputDataCount) + "; Drop probability: " + to_string(dropProbability));

	cout << "Forward prop passed. Input num channels: " << inputNumChannels << "; Input width: " << inputDataWidth << "; Input height: " << inputDataHeight <<
		"; Input data count: " << inputDataCount << "; Drop probability: " << dropProbability << endl;
}

void TestDropoutLayer::TestDoBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, float dropProbability)
{
	// Creating layers.
	MockInputLayer inputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount);
	MockDropoutLayer mockDropoutLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, dropProbability);
	mockDropoutLayer.AddPrevLayer(&inputLayer);
	DropoutLayer dropoutLayer(ParallelismMode::Data, 0, 0, NULL, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false,
		dropProbability, true, false);
	dropoutLayer.AddPrevLayer(&inputLayer);
	MockOutputLayer outputLayer(inputNumChannels * inputDataWidth * inputDataHeight, inputDataCount, LossFunctionType::LogisticRegression, false, 0, true);
	mockDropoutLayer.AddNextLayer(&outputLayer);
	dropoutLayer.AddNextLayer(&outputLayer);

	// Doing forward and backward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	inputLayer.LoadInputs();
	inputLayer.DoForwardProp(propagationMode);
	mockDropoutLayer.LoadInputs();
	mockDropoutLayer.DoForwardProp(propagationMode);
	dropoutLayer.CopyDropoutFilterFromHost(mockDropoutLayer.GetDropoutFilter());
	dropoutLayer.LoadInputs();
	dropoutLayer.DoForwardProp(propagationMode);
	outputLayer.DoBackwardProp();
	dropoutLayer.LoadActivationGradients();
	dropoutLayer.DoBackwardProp();
	mockDropoutLayer.LoadActivationGradients();
	mockDropoutLayer.DoBackwardProp();
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	size_t gradientsBufferSize = inputLayer.GetActivationBufferSize();
	float* dropoutLayerInputGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&dropoutLayerInputGradientsBuffer, gradientsBufferSize));
	CudaAssert(cudaMemcpy(dropoutLayerInputGradientsBuffer, dropoutLayer.GetInputGradientsBuffer(), gradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult = true;
	size_t numDifferences = 0;
	float firstDifference = 0.f;
	float firstDifferentMock = 0.f;
	float firstDifferentReg = 0.f;
	bool foundDifferentFromZeroMock = false;
	bool foundDifferentFromZeroReg = false;
	size_t gradientsBufferLength = gradientsBufferSize / sizeof(float);
	const float* mockDropoutLayerInputGradientsBuffer = mockDropoutLayer.GetInputGradientsBuffer();
	const float maxDiff = 0.000001f;
	const float maxDiffPercentage = 0.001f;
	const float maxDiffPercentageThreshold = 0.00000001f;
	CompareBuffers(dropoutLayerInputGradientsBuffer, mockDropoutLayerInputGradientsBuffer, gradientsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(dropoutLayerInputGradientsBuffer));

	TestingAssert(foundDifferentFromZeroMock, "All mock dropout layer input gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data width: " +
		to_string(inputDataWidth) + "; Input data height: " + to_string(inputDataHeight) + "; Input data count: " + to_string(inputDataCount) + "; Drop probability: " +
		to_string(dropProbability));
	TestingAssert(foundDifferentFromZeroReg, "All dropout layer input gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data width: " +
		to_string(inputDataWidth) + "; Input data height: " + to_string(inputDataHeight) + "; Input data count: " + to_string(inputDataCount) + "; Drop probability: " +
		to_string(dropProbability));
	TestingAssert(correctResult, "Incorrect backward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
		"; First different mock input gradient: " + to_string(firstDifferentMock) + "; First different regular input gradient: " + to_string(firstDifferentReg) +
		"; Input num channels: " + to_string(inputNumChannels) + "; Input data width: " + to_string(inputDataWidth) + "; Input data height: " +
		to_string(inputDataHeight) + "; Input data count: " + to_string(inputDataCount) + "; Drop probability: " + to_string(dropProbability));

	cout << "Backward prop passed. Input num channels: " << inputNumChannels << "; Input width: " << inputDataWidth << "; Input height: " << inputDataHeight <<
		"; Input data count: " << inputDataCount << "; Drop probability: " << dropProbability << endl;
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

void TestDropoutLayer::TestDoForwardProp()
{
	// lastBatch == true

	// dropProbability == 0.2f
	TestDoForwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 119 /*inputDataCount*/, 0.2f /*dropProbability*/);
	TestDoForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 97 /*inputDataCount*/, 0.2f /*dropProbability*/);
	TestDoForwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 0.2f /*dropProbability*/);
	TestDoForwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, 0.2f /*dropProbability*/);
	TestDoForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 13 /*inputDataCount*/, 0.2f /*dropProbability*/);

	// dropProbability == 0.5f
	TestDoForwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 119 /*inputDataCount*/, 0.5f /*dropProbability*/);
	TestDoForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 97 /*inputDataCount*/, 0.5f /*dropProbability*/);
	TestDoForwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 0.5f /*dropProbability*/);
	TestDoForwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, 0.5f /*dropProbability*/);
	TestDoForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 13 /*inputDataCount*/, 0.5f /*dropProbability*/);

	// dropProbability == 0.7f
	TestDoForwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 119 /*inputDataCount*/, 0.7f /*dropProbability*/);
	TestDoForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 97 /*inputDataCount*/, 0.7f /*dropProbability*/);
	TestDoForwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 0.7f /*dropProbability*/);
	TestDoForwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, 0.7f /*dropProbability*/);
	TestDoForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 13 /*inputDataCount*/, 0.7f /*dropProbability*/);


	// lastBatch == false

	// dropProbability == 0.2f
	TestDoForwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/);
	TestDoForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/);
	TestDoForwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/);
	TestDoForwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/);
	TestDoForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/);

	// dropProbability == 0.5f
	TestDoForwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/);
	TestDoForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/);
	TestDoForwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/);
	TestDoForwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/);
	TestDoForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/);

	// dropProbability == 0.7f
	TestDoForwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/);
	TestDoForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/);
	TestDoForwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/);
	TestDoForwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/);
	TestDoForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/);
}

/*
	Current speed record over 1000 launches: 
*/
//void TestDropoutLayer::TestForwardPropSpeed()
//{
//	// Creating cuBLAS handle to use in tests.
//	cublasHandle_t cublasHandle;
//	CudaCublasAssert(cublasCreate(&cublasHandle));
//
//	// Creating layers.
//	uint inputNumChannels = 256;
//	uint inputDataWidth = 13;
//	uint inputDataHeight = 13;
//	uint inputDataCount = 128;
//	uint numNeurons = 2048;
//	MockInputLayer inputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount);
//	float weightsDeviation = 0.01f;
//	float biasesInitialValue = 1.0f;
//	ActivationType activationType = ActivationType::ReLu;
//	StandardLayer standardLayer(ParallelismMode::Data, 0, cublasHandle, 0, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false,
//		numNeurons, true, weightsDeviation, true, biasesInitialValue, activationType);
//	standardLayer.AddPrevLayer(&inputLayer);
//
//	// Doing forward prop and measuring time.
//	inputLayer.LoadInputs();
//	inputLayer.DoForwardProp();
//	standardLayer.LoadInputs();
//	const uint c_timesToLaunch = 1000;
//	high_resolution_clock::time_point startTime = high_resolution_clock::now();
//	for (uint i = 0; i < c_timesToLaunch; ++i)
//	{
//		standardLayer.DoForwardProp();
//	}
//	CudaAssert(cudaDeviceSynchronize());
//	high_resolution_clock::time_point endTime = high_resolution_clock::now();
//
//	// Reporting time.
//	long long durationInMilliseconds = duration_cast<milliseconds>(endTime - startTime).count();
//	cout << "Forward prop took " << (float)durationInMilliseconds / (float)c_timesToLaunch << "ms in average to process." << endl;
//
//	// Destroying cuBLAS handle.
//	CudaCublasAssert(cublasDestroy(cublasHandle));
//}

void TestDropoutLayer::TestDoBackwardProp()
{
	// lastBatch == true

	// dropProbability == 0.2f
	TestDoBackwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 119 /*inputDataCount*/, 0.2f /*dropProbability*/);
	TestDoBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 97 /*inputDataCount*/, 0.2f /*dropProbability*/);
	TestDoBackwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 0.2f /*dropProbability*/);
	TestDoBackwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, 0.2f /*dropProbability*/);
	TestDoBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 13 /*inputDataCount*/, 0.2f /*dropProbability*/);

	// dropProbability == 0.5f
	TestDoBackwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 119 /*inputDataCount*/, 0.5f /*dropProbability*/);
	TestDoBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 97 /*inputDataCount*/, 0.5f /*dropProbability*/);
	TestDoBackwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 0.5f /*dropProbability*/);
	TestDoBackwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, 0.5f /*dropProbability*/);
	TestDoBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 13 /*inputDataCount*/, 0.5f /*dropProbability*/);

	// dropProbability == 0.7f
	TestDoBackwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 119 /*inputDataCount*/, 0.7f /*dropProbability*/);
	TestDoBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 97 /*inputDataCount*/, 0.7f /*dropProbability*/);
	TestDoBackwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 0.7f /*dropProbability*/);
	TestDoBackwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, 0.7f /*dropProbability*/);
	TestDoBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 13 /*inputDataCount*/, 0.7f /*dropProbability*/);


	// lastBatch == false

	// dropProbability == 0.2f
	TestDoBackwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/);
	TestDoBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/);
	TestDoBackwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/);
	TestDoBackwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/);
	TestDoBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.2f /*dropProbability*/);

	// dropProbability == 0.5f
	TestDoBackwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/);
	TestDoBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/);
	TestDoBackwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/);
	TestDoBackwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/);
	TestDoBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.5f /*dropProbability*/);

	// dropProbability == 0.7f
	TestDoBackwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/);
	TestDoBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/);
	TestDoBackwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/);
	TestDoBackwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/);
	TestDoBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 0.7f /*dropProbability*/);
}