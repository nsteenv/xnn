// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for response normalization layer.
// Created: 02/11/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/testresponsenormalizationlayer.cuh"

TestResponseNormalizationLayer::TestResponseNormalizationLayer(string outputFolder)
{
	m_outputFolder = outputFolder;

	// Registering tests.
	m_responseNormalizationLayerTests["doforwardprop"] = &TestResponseNormalizationLayer::TestDoForwardProp;
	m_responseNormalizationLayerTests["dobackwardprop"] = &TestResponseNormalizationLayer::TestDoBackwardProp;
}


bool TestResponseNormalizationLayer::HasTest(string testName)
{
	auto test = m_responseNormalizationLayerTests.find(testName);
	return test != m_responseNormalizationLayerTests.end();
}

void TestResponseNormalizationLayer::RunTest(string testName)
{
	auto test = m_responseNormalizationLayerTests.find(testName);
	TestingAssert(test != m_responseNormalizationLayerTests.end(), "Test not found!");

	((*this).*(test->second))();
}

void TestResponseNormalizationLayer::RunAllTests()
{
	for (auto test = m_responseNormalizationLayerTests.begin(); test != m_responseNormalizationLayerTests.end(); ++test)
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

void TestResponseNormalizationLayer::TestDoForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint depth,
	float bias, float alphaCoeff, float betaCoeff)
{
	// Creating layers.
	MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount);
	MockResponseNormalizationLayer mockReNormLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, depth, bias,
		alphaCoeff, betaCoeff);
	mockReNormLayer.AddPrevLayer(&mockInputLayer);
	ResponseNormalizationLayer reNormLayer(ParallelismMode::Data, 0, 0, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false,
		depth, bias, alphaCoeff, betaCoeff, false);
	reNormLayer.AddPrevLayer(&mockInputLayer);

	// Doing forward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	mockInputLayer.LoadInputs();
	mockInputLayer.DoForwardProp(propagationMode);
	mockReNormLayer.LoadInputs();
	reNormLayer.LoadInputs();
	reNormLayer.DoForwardProp(propagationMode);
	mockReNormLayer.DoForwardProp(propagationMode);
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	size_t activationsBufferSize = mockReNormLayer.GetActivationBufferSize();
	float* reNormLayerActivationBuffer;
	CudaAssert(cudaMallocHost<float>(&reNormLayerActivationBuffer, activationsBufferSize));
	CudaAssert(cudaMemcpy(reNormLayerActivationBuffer, reNormLayer.GetActivationDataBuffer(),
		activationsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult = true;
	size_t numDifferences = 0;
	float firstDifference = 0.f;
	float firstDifferentMock = 0.f;
	float firstDifferentReg = 0.f;
	bool foundDifferentFromZeroMock = false;
	bool foundDifferentFromZeroReg = false;
	size_t activationsBufferLength = activationsBufferSize / sizeof(float);
	const float* mockReNormLayerActivationBuffer = mockReNormLayer.GetActivationDataBuffer();
	const float maxDiff = 0.0001f;
	const float maxDiffPercentage = 0.1f;
	const float maxDiffPercentageThreshold = 0.00005f;
	CompareBuffers(reNormLayerActivationBuffer, mockReNormLayerActivationBuffer, activationsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(reNormLayerActivationBuffer));

	TestingAssert(foundDifferentFromZeroMock, "All mock response normalization activations are zeros! Input num channels: " + to_string(inputNumChannels) +
		"; Input data count: " + to_string(inputDataCount) + "; Depth: " + to_string(depth) + "; Bias: " + to_string(bias) + "; Alha coeff: " + to_string(alphaCoeff) +
		"; Beta coeff: " + to_string(betaCoeff));
	TestingAssert(foundDifferentFromZeroReg, "All response normalization activations are zeros! Input num channels: " + to_string(inputNumChannels) +
		"; Input data count: " + to_string(inputDataCount) + "; Depth: " + to_string(depth) + "; Bias: " + to_string(bias) + "; Alha coeff: " + to_string(alphaCoeff) +
		"; Beta coeff: " + to_string(betaCoeff));
	TestingAssert(correctResult, "Incorrect forward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
		"; First different mock activation: " + to_string(firstDifferentMock) + "; First different regular activation: " + to_string(firstDifferentReg) +
		"; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) + "; Depth: " + to_string(depth) +
		"; Bias: " + to_string(bias) + "; Alha coeff: " + to_string(alphaCoeff) + "; Beta coeff: " + to_string(betaCoeff));

	cout << "Forward prop passed. Input num channels: " << inputNumChannels << "; Input data count: " << inputDataCount << "; Depth: " << depth <<
		"; Bias: " << bias << "; Alpha coeff: " << alphaCoeff << "; Beta coeff: " << betaCoeff << endl;
}

void TestResponseNormalizationLayer::TestDoBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, uint depth,
	float bias, float alphaCoeff, float betaCoeff)
{
	// Creating layers.
	MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount);
	MockResponseNormalizationLayer mockReNormLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, depth, bias,
		alphaCoeff, betaCoeff);
	mockReNormLayer.AddPrevLayer(&mockInputLayer);
	ResponseNormalizationLayer reNormLayer(ParallelismMode::Data, 0, 0, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false,
		depth, bias, alphaCoeff, betaCoeff, false);
	reNormLayer.AddPrevLayer(&mockInputLayer);
	MockOutputLayer mockOutputLayer(inputNumChannels * inputDataWidth * inputDataHeight, inputDataCount, LossFunctionType::LogisticRegression, false, 0, true);
	mockReNormLayer.AddNextLayer(&mockOutputLayer);
	reNormLayer.AddNextLayer(&mockOutputLayer);

	// Doing forward and backward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	mockInputLayer.LoadInputs();
	mockInputLayer.DoForwardProp(propagationMode);
	mockReNormLayer.LoadInputs();
	reNormLayer.LoadInputs();
	reNormLayer.DoForwardProp(propagationMode);
	mockReNormLayer.DoForwardProp(propagationMode);
	mockOutputLayer.DoBackwardProp();
	reNormLayer.LoadActivationGradients();
	reNormLayer.DoBackwardProp();
	mockReNormLayer.LoadActivationGradients();
	mockReNormLayer.DoBackwardProp();
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	size_t inputGradientsBufferSize = mockInputLayer.GetActivationBufferSize();
	float* reNormLayerInputGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&reNormLayerInputGradientsBuffer, inputGradientsBufferSize));
	CudaAssert(cudaMemcpy(reNormLayerInputGradientsBuffer, reNormLayer.GetInputGradientsBuffer(), inputGradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult = true;
	size_t numDifferences = 0;
	float firstDifference = 0.f;
	float firstDifferentMock = 0.f;
	float firstDifferentReg = 0.f;
	bool foundDifferentFromZeroMock = false;
	bool foundDifferentFromZeroReg = false;
	size_t inputGradientsBufferLength = inputGradientsBufferSize / sizeof(float);
	const float* mockReNormLayerInputGradientsBuffer = mockReNormLayer.GetInputGradientsBuffer();
	const float maxDiff = 0.0001f;
	const float maxDiffPercentage = 0.1f;
	const float maxDiffPercentageThreshold = 0.00005f;
	CompareBuffers(reNormLayerInputGradientsBuffer, mockReNormLayerInputGradientsBuffer, inputGradientsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(reNormLayerInputGradientsBuffer));

	TestingAssert(foundDifferentFromZeroMock, "All mock response normalization input gradients are zeros! Input num channels: " + to_string(inputNumChannels) +
		"; Input data count: " + to_string(inputDataCount) + "; Depth: " + to_string(depth) + "; Bias: " + to_string(bias) + "; Alha coeff: " + to_string(alphaCoeff) +
		"; Beta coeff: " + to_string(betaCoeff));
	TestingAssert(foundDifferentFromZeroReg, "All response normalization input gradients are zeros! Input num channels: " + to_string(inputNumChannels) +
		"; Input data count: " + to_string(inputDataCount) + "; Depth: " + to_string(depth) + "; Bias: " + to_string(bias) + "; Alha coeff: " + to_string(alphaCoeff) +
		"; Beta coeff: " + to_string(betaCoeff));
	TestingAssert(correctResult, "Incorrect backward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
		"; First different mock input gradient: " + to_string(firstDifferentMock) + "; First different regular input gradient: " + to_string(firstDifferentReg) +
		"; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) + "; Depth: " + to_string(depth) +
		"; Bias: " + to_string(bias) + "; Alha coeff: " + to_string(alphaCoeff) + "; Beta coeff: " + to_string(betaCoeff));

	cout << "Backward prop passed. Input num channels: " << inputNumChannels << "; Input data count: " << inputDataCount << "; Depth: " << depth <<
		"; Bias: " << bias << "; Alpha coeff: " << alphaCoeff << "; Beta coeff: " << betaCoeff << endl;
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

void TestResponseNormalizationLayer::TestDoForwardProp()
{
	// lastBatch == true
	
	TestDoForwardProp(48 /*inputNumChannels*/, 56 /*inputDataWidth*/, 56 /*inputDataHeight*/, 127 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoForwardProp(64 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 119 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoForwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 97 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoForwardProp(192 /*inputNumChannels*/, 14 /*inputDataWidth*/, 14 /*inputDataHeight*/, 74 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoForwardProp(384 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 16 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);

	// lastBatch == false

	TestDoForwardProp(48 /*inputNumChannels*/, 56 /*inputDataWidth*/, 56 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoForwardProp(64 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoForwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoForwardProp(192 /*inputNumChannels*/, 14 /*inputDataWidth*/, 14 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoForwardProp(384 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);

	// Various formula parameters

	TestDoForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 6 /*depth*/, 1.2f /*bias*/,
		0.0003f /*alphaCoeff*/, 2.0f /*betaCoeff*/);
	TestDoForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 3 /*depth*/, 6.7f /*bias*/,
		0.03f /*alphaCoeff*/, 0.2f /*betaCoeff*/);
	TestDoForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 2 /*depth*/, 0.5f /*bias*/,
		0.001f /*alphaCoeff*/, 3.0f /*betaCoeff*/);
	TestDoForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 10 /*depth*/, 3.1f /*bias*/,
		0.009f /*alphaCoeff*/, 1.0f /*betaCoeff*/);
	TestDoForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 4 /*depth*/, 4.9f /*bias*/,
		1.0f /*alphaCoeff*/, 1.0f /*betaCoeff*/);
	TestDoForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 8 /*depth*/, 2.8f /*bias*/,
		0.0003f /*alphaCoeff*/, 2.0f /*betaCoeff*/);
}

void TestResponseNormalizationLayer::TestDoBackwardProp()
{
	// lastBatch == true

	TestDoBackwardProp(48 /*inputNumChannels*/, 56 /*inputDataWidth*/, 56 /*inputDataHeight*/, 127 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoBackwardProp(64 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 119 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoBackwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 97 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoBackwardProp(192 /*inputNumChannels*/, 14 /*inputDataWidth*/, 14 /*inputDataHeight*/, 74 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoBackwardProp(384 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 16 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);

	// lastBatch == false

	TestDoBackwardProp(48 /*inputNumChannels*/, 56 /*inputDataWidth*/, 56 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoBackwardProp(64 /*inputNumChannels*/, 55 /*inputDataWidth*/, 55 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoBackwardProp(128 /*inputNumChannels*/, 27 /*inputDataWidth*/, 27 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoBackwardProp(192 /*inputNumChannels*/, 14 /*inputDataWidth*/, 14 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);
	TestDoBackwardProp(384 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, 5 /*depth*/, 2.0f /*bias*/,
		0.0001f /*alphaCoeff*/, 0.75f /*betaCoeff*/);

	// Various formula parameters

	TestDoBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 6 /*depth*/, 1.2f /*bias*/,
		0.0003f /*alphaCoeff*/, 2.0f /*betaCoeff*/);
	TestDoBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 3 /*depth*/, 6.7f /*bias*/,
		0.03f /*alphaCoeff*/, 0.2f /*betaCoeff*/);
	TestDoBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 2 /*depth*/, 0.5f /*bias*/,
		0.001f /*alphaCoeff*/, 3.0f /*betaCoeff*/);
	TestDoBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 10 /*depth*/, 3.1f /*bias*/,
		0.009f /*alphaCoeff*/, 1.0f /*betaCoeff*/);
	TestDoBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 4 /*depth*/, 4.9f /*bias*/,
		1.0f /*alphaCoeff*/, 1.0f /*betaCoeff*/);
	TestDoBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 32 /*inputDataCount*/, 8 /*depth*/, 2.8f /*bias*/,
		0.0003f /*alphaCoeff*/, 2.0f /*betaCoeff*/);
}