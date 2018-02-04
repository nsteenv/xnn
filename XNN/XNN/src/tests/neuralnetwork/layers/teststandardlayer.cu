// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for standard layer.
// Created: 02/13/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/teststandardlayer.cuh"

TestStandardLayer::TestStandardLayer(string outputFolder)
{
	m_outputFolder = outputFolder;

	// Registering tests.
	m_standardLayerTests["doforwardprop"] = &TestStandardLayer::TestDoForwardProp;
	m_standardLayerTests["forwardpropspeed"] = &TestStandardLayer::TestForwardPropSpeed;
	m_standardLayerTests["dobackwardprop"] = &TestStandardLayer::TestDoBackwardProp;
}


bool TestStandardLayer::HasTest(string testName)
{
	auto test = m_standardLayerTests.find(testName);
	return test != m_standardLayerTests.end();
}

void TestStandardLayer::RunTest(string testName)
{
	auto test = m_standardLayerTests.find(testName);
	TestingAssert(test != m_standardLayerTests.end(), "Test not found!");

	((*this).*(test->second))();
}

void TestStandardLayer::RunAllTests()
{
	for (auto test = m_standardLayerTests.begin(); test != m_standardLayerTests.end(); ++test)
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

void TestStandardLayer::TestDoForwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, cublasHandle_t cublasHandle,
	uint numNeurons)
{
	// Creating layers.
	MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount);
	float weightsDeviation = 0.01f;
	float biasesInitialValue = 1.0f;
	ActivationType activationType = ActivationType::ReLu;
	float weightsUpdateMomentum = 0.9f;
	float weightsUpdateDecay = 0.0005f;
	float weightsUpdateLearningRateProgressStep = 0.25f;
	float weightsUpdateStartingLearningRate = 0.01f;
	float weightsUpdateLearningRateUpdateFactor = 0.2f;
	float biasesUpdateMomentum = 0.9f;
	float biasesUpdateDecay = 0.f;
	float biasesUpdateLearningRateProgressStep = 0.5f;
	float biasesUpdateStartingLearningRate = 0.02f;
	float biasesUpdateLearningRateUpdateFactor = 0.1f;
	MockStandardLayer mockStandardLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, numNeurons, weightsDeviation, biasesInitialValue,
		weightsUpdateMomentum, weightsUpdateDecay, weightsUpdateLearningRateProgressStep, weightsUpdateStartingLearningRate, weightsUpdateLearningRateUpdateFactor,
		biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep, biasesUpdateStartingLearningRate, biasesUpdateLearningRateUpdateFactor,
		activationType);
	mockStandardLayer.AddPrevLayer(&mockInputLayer);
	StandardLayer standardLayer(ParallelismMode::Data, 0, 0, cublasHandle, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false,
		numNeurons, false, weightsDeviation, false, biasesInitialValue, weightsUpdateMomentum, weightsUpdateDecay, weightsUpdateLearningRateProgressStep,
		weightsUpdateStartingLearningRate, weightsUpdateLearningRateUpdateFactor, biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep,
		biasesUpdateStartingLearningRate, biasesUpdateLearningRateUpdateFactor, activationType, false);
	standardLayer.CopyWeightsFromHost(mockStandardLayer.GetWeightsBuffer());
	standardLayer.CopyBiasesFromHost(mockStandardLayer.GetBiasesBuffer());
	standardLayer.AddPrevLayer(&mockInputLayer);

	// Doing forward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	mockInputLayer.LoadInputs();
	mockInputLayer.DoForwardProp(propagationMode);
	mockStandardLayer.LoadInputs();
	standardLayer.LoadInputs();
	standardLayer.DoForwardProp(propagationMode);
	mockStandardLayer.DoForwardProp(propagationMode);
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	size_t activationsBufferSize = mockStandardLayer.GetActivationBufferSize();
	float* standardLayerActivationBuffer;
	CudaAssert(cudaMallocHost<float>(&standardLayerActivationBuffer, activationsBufferSize));
	CudaAssert(cudaMemcpy(standardLayerActivationBuffer, standardLayer.GetActivationDataBuffer(), activationsBufferSize, cudaMemcpyDeviceToHost));

	// Checking correctness.
	bool correctResult = true;
	size_t numDifferences = 0;
	float firstDifference = 0.f;
	float firstDifferentMock = 0.f;
	float firstDifferentReg = 0.f;
	bool foundDifferentFromZeroMock = false;
	bool foundDifferentFromZeroReg = false;
	size_t activationsBufferLength = activationsBufferSize / sizeof(float);
	const float* mockStandardLayerActivationBuffer = mockStandardLayer.GetActivationDataBuffer();
	const float maxDiff = 0.01f;
	const float maxDiffPercentage = 0.1f;
	const float maxDiffPercentageThreshold = 0.0005f;
	CompareBuffers(standardLayerActivationBuffer, mockStandardLayerActivationBuffer, activationsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(standardLayerActivationBuffer));

	TestingAssert(foundDifferentFromZeroMock, "All mock standard layer activations are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
		to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
	TestingAssert(foundDifferentFromZeroReg, "All standard layer activations are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
		to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
	TestingAssert(correctResult, "Incorrect forward prop! Num differences: " + to_string(numDifferences) + "; First difference: " + to_string(firstDifference) +
		"; First different mock activation: " + to_string(firstDifferentMock) + "; First different regular activation: " + to_string(firstDifferentReg) +
		"; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));

	cout << "Forward prop passed. Input num channels: " << inputNumChannels << "; Input data count: " << inputDataCount << "; Number of neurons: " << numNeurons << endl;
}

void TestStandardLayer::TestDoBackwardProp(uint inputNumChannels, uint inputDataWidth, uint inputDataHeight, uint inputDataCount, cublasHandle_t cublasHandle,
	uint numNeurons)
{
	// Creating layers.
	MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount);
	float weightsDeviation = 0.01f;
	float biasesInitialValue = 1.0f;
	ActivationType activationType = ActivationType::ReLu;
	float weightsUpdateMomentum = 0.9f;
	float weightsUpdateDecay = 0.0005f;
	float weightsUpdateLearningRateProgressStep = 0.25f;
	float weightsUpdateStartingLearningRate = 0.01f;
	float weightsUpdateLearningRateUpdateFactor = 0.2f;
	float biasesUpdateMomentum = 0.9f;
	float biasesUpdateDecay = 0.f;
	float biasesUpdateLearningRateProgressStep = 0.5f;
	float biasesUpdateStartingLearningRate = 0.02f;
	float biasesUpdateLearningRateUpdateFactor = 0.1f;
	MockStandardLayer mockStandardLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, numNeurons, weightsDeviation, biasesInitialValue,
		weightsUpdateMomentum, weightsUpdateDecay, weightsUpdateLearningRateProgressStep, weightsUpdateStartingLearningRate, weightsUpdateLearningRateUpdateFactor,
		biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep, biasesUpdateStartingLearningRate, biasesUpdateLearningRateUpdateFactor,
		activationType);
	mockStandardLayer.AddPrevLayer(&mockInputLayer);
	StandardLayer standardLayer(ParallelismMode::Data, 0, 0, cublasHandle, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false,
		numNeurons, false, weightsDeviation, false, biasesInitialValue, weightsUpdateMomentum, weightsUpdateDecay, weightsUpdateLearningRateProgressStep,
		weightsUpdateStartingLearningRate, weightsUpdateLearningRateUpdateFactor, biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep,
		biasesUpdateStartingLearningRate, biasesUpdateLearningRateUpdateFactor, activationType, false);
	standardLayer.CopyWeightsFromHost(mockStandardLayer.GetWeightsBuffer());
	standardLayer.CopyBiasesFromHost(mockStandardLayer.GetBiasesBuffer());
	standardLayer.AddPrevLayer(&mockInputLayer);
	MockOutputLayer mockOutputLayer(mockStandardLayer.GetActivationDataSize(), inputDataCount, LossFunctionType::LogisticRegression, false, 0, true);
	mockStandardLayer.AddNextLayer(&mockOutputLayer);
	standardLayer.AddNextLayer(&mockOutputLayer);

	// Doing forward and backward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	mockInputLayer.LoadInputs();
	mockInputLayer.DoForwardProp(propagationMode);
	mockStandardLayer.LoadInputs();
	standardLayer.LoadInputs();
	standardLayer.DoForwardProp(propagationMode);
	mockStandardLayer.DoForwardProp(propagationMode);
	mockOutputLayer.DoBackwardProp();
	standardLayer.LoadActivationGradients();
	standardLayer.DoBackwardProp();
	mockStandardLayer.LoadActivationGradients();
	mockStandardLayer.DoBackwardProp();
	CudaAssert(cudaDeviceSynchronize());

	// Transferring input gradients results to host.
	size_t inputGradientsBufferSize = mockInputLayer.GetActivationBufferSize();
	float* standardLayerInputGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&standardLayerInputGradientsBuffer, inputGradientsBufferSize));
	CudaAssert(cudaMemcpy(standardLayerInputGradientsBuffer, standardLayer.GetInputGradientsBuffer(), inputGradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking input gradients correctness.
	bool correctResult = true;
	size_t numDifferences = 0;
	float firstDifference = 0.f;
	float firstDifferentMock = 0.f;
	float firstDifferentReg = 0.f;
	bool foundDifferentFromZeroMock = false;
	bool foundDifferentFromZeroReg = false;
	size_t inputGradientsBufferLength = inputGradientsBufferSize / sizeof(float);
	const float* mockStandardLayerInputGradientsBuffer = mockStandardLayer.GetInputGradientsBuffer();
	const float maxDiff = 0.01f;
	const float maxDiffPercentage = 0.1f;
	const float maxDiffPercentageThreshold = 0.0005f;
	CompareBuffers(standardLayerInputGradientsBuffer, mockStandardLayerInputGradientsBuffer, inputGradientsBufferLength, maxDiff, maxDiffPercentage, maxDiffPercentageThreshold,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(standardLayerInputGradientsBuffer));

	TestingAssert(foundDifferentFromZeroMock, "All mock standard layer input gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
		to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
	TestingAssert(foundDifferentFromZeroReg, "All standard layer input gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
		to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
	TestingAssert(correctResult, "Incorrect backward prop (input gradients)! Num differences: " + to_string(numDifferences) + "; First difference: " +
		to_string(firstDifference) + "; First different mock input gradient: " + to_string(firstDifferentMock) + "; First different regular input gradient: " +
		to_string(firstDifferentReg) + "; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) +
		"; Number of neurons: " + to_string(numNeurons));

	// Transferring weights gradients results to host.
	size_t weightsGradientsBufferSize = mockStandardLayer.GetWeightsBufferSize();
	float* standardLayerWeightsGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&standardLayerWeightsGradientsBuffer, weightsGradientsBufferSize));
	CudaAssert(cudaMemcpy(standardLayerWeightsGradientsBuffer, standardLayer.GetWeightsGradientsBuffer(), weightsGradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking weights gradients correctness.
	correctResult = true;
	foundDifferentFromZeroMock = false;
	foundDifferentFromZeroReg = false;
	size_t weightsGradientsBufferLength = weightsGradientsBufferSize / sizeof(float);
	const float* mockStandardLayerWeightsGradientsBuffer = mockStandardLayer.GetWeightsGradientsBuffer();
	const float maxDiffWG = 0.01f;
	const float maxDiffPercentageWG = 0.1f;
	const float maxDiffPercentageThresholdWG = 0.005f;
	CompareBuffers(standardLayerWeightsGradientsBuffer, mockStandardLayerWeightsGradientsBuffer, weightsGradientsBufferLength, maxDiffWG, maxDiffPercentageWG,
		maxDiffPercentageThresholdWG, correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock,
		foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(standardLayerWeightsGradientsBuffer));

	TestingAssert(foundDifferentFromZeroMock, "All mock standard layer weights gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
		to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
	TestingAssert(foundDifferentFromZeroReg, "All standard layer weights gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
		to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
	TestingAssert(correctResult, "Incorrect backward prop (weights gradients)! Num differences: " + to_string(numDifferences) + "; First difference: " +
		to_string(firstDifference) + "; First different mock weights gradient: " + to_string(firstDifferentMock) + "; First different regular weights gradient: " +
		to_string(firstDifferentReg) + "; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) +
		"; Number of neurons: " + to_string(numNeurons));

	// Transferring biases gradients results to host.
	size_t biasesGradientsBufferSize = mockStandardLayer.GetBiasesBufferSize();
	float* standardLayerBiasesGradientsBuffer;
	CudaAssert(cudaMallocHost<float>(&standardLayerBiasesGradientsBuffer, biasesGradientsBufferSize));
	CudaAssert(cudaMemcpy(standardLayerBiasesGradientsBuffer, standardLayer.GetBiasesGradientsBuffer(), biasesGradientsBufferSize, cudaMemcpyDeviceToHost));

	// Checking biases gradients correctness.
	correctResult = true;
	foundDifferentFromZeroMock = false;
	foundDifferentFromZeroReg = false;
	size_t biasesGradientsBufferLength = biasesGradientsBufferSize / sizeof(float);
	const float* mockStandardLayerBiasesGradientsBuffer = mockStandardLayer.GetBiasesGradientsBuffer();
	const float maxDiffBG = 0.01f;
	const float maxDiffPercentageBG = 0.1f;
	const float maxDiffPercentageThresholdBG = 0.005f;
	CompareBuffers(standardLayerBiasesGradientsBuffer, mockStandardLayerBiasesGradientsBuffer, biasesGradientsBufferLength, maxDiffBG, maxDiffPercentageBG,
		maxDiffPercentageThresholdBG, correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock,
		foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(standardLayerBiasesGradientsBuffer));

	TestingAssert(foundDifferentFromZeroMock, "All mock standard layer biases gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
		to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
	TestingAssert(foundDifferentFromZeroReg, "All standard layer biases gradients are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
		to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
	TestingAssert(correctResult, "Incorrect backward prop (biases gradients)! Num differences: " + to_string(numDifferences) + "; First difference: " +
		to_string(firstDifference) + "; First different mock biases gradient: " + to_string(firstDifferentMock) + "; First different regular biases gradient: " +
		to_string(firstDifferentReg) + "; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) +
		"; Number of neurons: " + to_string(numNeurons));

	// Updating parameters.
	float progress = 0.6f;
	standardLayer.UpdateLayerParameters(progress);
	mockStandardLayer.UpdateLayerParameters(progress);
	CudaAssert(cudaDeviceSynchronize());

	// Transferring weights to host.
	size_t weightsBufferSize = mockStandardLayer.GetWeightsBufferSize();
	float* standardLayerWeightsBuffer;
	CudaAssert(cudaMallocHost<float>(&standardLayerWeightsBuffer, weightsBufferSize));
	CudaAssert(cudaMemcpy(standardLayerWeightsBuffer, standardLayer.GetWeightsBuffer(), weightsBufferSize, cudaMemcpyDeviceToHost));

	// Checking weights correctness.
	correctResult = true;
	foundDifferentFromZeroMock = false;
	foundDifferentFromZeroReg = false;
	size_t weightsBufferLength = weightsBufferSize / sizeof(float);
	const float* mockStandardLayerWeightsBuffer = mockStandardLayer.GetWeightsBuffer();
	const float maxDiffW = 0.01f;
	const float maxDiffPercentageW = 0.1f;
	const float maxDiffPercentageThresholdW = 0.005f;
	CompareBuffers(standardLayerWeightsBuffer, mockStandardLayerWeightsBuffer, weightsBufferLength, maxDiffW, maxDiffPercentageW, maxDiffPercentageThresholdW,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(standardLayerWeightsBuffer));

	TestingAssert(foundDifferentFromZeroMock, "All mock standard layer weights are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
		to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
	TestingAssert(foundDifferentFromZeroReg, "All standard layer weights are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
		to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
	TestingAssert(correctResult, "Incorrect backward prop (updated weights)! Num differences: " + to_string(numDifferences) + "; First difference: " +
		to_string(firstDifference) + "; First different mock weights: " + to_string(firstDifferentMock) + "; First different regular weights: " +
		to_string(firstDifferentReg) + "; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) +
		"; Number of neurons: " + to_string(numNeurons));

	// Transferring biases to host.
	size_t biasesBufferSize = mockStandardLayer.GetBiasesBufferSize();
	float* standardLayerBiasesBuffer;
	CudaAssert(cudaMallocHost<float>(&standardLayerBiasesBuffer, biasesBufferSize));
	CudaAssert(cudaMemcpy(standardLayerBiasesBuffer, standardLayer.GetBiasesBuffer(), biasesBufferSize, cudaMemcpyDeviceToHost));

	// Checking biases correctness.
	correctResult = true;
	foundDifferentFromZeroMock = false;
	foundDifferentFromZeroReg = false;
	size_t biasesBufferLength = biasesBufferSize / sizeof(float);
	const float* mockStandardLayerBiasesBuffer = mockStandardLayer.GetBiasesBuffer();
	const float maxDiffB = 0.01f;
	const float maxDiffPercentageB = 0.1f;
	const float maxDiffPercentageThresholdB = 0.005f;
	CompareBuffers(standardLayerBiasesBuffer, mockStandardLayerBiasesBuffer, biasesBufferLength, maxDiffB, maxDiffPercentageB, maxDiffPercentageThresholdB,
		correctResult, numDifferences, firstDifference, firstDifferentMock, firstDifferentReg, foundDifferentFromZeroMock, foundDifferentFromZeroReg);

	CudaAssert(cudaFreeHost(standardLayerBiasesBuffer));

	TestingAssert(foundDifferentFromZeroMock, "All mock standard layer biases are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
		to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
	TestingAssert(foundDifferentFromZeroReg, "All standard layer biases are zeros! Input num channels: " + to_string(inputNumChannels) + "; Input data count: " +
		to_string(inputDataCount) + "; Number of neurons: " + to_string(numNeurons));
	TestingAssert(correctResult, "Incorrect backward prop (updated biases)! Num differences: " + to_string(numDifferences) + "; First difference: " +
		to_string(firstDifference) + "; First different mock biases: " + to_string(firstDifferentMock) + "; First different regular biases: " +
		to_string(firstDifferentReg) + "; Input num channels: " + to_string(inputNumChannels) + "; Input data count: " + to_string(inputDataCount) +
		"; Number of neurons: " + to_string(numNeurons));

	cout << "Backward prop passed. Input num channels: " << inputNumChannels << "; Input data count: " << inputDataCount << "; Number of neurons: " << numNeurons << endl;
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

void TestStandardLayer::TestDoForwardProp()
{
	// Creating cuBLAS handle to use in tests.
	cublasHandle_t cublasHandle;
	CudaCublasAssert(cublasCreate(&cublasHandle));

	// lastBatch == true
	TestDoForwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 119 /*inputDataCount*/, cublasHandle, 1000 /*numNeurons*/);
	TestDoForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 97 /*inputDataCount*/, cublasHandle, 2048 /*numNeurons*/);
	TestDoForwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, cublasHandle, 2048 /*numNeurons*/);
	TestDoForwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, cublasHandle, 2048 /*numNeurons*/);
	TestDoForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 13 /*inputDataCount*/, cublasHandle, 2048 /*numNeurons*/);

	// lastBatch == false
	TestDoForwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 128 /*inputDataCount*/, cublasHandle, 1000 /*numNeurons*/);
	TestDoForwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, cublasHandle, 2048 /*numNeurons*/);
	TestDoForwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, cublasHandle, 2048 /*numNeurons*/);
	TestDoForwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, cublasHandle, 2048 /*numNeurons*/);
	TestDoForwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, cublasHandle, 2048 /*numNeurons*/);

	// Destroying cuBLAS handle.
	CudaCublasAssert(cublasDestroy(cublasHandle));
}

/*
	Current speed record over 1000 launches: 7.585ms
*/
void TestStandardLayer::TestForwardPropSpeed()
{
	// Creating cuBLAS handle to use in tests.
	cublasHandle_t cublasHandle;
	CudaCublasAssert(cublasCreate(&cublasHandle));

	// Creating layers.
	uint inputNumChannels = 256;
	uint inputDataWidth = 13;
	uint inputDataHeight = 13;
	uint inputDataCount = 128;
	uint numNeurons = 2048;
	MockInputLayer mockInputLayer(inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount);
	float weightsDeviation = 0.01f;
	float biasesInitialValue = 1.0f;
	ActivationType activationType = ActivationType::ReLu;
	float weightsUpdateMomentum = 0.9f;
	float weightsUpdateDecay = 0.0005f;
	float weightsUpdateLearningRateProgressStep = 0.25f;
	float weightsUpdateStartingLearningRate = 0.01f;
	float weightsUpdateLearningRateUpdateFactor = 0.2f;
	float biasesUpdateMomentum = 0.9f;
	float biasesUpdateDecay = 0.f;
	float biasesUpdateLearningRateProgressStep = 0.5f;
	float biasesUpdateStartingLearningRate = 0.02f;
	float biasesUpdateLearningRateUpdateFactor = 0.1f;
	StandardLayer standardLayer(ParallelismMode::Data, 0, 0, cublasHandle, 0, 1, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, false,
		numNeurons, false, weightsDeviation, false, biasesInitialValue, weightsUpdateMomentum, weightsUpdateDecay, weightsUpdateLearningRateProgressStep,
		weightsUpdateStartingLearningRate, weightsUpdateLearningRateUpdateFactor, biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep,
		biasesUpdateStartingLearningRate, biasesUpdateLearningRateUpdateFactor, activationType, false);
	standardLayer.AddPrevLayer(&mockInputLayer);

	// Doing forward prop and measuring time.
	PropagationMode propagationMode = PropagationMode::Train;
	mockInputLayer.LoadInputs();
	mockInputLayer.DoForwardProp(propagationMode);
	standardLayer.LoadInputs();
	const uint c_timesToLaunch = 1000;
	high_resolution_clock::time_point startTime = high_resolution_clock::now();
	for (uint i = 0; i < c_timesToLaunch; ++i)
	{
		standardLayer.DoForwardProp(propagationMode);
	}	
	CudaAssert(cudaDeviceSynchronize());
	high_resolution_clock::time_point endTime = high_resolution_clock::now();

	// Reporting time.
	long long durationInMilliseconds = duration_cast<milliseconds>(endTime - startTime).count();
	cout << "Forward prop took " << (float)durationInMilliseconds / (float)c_timesToLaunch << "ms in average to process." << endl;

	// Destroying cuBLAS handle.
	CudaCublasAssert(cublasDestroy(cublasHandle));
}

void TestStandardLayer::TestDoBackwardProp()
{
	// Creating cuBLAS handle to use in tests.
	cublasHandle_t cublasHandle;
	CudaCublasAssert(cublasCreate(&cublasHandle));

	// lastBatch == true
	TestDoBackwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 119 /*inputDataCount*/, cublasHandle, 1000 /*numNeurons*/);
	TestDoBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 97 /*inputDataCount*/, cublasHandle, 2048 /*numNeurons*/);
	TestDoBackwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 55 /*inputDataCount*/, cublasHandle, 2048 /*numNeurons*/);
	TestDoBackwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 27 /*inputDataCount*/, cublasHandle, 2048 /*numNeurons*/);
	TestDoBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 13 /*inputDataCount*/, cublasHandle, 2048 /*numNeurons*/);

	// lastBatch == false
	TestDoBackwardProp(3 /*inputNumChannels*/, 64 /*inputDataWidth*/, 64 /*inputDataHeight*/, 128 /*inputDataCount*/, cublasHandle, 1000 /*numNeurons*/);
	TestDoBackwardProp(64 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, cublasHandle, 2048 /*numNeurons*/);
	TestDoBackwardProp(128 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, cublasHandle, 2048 /*numNeurons*/);
	TestDoBackwardProp(192 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, cublasHandle, 2048 /*numNeurons*/);
	TestDoBackwardProp(256 /*inputNumChannels*/, 13 /*inputDataWidth*/, 13 /*inputDataHeight*/, 128 /*inputDataCount*/, cublasHandle, 2048 /*numNeurons*/);

	// Destroying cuBLAS handle.
	CudaCublasAssert(cublasDestroy(cublasHandle));
}