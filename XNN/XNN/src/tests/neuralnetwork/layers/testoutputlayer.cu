// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for output layer.
// Created: 02/21/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/testoutputlayer.cuh"

TestOutputLayer::TestOutputLayer(string outputFolder)
{
	m_outputFolder = outputFolder;

	// Registering tests.
	m_outputLayerTests["doforwardprop"] = &TestOutputLayer::TestDoForwardProp;
	//m_outputLayerTests["forwardpropspeed"] = &TestOutputLayer::TestForwardPropSpeed;
}


bool TestOutputLayer::HasTest(string testName)
{
	auto test = m_outputLayerTests.find(testName);
	return test != m_outputLayerTests.end();
}

void TestOutputLayer::RunTest(string testName)
{
	auto test = m_outputLayerTests.find(testName);
	TestingAssert(test != m_outputLayerTests.end(), "Test not found!");

	((*this).*(test->second))();
}

void TestOutputLayer::RunAllTests()
{
	for (auto test = m_outputLayerTests.begin(); test != m_outputLayerTests.end(); ++test)
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

void TestOutputLayer::TestDoForwardProp(uint inputDataSize, uint inputDataCount)
{
	// Creating layers.
	MockInputLayer inputLayer(1, inputDataSize, 1, inputDataCount);
	float weightsDeviation = 0.001f;
	float biasesInitialValue = 0.1f;
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
	MockStandardLayer mockStandardLayer(1, inputDataSize, 1, inputDataCount, inputDataSize, weightsDeviation, biasesInitialValue,
		weightsUpdateMomentum, weightsUpdateDecay, weightsUpdateLearningRateProgressStep, weightsUpdateStartingLearningRate, weightsUpdateLearningRateUpdateFactor,
		biasesUpdateMomentum, biasesUpdateDecay, biasesUpdateLearningRateProgressStep, biasesUpdateStartingLearningRate, biasesUpdateLearningRateUpdateFactor,
		activationType, true);
	mockStandardLayer.AddPrevLayer(&inputLayer);
	SoftMaxLayer softMaxLayer(ParallelismMode::Model, 0, 0, inputDataSize, inputDataCount, false);
	softMaxLayer.AddPrevLayer(&mockStandardLayer);
	MockOutputLayer mockOutputLayer(inputDataSize, inputDataCount, LossFunctionType::LogisticRegression, true, 5);
	mockOutputLayer.AddPrevLayer(&softMaxLayer);
	OutputLayer outputLayer(0, 0, inputDataSize, inputDataCount, inputDataCount, LossFunctionType::LogisticRegression, true, 5, 0);
	outputLayer.AddPrevLayer(&softMaxLayer);

	// Creating random labels.
	vector<uint> labels;
	for (uint i = 0; i < inputDataCount; ++i)
	{
		labels.push_back((57 * i * i) % inputDataSize);
	}

	// Doing forward prop.
	PropagationMode propagationMode = PropagationMode::Train;
	inputLayer.LoadInputs();
	inputLayer.DoForwardProp(propagationMode);
	mockStandardLayer.LoadInputs();
	mockStandardLayer.DoForwardProp(propagationMode);
	softMaxLayer.LoadInputs();
	softMaxLayer.DoForwardProp(propagationMode);
	mockOutputLayer.LoadDataLabels(labels);
	outputLayer.LoadDataLabels(labels);
	mockOutputLayer.LoadInputs();
	outputLayer.LoadInputs();
	outputLayer.DoForwardProp(propagationMode);
	mockOutputLayer.DoForwardProp(propagationMode);
	CudaAssert(cudaDeviceSynchronize());

	// Transferring results to host.
	float* outputLayerLogLikelihoods = outputLayer.GetHostLogLikelihoods();
	float* mockOutputLayerLogLikelihoods = mockOutputLayer.GetLogLikelihoods();
	float* outputLayerScores = outputLayer.GetHostScores();
	float* mockOutputLayerScores = mockOutputLayer.GetScores();
	float* outputLayerMultipleGuessScores = outputLayer.GetHostMultipleGuessScores();
	float* mockOutputLayerMultipleGuessScores = mockOutputLayer.GetMultipleGuessScores();

	// Checking correctness.
	bool foundDifferentLogLikelihoods = false;
	bool foundDifferentScores = false;
	bool foundDifferentMultipleGuessScores = false;
	float firstDifference = 0.f;
	const float c_allowedDiff = 0.0001f;
	const float c_allowedDiffCoeff = 0.01f;
	const float c_allowedDiffCoeffThreshold = 0.00001f;
	for (uint i = 0; i < inputDataCount; ++i)
	{
		float diffLikelihoods = fabs(outputLayerLogLikelihoods[i] - mockOutputLayerLogLikelihoods[i]);
		if (diffLikelihoods > c_allowedDiff || (diffLikelihoods > c_allowedDiffCoeffThreshold &&
			diffLikelihoods > c_allowedDiffCoeff * max(outputLayerLogLikelihoods[i], mockOutputLayerLogLikelihoods[i])))
		{
			foundDifferentLogLikelihoods = true;
			firstDifference = outputLayerLogLikelihoods[i] - mockOutputLayerLogLikelihoods[i];
			break;
		}
		float diffScores = fabs(outputLayerScores[i] - mockOutputLayerScores[i]);
		if (diffScores > c_allowedDiff || (diffScores > c_allowedDiffCoeffThreshold &&
			diffScores > c_allowedDiffCoeff * max(outputLayerScores[i], mockOutputLayerScores[i])))
		{
			foundDifferentScores = true;
			firstDifference = outputLayerScores[i] - mockOutputLayerScores[i];
			break;
		}
		float diffMultipleGuessScores = fabs(outputLayerMultipleGuessScores[i] - mockOutputLayerMultipleGuessScores[i]);
		if (diffMultipleGuessScores > c_allowedDiff || (diffMultipleGuessScores > c_allowedDiffCoeffThreshold &&
			diffMultipleGuessScores > c_allowedDiffCoeff * max(outputLayerMultipleGuessScores[i], mockOutputLayerMultipleGuessScores[i])))
		{
			foundDifferentMultipleGuessScores = true;
			firstDifference = outputLayerMultipleGuessScores[i] - mockOutputLayerMultipleGuessScores[i];
			break;
		}
	}

	TestingAssert(!foundDifferentLogLikelihoods, "Found different log-likelihoods! Input data size: " + to_string(inputDataSize) +
		"; Input data count: " + to_string(inputDataCount) + "; First difference: " + to_string(firstDifference));
	TestingAssert(!foundDifferentScores, "Found different scores! Input data size: " + to_string(inputDataSize) +
		"; Input data count: " + to_string(inputDataCount) + "; First difference: " + to_string(firstDifference));
	TestingAssert(!foundDifferentMultipleGuessScores, "Found different multiple guess scores! Input data size: " + to_string(inputDataSize) +
		"; Input data count: " + to_string(inputDataCount) + "; First difference: " + to_string(firstDifference));

	float mockLoss = mockOutputLayer.GetLoss() / inputDataCount;
	float regLoss = outputLayer.GetLoss() / inputDataCount;
	float diffLoss = fabs(mockLoss - regLoss);
	TestingAssert(diffLoss <= c_allowedDiff && diffLoss <= c_allowedDiffCoeff * max(mockLoss, regLoss), "Calculated different losses! Input data size: " +
		to_string(inputDataSize) + "; Input data count: " + to_string(inputDataCount) + "; Mock loss: " + to_string(mockLoss) + "; Regular loss: " +
		to_string(regLoss));
	float mockAccuracy = mockOutputLayer.GetAccuracy() / inputDataCount;
	float regAccuracy = outputLayer.GetAccuracy() / inputDataCount;
	float diffAccuracy = fabs(mockAccuracy - regAccuracy);
	TestingAssert(diffAccuracy <= c_allowedDiff && diffAccuracy <= c_allowedDiffCoeff * max(mockAccuracy, regAccuracy),
		"Calculated different accuracies! Input data size: " + to_string(inputDataSize) + "; Input data count: " + to_string(inputDataCount) +
		"; Mock accuracy: " + to_string(mockAccuracy) + "; Regular accuracy: " + to_string(regAccuracy));
	float mockMultipleGuessAccuracy = mockOutputLayer.GetMultipleGuessAccuracy() / inputDataCount;
	float regMultipleGuessAccuracy = outputLayer.GetMultipleGuessAccuracy() / inputDataCount;
	float diffMultipleGuessAccuracy = fabs(mockMultipleGuessAccuracy - regMultipleGuessAccuracy);
	TestingAssert(diffMultipleGuessAccuracy <= c_allowedDiff && diffMultipleGuessAccuracy <= c_allowedDiffCoeff * max(mockMultipleGuessAccuracy, regMultipleGuessAccuracy),
		"Calculated different multiple guess accuracies! Input data size: " + to_string(inputDataSize) + "; Input data count: " + to_string(inputDataCount) +
		"; Mock multiple guess accuracy: " + to_string(mockMultipleGuessAccuracy) + "; Regular multiple guess accuracy: " + to_string(regMultipleGuessAccuracy));

	cout << "Forward prop passed. Input data size: " << inputDataSize << "; Input data count: " << inputDataCount << endl;
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

void TestOutputLayer::TestDoForwardProp()
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