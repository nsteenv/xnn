// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural networks configuration parser.
// Created: 03/17/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/configurationparser.cuh"

NeuralNet* ConfigurationParser::ParseNetworkFromConfiguration(ParsingMode parsingMode, string configurationFile, string dataFolder, uint batchSize,
	bool initializeLayersParams)
{
	m_parsingMode = parsingMode;
	m_dataFolder = dataFolder;
	m_batchSize = batchSize;
	m_initializeLayersParams = initializeLayersParams;
	m_layersTiers.clear();
	m_tiersLines.clear();

	ParseTierLines(configurationFile);
	FindMaxNetworkTierSize();

	m_neuralNet = new NeuralNet(m_maxNetworkTierSize);

	ParseLayersTiers();
	
	// Reverting back to default device.
	CudaAssert(cudaSetDevice(0));

	for (vector<Layer*>& layersTier: m_layersTiers)
	{
		m_neuralNet->AddLayersTier(layersTier);
	}
	
	return m_neuralNet;
}

string ConfigurationParser::TrimLine(string line)
{
	if (line == "")
	{
		return line;
	}

	string trimmedLine;

	// Trim leading whitespace.
	size_t firstNonWs = line.find_first_not_of(" \t");
	if (firstNonWs != string::npos)
	{
		trimmedLine = line.substr(firstNonWs);
	}

	// Trim trailing whitespace.
	size_t lastNonWs = trimmedLine.find_last_not_of(" \t");
	if (lastNonWs != string::npos)
	{
		trimmedLine = trimmedLine.substr(0, lastNonWs + 1);
	}

	return trimmedLine;
}

void ConfigurationParser::ParseTierLines(string configurationFile)
{
	ifstream configuration(configurationFile);
	string line;
	vector<string> currTierLines;
	bool encounteredFirstLayer = false;
	while (getline(configuration, line))
	{
		string trimmedLine = TrimLine(line);

		if (trimmedLine.find("layer:") == 0)
		{
			if (!currTierLines.empty() && encounteredFirstLayer)
			{
				m_tiersLines.push_back(currTierLines);
			}

			encounteredFirstLayer = true;

			currTierLines.clear();
			currTierLines.push_back(trimmedLine);
		}
		else if (trimmedLine != "")
		{
			currTierLines.push_back(trimmedLine);
		}
	}

	if (!currTierLines.empty() && encounteredFirstLayer)
	{
		m_tiersLines.push_back(currTierLines);
	}
}

bool ConfigurationParser::ParseParameterUint(string line, string parameterName, uint& parameterValue)
{
	if (line.find(parameterName) != 0)
	{
		return false;
	}

	size_t valuePosition = line.find_last_of(":");
	ShipAssert(valuePosition != string::npos, "Can't parse parameter: " + parameterName + " from line: " + line);

	string lineValue = line.substr(valuePosition + 1);
	parameterValue = stoi(TrimLine(lineValue));

	return true;
}

bool ConfigurationParser::ParseParameterFloat(string line, string parameterName, float& parameterValue)
{
	if (line.find(parameterName) != 0)
	{
		return false;
	}

	size_t valuePosition = line.find_last_of(":");
	ShipAssert(valuePosition != string::npos, "Can't parse parameter: " + parameterName + " from line: " + line);

	string lineValue = line.substr(valuePosition + 1);
	parameterValue = stof(TrimLine(lineValue));

	return true;
}

bool ConfigurationParser::ParseParameterString(string line, string parameterName, string& parameterValue)
{
	if (line.find(parameterName) != 0)
	{
		return false;
	}

	size_t valuePosition = line.find_last_of(":");
	ShipAssert(valuePosition != string::npos, "Can't parse parameter: " + parameterName + " from line: " + line);

	string lineValue = line.substr(valuePosition + 1);
	parameterValue = ConvertToLowercase(TrimLine(lineValue));

	return true;
}

void ConfigurationParser::FindMaxNetworkTierSize()
{
	m_maxNetworkTierSize = 1;
	
	if (m_parsingMode == ParsingMode::Training)
	{
		for (vector<string>& tierLines : m_tiersLines)
		{
			for (string& line : tierLines)
			{
				uint tierSize = 1;
				ParseParameterUint(line, "tierSize", tierSize);
				m_maxNetworkTierSize = max((uint)m_maxNetworkTierSize, tierSize);
			}
		}
	}
}

LayerType ConfigurationParser::GetLayerType(string layerTypeName)
{
	if (layerTypeName == "input")
	{
		return LayerType::Input;
	}
	else if (layerTypeName == "convolutional")
	{
		return LayerType::Convolutional;
	}
	else if (layerTypeName == "responsenormalization")
	{
		return LayerType::ResponseNormalization;
	}
	else if (layerTypeName == "maxpool")
	{
		return LayerType::MaxPool;
	}
	else if (layerTypeName == "standard")
	{
		return LayerType::Standard;
	}
	else if (layerTypeName == "dropout")
	{
		return LayerType::Dropout;
	}
	else if (layerTypeName == "softmax")
	{
		return LayerType::SoftMax;
	}
	else if (layerTypeName == "output")
	{
		return LayerType::Output;
	}
	else
	{
		ShipAssert(false, "Unknown layer type name: " + layerTypeName);
		return LayerType::Standard;
	}
}

ActivationType ConfigurationParser::GetActivationType(string activationTypeName)
{
	if (activationTypeName == "linear")
	{
		return ActivationType::Linear;
	}
	else if (activationTypeName == "relu")
	{
		return ActivationType::ReLu;
	}
	else if (activationTypeName == "sigmoid")
	{
		return ActivationType::Sigmoid;
	}
	else if (activationTypeName == "tanh")
	{
		return ActivationType::Tanh;
	}
	else
	{
		ShipAssert(false, "Unknown activation type name: " + activationTypeName);
		return ActivationType::Linear;
	}
}

LossFunctionType ConfigurationParser::GetLossFunctionType(string lossFunctionName)
{
	if (lossFunctionName == "logisticregression")
	{
		return LossFunctionType::LogisticRegression;
	}
	else
	{
		ShipAssert(false, "Unknown loss function name: " + lossFunctionName);
		return LossFunctionType::LogisticRegression;
	}
}

DataType ConfigurationParser::GetDataType(string dataTypeName)
{
	if (dataTypeName == "image")
	{
		return DataType::Image;
	}
	else if (dataTypeName == "text")
	{
		return DataType::Text;
	}
	else
	{
		ShipAssert(false, "Unknown data type name: " + dataTypeName);
		return DataType::Text;
	}
}

void ConfigurationParser::ParseLayersTiers()
{
	for (size_t tierIndex = 0; tierIndex < m_tiersLines.size(); ++tierIndex)
	{
		vector<string>& tierLines = m_tiersLines[tierIndex];
		string layerTypeName;
		ParseParameterString(tierLines[0], "layer", layerTypeName);
		LayerType tierLayerType = GetLayerType(layerTypeName);

		if (tierIndex == 0)
		{			
			ShipAssert(tierLayerType == LayerType::Input, "First layer in the network should be input layer!");
		}
		else if (tierIndex == m_tiersLines.size() - 1)
		{
			ShipAssert(tierLayerType == LayerType::Output, "Last layer in the network should be output layer!");
		}

		vector<Layer*> layerTier = ParseLayersTier(tierIndex, tierLayerType);

		m_layersTiers.push_back(layerTier);
	}
}

vector<Layer*> ConfigurationParser::FindPrevLayers(ParallelismMode currTierParallelismMode, uint layerIndex, uint currTierSize, size_t prevTierIndex, string prevLayersParam)
{
	vector<Layer*> prevLayers;

	if (m_parsingMode == ParsingMode::Prediction || (prevLayersParam != "all" && currTierParallelismMode == m_layersTiers[prevTierIndex][0]->GetParallelismMode() &&
		currTierSize == m_layersTiers[prevTierIndex].size()))
	{
		prevLayers.push_back(m_layersTiers[prevTierIndex][layerIndex]);
	}
	else
	{
		prevLayers = m_layersTiers[prevTierIndex];
	}

	return prevLayers;
}

bool ConfigurationParser::ShouldHoldActivationGradients(ParallelismMode currTierParallelismMode, uint currTierSize, size_t currTierIndex, uint layerIndex)
{
	if (m_parsingMode == ParsingMode::Prediction || currTierIndex == m_tiersLines.size() - 1)
	{
		return false;
	}

	uint nextTierSize = 1;
	bool parsedTierSize = false;
	string parallelismValue;
	ParallelismMode nextTierParallelismMode = ParallelismMode::Model;
	bool parsedParallelism = false;
	string prevLayersParam;
	bool parsedPrevLayers = false;
	vector<string>& tierLines = m_tiersLines[currTierIndex + 1];
	for (string& line : tierLines)
	{
		parsedTierSize = parsedTierSize || ParseParameterUint(line, "tierSize", nextTierSize);
		parsedParallelism = parsedParallelism || ParseParameterString(line, "parallelism", parallelismValue);
		parsedPrevLayers = parsedPrevLayers || ParseParameterString(line, "prevLayers", prevLayersParam);
	}
	if (parsedParallelism)
	{
		nextTierParallelismMode = parallelismValue == "data" ? ParallelismMode::Data : ParallelismMode::Model;
	}

	if ((nextTierSize == 1 && layerIndex == 0) || (prevLayersParam != "all" && currTierParallelismMode == nextTierParallelismMode && currTierSize == nextTierSize))
	{
		return false;
	}
	else
	{
		return true;
	}
}

void ConfigurationParser::FindInputParams(vector<Layer*>& prevLayers, uint layerIndex, uint tierSize, ParallelismMode parallelismMode, uint& inputNumChannels,
	uint& inputDataWidth, uint& inputDataHeight, uint& inputDataCount, bool& holdsInputData)
{
	if (prevLayers[0]->GetLayerType() == LayerType::Input)
	{
		InputLayer* inputLayer = static_cast<InputLayer*>(prevLayers[0]);
		inputNumChannels = inputLayer->GetActivationNumChannels();
		inputDataWidth = inputLayer->GetActivationDataWidth();
		inputDataHeight = inputLayer->GetActivationDataHeight();
		if (parallelismMode == ParallelismMode::Data)
		{
			inputDataCount = inputLayer->GetInputDataCount() / tierSize;
		}
		else
		{
			inputDataCount = inputLayer->GetInputDataCount();
		}
	}
	else if (prevLayers[0]->GetParallelismMode() == ParallelismMode::Data)
	{
		inputNumChannels = prevLayers[0]->GetActivationNumChannels();
		inputDataWidth = prevLayers[0]->GetActivationDataWidth();
		inputDataHeight = prevLayers[0]->GetActivationDataHeight();
		inputDataCount = prevLayers[0]->GetActivationDataCount();
	}
	else if (prevLayers[0]->GetLayerType() == LayerType::Convolutional || prevLayers[0]->GetLayerType() == LayerType::ResponseNormalization ||
		prevLayers[0]->GetLayerType() == LayerType::MaxPool)
	{
		inputNumChannels = prevLayers[0]->GetActivationNumChannels();
		for (size_t i = 1; i < prevLayers.size(); ++i)
		{
			inputNumChannels += prevLayers[i]->GetActivationNumChannels();
		}
		inputDataWidth = prevLayers[0]->GetActivationDataWidth();
		inputDataHeight = prevLayers[0]->GetActivationDataHeight();
		inputDataCount = prevLayers[0]->GetActivationDataCount();
	}
	else
	{
		inputNumChannels = prevLayers[0]->GetActivationNumChannels();
		inputDataWidth = prevLayers[0]->GetActivationDataWidth();
		for (size_t i = 1; i < prevLayers.size(); ++i)
		{
			inputDataWidth += prevLayers[i]->GetActivationDataWidth();
		}
		inputDataHeight = prevLayers[0]->GetActivationDataHeight();
		inputDataCount = prevLayers[0]->GetActivationDataCount();
	}

	holdsInputData = prevLayers.size() > 1 || (prevLayers[0]->GetLayerType() != LayerType::Input && prevLayers[0]->GetIndexInTier() != layerIndex);
}

void ConfigurationParser::ParseInputLayerTier(vector<Layer*>& outLayerTier)
{
	string dataTypeValue;
	DataType dataType;
	bool parsedDataType = false;
	uint numChannels;
	bool parsedNumChannels = false;
	uint inputDataWidth;
	bool parsedInputDataWidth = false;
	uint inputDataHeight;
	bool parsedInputDataHeight = false;
	uint trainDataWidth = 0;
	bool parsedTrainDataWidth = false;
	uint trainDataHeight = 0;
	bool parsedTrainDataHeight = false;
	uint numTestPatches;
	bool parsedNumTestPatches = false;
	bool testOnFlips;
	string testOnFlipsValue;
	bool parsedTestOnFlips = false;

	vector<string>& tierLines = m_tiersLines[0];
	for (string& line : tierLines)
	{
		parsedDataType = parsedDataType || ParseParameterString(line, "data", dataTypeValue);
		parsedNumChannels = parsedNumChannels || ParseParameterUint(line, "numChannels", numChannels);
		parsedInputDataWidth = parsedInputDataWidth || ParseParameterUint(line, "inputDataWidth", inputDataWidth);
		parsedInputDataHeight = parsedInputDataHeight || ParseParameterUint(line, "inputDataHeight", inputDataHeight);
		parsedTrainDataWidth = parsedTrainDataWidth || ParseParameterUint(line, "trainDataWidth", trainDataWidth);
		parsedTrainDataHeight = parsedTrainDataHeight || ParseParameterUint(line, "trainDataHeight", trainDataHeight);
		parsedNumTestPatches = parsedNumTestPatches || ParseParameterUint(line, "numTestPatches", numTestPatches);
		parsedTestOnFlips = parsedTestOnFlips || ParseParameterString(line, "testOnFlips", testOnFlipsValue);
	}

	ShipAssert(parsedDataType, "Can't parse data type for Input layer!");
	ShipAssert(parsedNumChannels, "Can't parse number of channels for Input layer!");
	ShipAssert(parsedInputDataWidth, "Can't parse input data width for Input layer!");
	ShipAssert(parsedInputDataHeight, "Can't parse input data height for Input layer!");
	if (dataTypeValue == "image")
	{
		ShipAssert(parsedTrainDataWidth, "Can't parse train data width for Input layer!");
		ShipAssert(parsedTrainDataHeight, "Can't parse train data height for Input layer!");
	}
	ShipAssert(parsedNumTestPatches, "Can't parse number of test patches for Input layer!");
	ShipAssert(parsedTestOnFlips, "Can't parse should we test on flips for Input layer!");

	testOnFlips = testOnFlipsValue == "yes" ? true : false;
	dataType = GetDataType(dataTypeValue);

	// Finding number of inputs.
	ShipAssert(m_tiersLines.size() > 1, "We need to have more than input layer to train network, you know...");
	uint nextTierSize = 1;
	bool parsedTierSize = false;
	string parallelismValue;
	ParallelismMode nextTierParallelismMode = ParallelismMode::Model;
	bool parsedParallelism = false;
	vector<string>& nextTierLines = m_tiersLines[1];
	for (string& line : nextTierLines)
	{
		parsedTierSize = parsedTierSize || ParseParameterUint(line, "tierSize", nextTierSize);
		parsedParallelism = parsedParallelism || ParseParameterString(line, "parallelism", parallelismValue);
	}
	if (parsedParallelism)
	{
		nextTierParallelismMode = parallelismValue == "data" ? ParallelismMode::Data : ParallelismMode::Model;
	}
	uint numInputs = nextTierParallelismMode == ParallelismMode::Data ? nextTierSize : 1;

	CudaAssert(cudaSetDevice(0));

	outLayerTier.push_back(new InputLayer(m_dataFolder, dataType, m_neuralNet->GetDeviceMemoryStreams(), numChannels, inputDataWidth, inputDataHeight,
		(uint)m_maxNetworkTierSize * m_batchSize, trainDataWidth, trainDataHeight, m_parsingMode == ParsingMode::Training ? numInputs : 1, numTestPatches, testOnFlips));
}

void ConfigurationParser::ParseConvolutionalLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier)
{
	uint tierSize;
	bool parsedTierSize = false;
	string parallelismValue;
	ParallelismMode parallelismMode;
	bool parsedParallelism = false;
	uint numFilters;
	bool parsedNumFilters = false;
	uint filterWidth;
	bool parsedFilterWidth = false;
	uint filterHeight;
	bool parsedFilterHeight = false;
	float weightsDeviation;
	bool parsedWeightsDeviation = false;
	float weightsMomentum;
	bool parsedWeightsMomentum = false;
	float weightsDecay;
	bool parsedWeightsDecay = false;
	float weightsStartingLR;
	bool parsedWeightsStartingLR = false;
	float weightsLRStep;
	bool parsedWeightsLRStep = false;
	float weightsLRFactor;
	bool parsedWeightsLRFactor = false;
	float biasesInitialValue;
	bool parsedBiasesInitialValue = false;
	float biasesMomentum;
	bool parsedBiasesMomentum = false;
	float biasesDecay;
	bool parsedBiasesDecay = false;
	float biasesStartingLR;
	bool parsedBiasesStartingLR = false;
	float biasesLRStep;
	bool parsedBiasesLRStep = false;
	float biasesLRFactor;
	bool parsedBiasesLRFactor = false;
	uint paddingX;
	bool parsedPaddingX = false;
	uint paddingY;
	bool parsedPaddingY = false;
	uint stride;
	bool parsedStride = false;
	string activationTypeValue;
	ActivationType activationType;
	bool parsedActivationType = false;
	string prevLayersParam;
	bool parsedPrevLayers = false;

	vector<string>& tierLines = m_tiersLines[tierIndex];
	for (string& line : tierLines)
	{
		parsedTierSize = parsedTierSize || ParseParameterUint(line, "tierSize", tierSize);
		parsedParallelism = parsedParallelism || ParseParameterString(line, "parallelism", parallelismValue);
		parsedNumFilters = parsedNumFilters || ParseParameterUint(line, "numFilters", numFilters);
		parsedFilterWidth = parsedFilterWidth || ParseParameterUint(line, "filterWidth", filterWidth);
		parsedFilterHeight = parsedFilterHeight || ParseParameterUint(line, "filterHeight", filterHeight);
		parsedWeightsDeviation = parsedWeightsDeviation || ParseParameterFloat(line, "weightsDeviation", weightsDeviation);
		parsedWeightsMomentum = parsedWeightsMomentum || ParseParameterFloat(line, "weightsMomentum", weightsMomentum);
		parsedWeightsDecay = parsedWeightsDecay || ParseParameterFloat(line, "weightsDecay", weightsDecay);
		parsedWeightsStartingLR = parsedWeightsStartingLR || ParseParameterFloat(line, "weightsStartingLR", weightsStartingLR);
		parsedWeightsLRStep = parsedWeightsLRStep || ParseParameterFloat(line, "weightsLRStep", weightsLRStep);
		parsedWeightsLRFactor = parsedWeightsLRFactor || ParseParameterFloat(line, "weightsLRFactor", weightsLRFactor);
		parsedBiasesInitialValue = parsedBiasesInitialValue || ParseParameterFloat(line, "biasesInitialValue", biasesInitialValue);
		parsedBiasesMomentum = parsedBiasesMomentum || ParseParameterFloat(line, "biasesMomentum", biasesMomentum);
		parsedBiasesDecay = parsedBiasesDecay || ParseParameterFloat(line, "biasesDecay", biasesDecay);
		parsedBiasesStartingLR = parsedBiasesStartingLR || ParseParameterFloat(line, "biasesStartingLR", biasesStartingLR);
		parsedBiasesLRStep = parsedBiasesLRStep || ParseParameterFloat(line, "biasesLRStep", biasesLRStep);
		parsedBiasesLRFactor = parsedBiasesLRFactor || ParseParameterFloat(line, "biasesLRFactor", biasesLRFactor);
		parsedPaddingX = parsedPaddingX || ParseParameterUint(line, "paddingX", paddingX);
		parsedPaddingY = parsedPaddingY || ParseParameterUint(line, "paddingY", paddingY);
		parsedStride = parsedStride || ParseParameterUint(line, "stride", stride);
		parsedActivationType = parsedActivationType || ParseParameterString(line, "activationType", activationTypeValue);
		parsedPrevLayers = parsedPrevLayers || ParseParameterString(line, "prevLayers", prevLayersParam);
	}

	ShipAssert(parsedTierSize, "Can't parse tier size for Convolutional layer!");
	ShipAssert(parsedParallelism, "Can't parse parallelism for Convolutional layer!");
	ShipAssert(parsedNumFilters, "Can't parse number of filters for Convolutional layer!");
	ShipAssert(parsedFilterWidth, "Can't parse filter width for Convolutional layer!");
	ShipAssert(parsedFilterHeight, "Can't parse filter height for Convolutional layer!");
	ShipAssert(parsedWeightsDeviation, "Can't parse weights deviation for Convolutional layer!");
	ShipAssert(parsedWeightsMomentum, "Can't parse weights momentum for Convolutional layer!");
	ShipAssert(parsedWeightsDecay, "Can't parse weights decay for Convolutional layer!");
	ShipAssert(parsedWeightsStartingLR, "Can't parse weights starting learning rate for Convolutional layer!");
	ShipAssert(parsedWeightsLRStep, "Can't parse weights learning rate step for Convolutional layer!");
	ShipAssert(parsedWeightsLRFactor, "Can't parse weights learning rate factor for Convolutional layer!");
	ShipAssert(parsedBiasesInitialValue, "Can't parse biases initial value for Convolutional layer!");
	ShipAssert(parsedBiasesMomentum, "Can't parse biases momentum for Convolutional layer!");
	ShipAssert(parsedBiasesDecay, "Can't parse biases decay for Convolutional layer!");
	ShipAssert(parsedBiasesStartingLR, "Can't parse biases starting learning rate for Convolutional layer!");
	ShipAssert(parsedBiasesLRStep, "Can't parse biases learning rate step for Convolutional layer!");
	ShipAssert(parsedBiasesLRFactor, "Can't parse biases learnign rate factor for Convolutional layer!");
	ShipAssert(parsedPaddingX, "Can't parse horizontal padding for Convolutional layer!");
	ShipAssert(parsedPaddingY, "Can't parse vertical padding for Convolutional layer!");
	ShipAssert(parsedStride, "Can't parse stride for Convolutional layer!");
	ShipAssert(parsedActivationType, "Can't parse activation type for Convolutional layer!");

	if (m_parsingMode == ParsingMode::Prediction)
	{
		tierSize = 1;
		parallelismMode = ParallelismMode::Model;
	}
	else
	{
		parallelismMode = parallelismValue == "data" ? ParallelismMode::Data : ParallelismMode::Model;
	}
	activationType = GetActivationType(activationTypeValue);

	for (uint layerIndex = 0; layerIndex < tierSize; ++layerIndex)
	{
		CudaAssert(cudaSetDevice(layerIndex));

		vector<Layer*> prevLayers = FindPrevLayers(parallelismMode, layerIndex, tierSize, tierIndex - 1, prevLayersParam);
		uint inputNumChannels;
		uint inputDataWidth;
		uint inputDataHeight;
		uint inputDataCount;
		bool holdsInputData;
		FindInputParams(prevLayers, layerIndex, tierSize, parallelismMode, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData);
		bool holdsActivationGradients = ShouldHoldActivationGradients(parallelismMode, tierSize, tierIndex, layerIndex);

		ConvolutionalLayer* convLayer = new ConvolutionalLayer(parallelismMode, m_neuralNet->GetDeviceCalculationStreams()[layerIndex],
			m_neuralNet->GetDeviceMemoryStreams()[layerIndex], layerIndex, tierSize, inputNumChannels, inputDataWidth, inputDataHeight,
			inputDataCount, holdsInputData, numFilters, filterWidth, filterHeight, inputNumChannels, m_initializeLayersParams, weightsDeviation,
			m_initializeLayersParams, biasesInitialValue, weightsMomentum, weightsDecay, weightsLRStep, weightsStartingLR, weightsLRFactor,
			biasesMomentum, biasesDecay, biasesLRStep, biasesStartingLR, biasesLRFactor, paddingX, paddingY, stride, activationType,
			holdsActivationGradients);

		for (Layer* prevLayer : prevLayers)
		{
			convLayer->AddPrevLayer(prevLayer);
			prevLayer->AddNextLayer(convLayer);
		}

		outLayerTier.push_back(convLayer);
	}
}

void ConfigurationParser::ParseResponseNormalizationLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier)
{
	uint tierSize;
	bool parsedTierSize = false;
	string parallelismValue;
	ParallelismMode parallelismMode;
	bool parsedParallelism = false;
	uint depth;
	bool parsedDepth = false;
	float bias;
	bool parsedBias = false;
	float alphaCoeff;
	bool parsedAlphaCoeff = false;
	float betaCoeff;
	bool parsedBetaCoeff = false;
	string prevLayersParam;
	bool parsedPrevLayers = false;

	vector<string>& tierLines = m_tiersLines[tierIndex];
	for (string& line : tierLines)
	{
		parsedTierSize = parsedTierSize || ParseParameterUint(line, "tierSize", tierSize);
		parsedParallelism = parsedParallelism || ParseParameterString(line, "parallelism", parallelismValue);
		parsedDepth = parsedDepth || ParseParameterUint(line, "depth", depth);
		parsedBias = parsedBias || ParseParameterFloat(line, "bias", bias);
		parsedAlphaCoeff = parsedAlphaCoeff || ParseParameterFloat(line, "alphaCoeff", alphaCoeff);
		parsedBetaCoeff = parsedBetaCoeff || ParseParameterFloat(line, "betaCoeff", betaCoeff);
		parsedPrevLayers = parsedPrevLayers || ParseParameterString(line, "prevLayers", prevLayersParam);
	}

	ShipAssert(parsedTierSize, "Can't parse tier size for Response Normalization layer!");
	ShipAssert(parsedParallelism, "Can't parse parallelism for Response Normalization layer!");
	ShipAssert(parsedDepth, "Can't parse depth for Response Normalization layer!");
	ShipAssert(parsedBias, "Can't parse bias for Response Normalization layer!");
	ShipAssert(parsedAlphaCoeff, "Can't parse alpha coefficient for Response Normalization layer!");
	ShipAssert(parsedBetaCoeff, "Can't parse beta coefficient for Response Normalization layer!");

	if (m_parsingMode == ParsingMode::Prediction)
	{
		tierSize = 1;
		parallelismMode = ParallelismMode::Model;
	}
	else
	{
		parallelismMode = parallelismValue == "data" ? ParallelismMode::Data : ParallelismMode::Model;
	}

	for (uint layerIndex = 0; layerIndex < tierSize; ++layerIndex)
	{
		CudaAssert(cudaSetDevice(layerIndex));

		vector<Layer*> prevLayers = FindPrevLayers(parallelismMode, layerIndex, tierSize, tierIndex - 1, prevLayersParam);
		uint inputNumChannels;
		uint inputDataWidth;
		uint inputDataHeight;
		uint inputDataCount;
		bool holdsInputData;
		FindInputParams(prevLayers, layerIndex, tierSize, parallelismMode, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData);
		bool holdsActivationGradients = ShouldHoldActivationGradients(parallelismMode, tierSize, tierIndex, layerIndex);

		ResponseNormalizationLayer* reNormLayer = new ResponseNormalizationLayer(parallelismMode, m_neuralNet->GetDeviceCalculationStreams()[layerIndex],
			m_neuralNet->GetDeviceMemoryStreams()[layerIndex], layerIndex, tierSize, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount,
			holdsInputData, depth, bias, alphaCoeff, betaCoeff, holdsActivationGradients);

		for (Layer* prevLayer : prevLayers)
		{
			reNormLayer->AddPrevLayer(prevLayer);
			prevLayer->AddNextLayer(reNormLayer);
		}

		outLayerTier.push_back(reNormLayer);
	}
}

void ConfigurationParser::ParseMaxPoolLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier)
{
	uint tierSize;
	bool parsedTierSize = false;
	string parallelismValue;
	ParallelismMode parallelismMode;
	bool parsedParallelism = false;
	uint unitWidth;
	bool parsedUnitWidth = false;
	uint unitHeight;
	bool parsedUnitHeight = false;
	uint paddingX;
	bool parsedPaddingX = false;
	uint paddingY;
	bool parsedPaddingY = false;
	uint unitStride;
	bool parsedUnitStride = false;
	string prevLayersParam;
	bool parsedPrevLayers = false;

	vector<string>& tierLines = m_tiersLines[tierIndex];
	for (string& line : tierLines)
	{
		parsedTierSize = parsedTierSize || ParseParameterUint(line, "tierSize", tierSize);
		parsedParallelism = parsedParallelism || ParseParameterString(line, "parallelism", parallelismValue);
		parsedUnitWidth = parsedUnitWidth || ParseParameterUint(line, "unitWidth", unitWidth);
		parsedUnitHeight = parsedUnitHeight || ParseParameterUint(line, "unitHeight", unitHeight);
		parsedPaddingX = parsedPaddingX || ParseParameterUint(line, "paddingX", paddingX);
		parsedPaddingY = parsedPaddingY || ParseParameterUint(line, "paddingY", paddingY);
		parsedUnitStride = parsedUnitStride || ParseParameterUint(line, "unitStride", unitStride);
		parsedPrevLayers = parsedPrevLayers || ParseParameterString(line, "prevLayers", prevLayersParam);
	}

	ShipAssert(parsedTierSize, "Can't parse tier size for Max Pool layer!");
	ShipAssert(parsedParallelism, "Can't parse parallelism for Max Pool layer!");
	ShipAssert(parsedUnitWidth, "Can't parse unit width for Max Pool layer!");
	ShipAssert(parsedUnitHeight, "Can't parse unit height for Max Pool layer!");
	ShipAssert(parsedPaddingX, "Can't parse padding X for Max Pool layer!");
	ShipAssert(parsedPaddingY, "Can't parse padding Y for Max Pool layer!");
	ShipAssert(parsedUnitStride, "Can't parse unit stride for Max Pool layer!");

	if (m_parsingMode == ParsingMode::Prediction)
	{
		tierSize = 1;
		parallelismMode = ParallelismMode::Model;
	}
	else
	{
		parallelismMode = parallelismValue == "data" ? ParallelismMode::Data : ParallelismMode::Model;
	}

	for (uint layerIndex = 0; layerIndex < tierSize; ++layerIndex)
	{
		CudaAssert(cudaSetDevice(layerIndex));

		vector<Layer*> prevLayers = FindPrevLayers(parallelismMode, layerIndex, tierSize, tierIndex - 1, prevLayersParam);
		uint inputNumChannels;
		uint inputDataWidth;
		uint inputDataHeight;
		uint inputDataCount;
		bool holdsInputData;
		FindInputParams(prevLayers, layerIndex, tierSize, parallelismMode, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData);
		bool holdsActivationGradients = ShouldHoldActivationGradients(parallelismMode, tierSize, tierIndex, layerIndex);

		MaxPoolLayer* maxPoolLayer = new MaxPoolLayer(parallelismMode, m_neuralNet->GetDeviceCalculationStreams()[layerIndex],
			m_neuralNet->GetDeviceMemoryStreams()[layerIndex], layerIndex, tierSize, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData,
			unitWidth, unitHeight, paddingX, paddingY, unitStride, holdsActivationGradients);

		for (Layer* prevLayer : prevLayers)
		{
			maxPoolLayer->AddPrevLayer(prevLayer);
			prevLayer->AddNextLayer(maxPoolLayer);
		}

		outLayerTier.push_back(maxPoolLayer);
	}
}

void ConfigurationParser::ParseStandardLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier)
{
	uint tierSize;
	bool parsedTierSize = false;
	string parallelismValue;
	ParallelismMode parallelismMode;
	bool parsedParallelism = false;
	uint numNeurons;
	bool parsedNumNeurons = false;
	float weightsDeviation;
	bool parsedWeightsDeviation = false;
	float weightsMomentum;
	bool parsedWeightsMomentum = false;
	float weightsDecay;
	bool parsedWeightsDecay = false;
	float weightsStartingLR;
	bool parsedWeightsStartingLR = false;
	float weightsLRStep;
	bool parsedWeightsLRStep = false;
	float weightsLRFactor;
	bool parsedWeightsLRFactor = false;
	float biasesInitialValue;
	bool parsedBiasesInitialValue = false;
	float biasesMomentum;
	bool parsedBiasesMomentum = false;
	float biasesDecay;
	bool parsedBiasesDecay = false;
	float biasesStartingLR;
	bool parsedBiasesStartingLR = false;
	float biasesLRStep;
	bool parsedBiasesLRStep = false;
	float biasesLRFactor;
	bool parsedBiasesLRFactor = false;
	string activationTypeValue;
	ActivationType activationType;
	bool parsedActivationType = false;
	string prevLayersParam;
	bool parsedPrevLayers = false;

	vector<string>& tierLines = m_tiersLines[tierIndex];
	for (string& line : tierLines)
	{
		parsedTierSize = parsedTierSize || ParseParameterUint(line, "tierSize", tierSize);
		parsedParallelism = parsedParallelism || ParseParameterString(line, "parallelism", parallelismValue);
		parsedNumNeurons = parsedNumNeurons || ParseParameterUint(line, "numNeurons", numNeurons);
		parsedWeightsDeviation = parsedWeightsDeviation || ParseParameterFloat(line, "weightsDeviation", weightsDeviation);
		parsedWeightsMomentum = parsedWeightsMomentum || ParseParameterFloat(line, "weightsMomentum", weightsMomentum);
		parsedWeightsDecay = parsedWeightsDecay || ParseParameterFloat(line, "weightsDecay", weightsDecay);
		parsedWeightsStartingLR = parsedWeightsStartingLR || ParseParameterFloat(line, "weightsStartingLR", weightsStartingLR);
		parsedWeightsLRStep = parsedWeightsLRStep || ParseParameterFloat(line, "weightsLRStep", weightsLRStep);
		parsedWeightsLRFactor = parsedWeightsLRFactor || ParseParameterFloat(line, "weightsLRFactor", weightsLRFactor);
		parsedBiasesInitialValue = parsedBiasesInitialValue || ParseParameterFloat(line, "biasesInitialValue", biasesInitialValue);
		parsedBiasesMomentum = parsedBiasesMomentum || ParseParameterFloat(line, "biasesMomentum", biasesMomentum);
		parsedBiasesDecay = parsedBiasesDecay || ParseParameterFloat(line, "biasesDecay", biasesDecay);
		parsedBiasesStartingLR = parsedBiasesStartingLR || ParseParameterFloat(line, "biasesStartingLR", biasesStartingLR);
		parsedBiasesLRStep = parsedBiasesLRStep || ParseParameterFloat(line, "biasesLRStep", biasesLRStep);
		parsedBiasesLRFactor = parsedBiasesLRFactor || ParseParameterFloat(line, "biasesLRFactor", biasesLRFactor);
		parsedActivationType = parsedActivationType || ParseParameterString(line, "activationType", activationTypeValue);
		parsedPrevLayers = parsedPrevLayers || ParseParameterString(line, "prevLayers", prevLayersParam);
	}

	ShipAssert(parsedTierSize, "Can't parse tier size for Standard layer!");
	ShipAssert(parsedParallelism, "Can't parse parallelism for Standard layer!");
	ShipAssert(parsedNumNeurons, "Can't parse number of neurons for Standard layer!");
	ShipAssert(parsedWeightsDeviation, "Can't parse weights deviation for Standard layer!");
	ShipAssert(parsedWeightsMomentum, "Can't parse weights momentum for Standard layer!");
	ShipAssert(parsedWeightsDecay, "Can't parse weights decay for Standard layer!");
	ShipAssert(parsedWeightsStartingLR, "Can't parse weights starting learning rate for Standard layer!");
	ShipAssert(parsedWeightsLRStep, "Can't parse weights learning rate step for Standard layer!");
	ShipAssert(parsedWeightsLRFactor, "Can't parse weights learning rate factor for Standard layer!");
	ShipAssert(parsedBiasesInitialValue, "Can't parse biases initial value for Standard layer!");
	ShipAssert(parsedBiasesMomentum, "Can't parse biases momentum for Standard layer!");
	ShipAssert(parsedBiasesDecay, "Can't parse biases decay for Standard layer!");
	ShipAssert(parsedBiasesStartingLR, "Can't parse biases starting learning rate for Standard layer!");
	ShipAssert(parsedBiasesLRStep, "Can't parse biases learning rate step for Standard layer!");
	ShipAssert(parsedBiasesLRFactor, "Can't parse biases learnign rate factor for Standard layer!");
	ShipAssert(parsedActivationType, "Can't parse activation type for Standard layer!");

	if (m_parsingMode == ParsingMode::Prediction)
	{
		tierSize = 1;
		parallelismMode = ParallelismMode::Model;
	}
	else
	{
		parallelismMode = parallelismValue == "data" ? ParallelismMode::Data : ParallelismMode::Model;
	}
	activationType = GetActivationType(activationTypeValue);

	for (uint layerIndex = 0; layerIndex < tierSize; ++layerIndex)
	{
		CudaAssert(cudaSetDevice(layerIndex));

		vector<Layer*> prevLayers = FindPrevLayers(parallelismMode, layerIndex, tierSize, tierIndex - 1, prevLayersParam);
		uint inputNumChannels;
		uint inputDataWidth;
		uint inputDataHeight;
		uint inputDataCount;
		bool holdsInputData;
		FindInputParams(prevLayers, layerIndex, tierSize, parallelismMode, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData);
		bool holdsActivationGradients = ShouldHoldActivationGradients(parallelismMode, tierSize, tierIndex, layerIndex);

		StandardLayer* standardLayer = new StandardLayer(parallelismMode, m_neuralNet->GetDeviceCalculationStreams()[layerIndex],
			m_neuralNet->GetDeviceMemoryStreams()[layerIndex], m_neuralNet->GetCublasHandles()[layerIndex], layerIndex, tierSize, inputNumChannels,
			inputDataWidth, inputDataHeight, inputDataCount, holdsInputData, numNeurons, m_initializeLayersParams, weightsDeviation,
			m_initializeLayersParams, biasesInitialValue, weightsMomentum, weightsDecay, weightsLRStep, weightsStartingLR, weightsLRFactor,
			biasesMomentum, biasesDecay, biasesLRStep, biasesStartingLR, biasesLRFactor, activationType, holdsActivationGradients);

		for (Layer* prevLayer : prevLayers)
		{
			standardLayer->AddPrevLayer(prevLayer);
			prevLayer->AddNextLayer(standardLayer);
		}

		outLayerTier.push_back(standardLayer);
	}
}

void ConfigurationParser::ParseDropoutLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier)
{
	uint tierSize;
	bool parsedTierSize = false;
	string parallelismValue;
	ParallelismMode parallelismMode;
	bool parsedParallelism = false;
	float dropProbability;
	bool parsedDropProbability = false;
	string prevLayersParam;
	bool parsedPrevLayers = false;

	vector<string>& tierLines = m_tiersLines[tierIndex];
	for (string& line : tierLines)
	{
		parsedTierSize = parsedTierSize || ParseParameterUint(line, "tierSize", tierSize);
		parsedParallelism = parsedParallelism || ParseParameterString(line, "parallelism", parallelismValue);
		parsedDropProbability = parsedDropProbability || ParseParameterFloat(line, "dropProbability", dropProbability);
		parsedPrevLayers = parsedPrevLayers || ParseParameterString(line, "prevLayers", prevLayersParam);
	}

	ShipAssert(parsedTierSize, "Can't parse tier size for Dropout layer!");
	ShipAssert(parsedParallelism, "Can't parse parallelism for Dropout layer!");
	ShipAssert(parsedDropProbability, "Can't parse drop probability for Dropout layer!");

	if (m_parsingMode == ParsingMode::Prediction)
	{
		tierSize = 1;
		parallelismMode = ParallelismMode::Model;
	}
	else
	{
		parallelismMode = parallelismValue == "data" ? ParallelismMode::Data : ParallelismMode::Model;
	}

	for (uint layerIndex = 0; layerIndex < tierSize; ++layerIndex)
	{
		CudaAssert(cudaSetDevice(layerIndex));

		vector<Layer*> prevLayers = FindPrevLayers(parallelismMode, layerIndex, tierSize, tierIndex - 1, prevLayersParam);
		uint inputNumChannels;
		uint inputDataWidth;
		uint inputDataHeight;
		uint inputDataCount;
		bool holdsInputData;
		FindInputParams(prevLayers, layerIndex, tierSize, parallelismMode, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData);
		bool holdsActivationGradients = ShouldHoldActivationGradients(parallelismMode, tierSize, tierIndex, layerIndex);

		DropoutLayer* dropoutLayer = new DropoutLayer(parallelismMode, m_neuralNet->GetDeviceCalculationStreams()[layerIndex],
			m_neuralNet->GetDeviceMemoryStreams()[layerIndex], m_neuralNet->GetCurandStatesBuffers()[layerIndex], layerIndex, tierSize, inputNumChannels,
			inputDataWidth, inputDataHeight, inputDataCount, holdsInputData, dropProbability, false, holdsActivationGradients);

		for (Layer* prevLayer : prevLayers)
		{
			dropoutLayer->AddPrevLayer(prevLayer);
			prevLayer->AddNextLayer(dropoutLayer);
		}

		outLayerTier.push_back(dropoutLayer);
	}
}

void ConfigurationParser::ParseSoftMaxLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier)
{
	uint tierSize;
	bool parsedTierSize = false;
	string parallelismValue;
	ParallelismMode parallelismMode;
	bool parsedParallelism = false;
	string prevLayersParam;
	bool parsedPrevLayers = false;

	vector<string>& tierLines = m_tiersLines[tierIndex];
	for (string& line : tierLines)
	{
		parsedTierSize = parsedTierSize || ParseParameterUint(line, "tierSize", tierSize);
		parsedParallelism = parsedParallelism || ParseParameterString(line, "parallelism", parallelismValue);
		parsedPrevLayers = parsedPrevLayers || ParseParameterString(line, "prevLayers", prevLayersParam);
	}

	ShipAssert(parsedTierSize, "Can't parse tier size for SoftMax layer!");
	ShipAssert(parsedParallelism, "Can't parse parallelism for SoftMax layer!");

	if (m_parsingMode == ParsingMode::Prediction)
	{
		tierSize = 1;
		parallelismMode = ParallelismMode::Model;
	}
	else
	{
		parallelismMode = parallelismValue == "data" ? ParallelismMode::Data : ParallelismMode::Model;
	}

	for (uint layerIndex = 0; layerIndex < tierSize; ++layerIndex)
	{
		CudaAssert(cudaSetDevice(layerIndex));

		vector<Layer*> prevLayers = FindPrevLayers(parallelismMode, layerIndex, tierSize, tierIndex - 1, prevLayersParam);
		uint inputNumChannels;
		uint inputDataWidth;
		uint inputDataHeight;
		uint inputDataCount;
		bool holdsInputData;
		FindInputParams(prevLayers, layerIndex, tierSize, parallelismMode, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData);
		bool holdsActivationGradients = ShouldHoldActivationGradients(parallelismMode, tierSize, tierIndex, layerIndex);

		SoftMaxLayer* softMaxLayer = new SoftMaxLayer(parallelismMode, m_neuralNet->GetDeviceCalculationStreams()[layerIndex],
			m_neuralNet->GetDeviceMemoryStreams()[layerIndex], inputNumChannels * inputDataWidth * inputDataHeight, inputDataCount, holdsInputData);

		for (Layer* prevLayer : prevLayers)
		{
			softMaxLayer->AddPrevLayer(prevLayer);
			prevLayer->AddNextLayer(softMaxLayer);
		}

		outLayerTier.push_back(softMaxLayer);
	}
}

void ConfigurationParser::ParseOutputLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier)
{
	string lossFunctionName;
	LossFunctionType lossFunction;
	bool parsedLossFunction = false;
	uint numGuesses;
	bool parsedNumGuesses = false;
	string prevLayersParam;
	bool parsedPrevLayers = false;

	vector<string>& tierLines = m_tiersLines[tierIndex];
	for (string& line : tierLines)
	{
		parsedLossFunction = parsedLossFunction || ParseParameterString(line, "lossFunction", lossFunctionName);
		parsedNumGuesses = parsedNumGuesses || ParseParameterUint(line, "numGuesses", numGuesses);
		parsedPrevLayers = parsedPrevLayers || ParseParameterString(line, "prevLayers", prevLayersParam);
	}

	ShipAssert(parsedLossFunction, "Can't parse loss function for Output layer!");

	lossFunction = GetLossFunctionType(lossFunctionName);

	uint tierSize = 1;
	uint layerIndex = 0;
	ParallelismMode parallelismMode = ParallelismMode::Model;
	vector<Layer*> prevLayers = FindPrevLayers(parallelismMode, layerIndex, tierSize, tierIndex - 1, prevLayersParam);
	uint inputNumChannels;
	uint inputDataWidth;
	uint inputDataHeight;
	uint inputDataCount;
	bool holdsInputData;
	FindInputParams(prevLayers, layerIndex, tierSize, parallelismMode, inputNumChannels, inputDataWidth, inputDataHeight, inputDataCount, holdsInputData);

	CudaAssert(cudaSetDevice(0));

	InputLayer* inputLayer = static_cast<InputLayer*>(m_layersTiers[0][0]);
	OutputLayer* outputLayer = new OutputLayer(m_neuralNet->GetDeviceCalculationStreams()[0], m_neuralNet->GetDeviceMemoryStreams()[0],
		inputNumChannels * inputDataWidth * inputDataHeight, inputDataCount, (uint)m_maxNetworkTierSize * inputDataCount, lossFunction, parsedNumGuesses, numGuesses,
		inputLayer->GetNumTestPasses());

	for (Layer* prevLayer : prevLayers)
	{
		outputLayer->AddPrevLayer(prevLayer);
		prevLayer->AddNextLayer(outputLayer);
	}

	outLayerTier.push_back(outputLayer);
}

vector<Layer*> ConfigurationParser::ParseLayersTier(size_t tierIndex, LayerType tierLayerType)
{
	vector<Layer*> layerTier;

	if (tierLayerType == LayerType::Input)
	{
		ParseInputLayerTier(layerTier);
	}
	else if (tierLayerType == LayerType::Convolutional)
	{
		ParseConvolutionalLayerTier(tierIndex, layerTier);
	}
	else if (tierLayerType == LayerType::ResponseNormalization)
	{
		ParseResponseNormalizationLayerTier(tierIndex, layerTier);
	}
	else if (tierLayerType == LayerType::MaxPool)
	{
		ParseMaxPoolLayerTier(tierIndex, layerTier);
	}
	else if (tierLayerType == LayerType::Standard)
	{
		ParseStandardLayerTier(tierIndex, layerTier);
	}
	else if (tierLayerType == LayerType::Dropout)
	{
		ParseDropoutLayerTier(tierIndex, layerTier);
	}
	else if (tierLayerType == LayerType::SoftMax)
	{
		ParseSoftMaxLayerTier(tierIndex, layerTier);
	}
	else if (tierLayerType == LayerType::Output)
	{
		ParseOutputLayerTier(tierIndex, layerTier);
	}

	return layerTier;
}