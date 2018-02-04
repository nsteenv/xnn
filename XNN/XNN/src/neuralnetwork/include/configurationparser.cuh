// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural networks configuration parser.
// Created: 03/17/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <fstream>

#include "neuralnet.cuh"

using namespace std;

// Parsing mode.
enum class ParsingMode
{
	Training,
	Prediction
};

class ConfigurationParser
{
private:
	// Neural network which is parsed.
	NeuralNet* m_neuralNet;

	// Parsing mode.
	ParsingMode m_parsingMode;

	// Folder with data for training.
	string m_dataFolder;

	// Training data batch size.
	uint m_batchSize;

	// Should we initialize parameters in layers.
	bool m_initializeLayersParams;

	// Parsed layers tiers.
	vector<vector<Layer*> > m_layersTiers;

	// Parsed tiers' lines.
	vector<vector<string> > m_tiersLines;

	// Size of network tier with maximal size.
	size_t m_maxNetworkTierSize;

	// Trims line.
	string TrimLine(string line);

	// Parses tiers' lines.
	void ParseTierLines(string configurationFile);

	// Finds size of network tier with maximal size.
	void FindMaxNetworkTierSize();

	// Parses unsigned int parameter from line, returns true if successful.
	bool ParseParameterUint(string line, string parameterName, uint& parameterValue);

	// Parses float parameter from line, returns true if successful.
	bool ParseParameterFloat(string line, string parameterName, float& parameterValue);

	// Parses string parameter from line, returns true if successful.
	bool ParseParameterString(string line, string parameterName, string& parameterValue);

	// Gets layer type based on layer type name.
	LayerType GetLayerType(string layerTypeName);

	// Gets activation type based on activation type name.
	ActivationType GetActivationType(string activationTypeName);

	// Gets loss function type based on loss function name.
	LossFunctionType GetLossFunctionType(string lossFunctionName);

	// Gets data type based on data type name.
	DataType GetDataType(string dataTypeName);

	// Parses layers tiers.
	void ParseLayersTiers();

	// Parses layers tier with specified type.
	vector<Layer*> ParseLayersTier(size_t tierIndex, LayerType tierLayerType);

	// Finds previous layers of current tier.
	vector<Layer*> FindPrevLayers(ParallelismMode currTierParallelismMode, uint layerIndex, uint currTierSize, size_t prevTierIndex, string prevLayersParam);

	// Finds if layers in current tier should hold activation gradients.
	bool ShouldHoldActivationGradients(ParallelismMode currTierParallelismMode, uint currTierSize, size_t currTierIndex, uint layerIndex);

	// Finds input parameters based on previous layers.
	void FindInputParams(vector<Layer*>& prevLayers, uint layerIndex, uint tierSize, ParallelismMode parallelismMode, uint& inputNumChannels, uint& inputDataWidth,
		uint& inputDataHeight, uint& inputDataCount, bool& holdsInputData);

	// Parses input layer tier.
	void ParseInputLayerTier(vector<Layer*>& outLayerTier);

	// Parses convolutional layer tier.
	void ParseConvolutionalLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier);

	// Parses response normalization layer tier.
	void ParseResponseNormalizationLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier);

	// Parses max pool layer tier.
	void ParseMaxPoolLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier);

	// Parses standard layer tier.
	void ParseStandardLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier);

	// Parses dropout layer tier.
	void ParseDropoutLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier);

	// Parses soft max layer tier.
	void ParseSoftMaxLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier);

	// Parses output layer tier.
	void ParseOutputLayerTier(size_t tierIndex, vector<Layer*>& outLayerTier);

public:
	// Parses network configuration and creates network.
	NeuralNet* ParseNetworkFromConfiguration(ParsingMode parsingMode, string configurationFile, string dataFolder, uint batchSize, bool initializeLayersParams);
};