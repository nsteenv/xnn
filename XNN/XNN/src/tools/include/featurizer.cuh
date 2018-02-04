// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network features extractor.
// Created: 04/03/2016.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include "datamaker.cuh"
#include "../../neuralnetwork/include/configurationparser.cuh"
#include "../../neuralnetwork/include/neuralnet.cuh"

using namespace std;

class Featurizer
{
private:
	// Neural network to extract features with.
	NeuralNet* m_neuralNet;

	// Neural networks configuration parser.
	ConfigurationParser m_configurationParser;

	// File with trained model of the network.
	string m_modelFile;

	// Folder with data for features extraction.
	string m_dataFolder;

	// Data for features extraction.
	vector<string> m_data;

	// Batch size.
	uint m_batchSize;

	// Index of layer from which to extract features.
	size_t m_targetLayerIndex;

	// Network configuration file.
	string m_networkConfigurationFile;

	// Host target layer activations.
	float* m_hostTargetLayerActivations;

	// Initializes network for features extraction.
	void InitializeNetwork();

	// Loads data for which to extract features.
	void LoadData();

	// Loads batch for featurizing.
	void LoadBatch(const vector<string>& dataFiles);

	// Extracts features on batch of data.
	void ExtractFeaturesOnBatch();

	// Extracts features.
	void ExtractFeatures();

public:
	// Default constructor.
	Featurizer();

	// Destructor.
	~Featurizer();

	// Parameters signatures.
	static const string c_configurationSignature;
	static const string c_modelFileSignature;
	static const string c_dataFolderSignature;
	static const string c_batchSizeSignature;
	static const string c_targetLayerSignature;

	// Parses arguments for features extracting.
	bool ParseArguments(int argc, char *argv[]);

	// Runs features extraction.
	void RunExtraction();
};