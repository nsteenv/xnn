// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Network trainer.
// Created: 12/27/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <condition_variable>

#include "datamaker.cuh"
#include "../../neuralnetwork/include/configurationparser.cuh"
#include "../../neuralnetwork/include/neuralnet.cuh"

using namespace std;

class Trainer
{
private:
	enum Direction
	{
		FORWARD,
		BACKWARD
	};

	// Neural network which is trained.
	NeuralNet* m_neuralNet;

	// Neural networks configuration parser.
	ConfigurationParser m_configurationParser;

	// Should we load model from checkpoint and resume training from there.
	bool m_loadFromCheckpoint;

	// Should we skip test part.
	bool m_noTest;

	// Default GPU device to train on.
	int m_defaultGPU;

	// Starting epoch.
	uint m_startEpoch;

	// Number of epochs to train.
	uint m_numEpochs;

	// Training data batch size.
	uint m_batchSize;

	// Network configuration file.
	string m_networkConfigurationFile;

	// Folder with data for training.
	string m_dataFolder;

	// Folder to save checkpoints and progress in.
	string m_workFolder;

	// Size of input batch to load.
	size_t m_inputBatchSize;

	// Calculated loss.
	float m_loss;

	// Calculated accuracy.
	float m_accuracy;

	// Calculated multiple guess accuracy.
	float m_multipleGuessAccuracy;

	// Train data.
	vector<pair<string, uint> > m_trainData;

	// Test data.
	vector<pair<string, uint> > m_testData;

	// Synchronization for when all layers in first tier fprop.
	condition_variable m_firstTierLayersFpropSync;

	// Whether all layers in first tier fpropped.
	bool m_allFirstTierLayersFpropped;

	// Number of layers in first tier that fpropped.
	size_t m_numFirstTierLayersFpropped;

	// Mutex for changing first tier layers fpropped condition.
	mutex m_firstTierLayersFropMutex;

	// Check if gradients are loaded to data parallelized layer at the transition from data to model parallelized layers,
	// before we write over them in backward propagation through model layers.
	bool m_gradientsToDataLayerLoaded;

	// Mutex for controlling are gradients to data parallelized layer loaded check.
	mutex m_gradientsToDataLayerLoadMutex;

	// Thread for loading gradients to data parallelized layer at the transition from data to model parallelized layers;
	thread m_gradientsToDataLayerLoadThread;

	// Helper buffers for transferring weights gradients buffers between layers in data parallel tiers.
	vector<float*> m_dataParallelTiersWeightsGradientBuffers;

	// Helper buffers for transferring biases gradients buffers between layers in data parallel tiers.
	vector<float*> m_dataParallelTiersBiasesGradientBuffers;

	// Initializes trainer parameters from given training configuration file.
	void InitializeTrainer();

	// Finds best choice for default device.
	void SetDefaultDevice();

	// Initializes network from given network configuration file.
	void InitializeNetwork(ParsingMode parsingMode);

	// Checks if configuration is valid.
	void ValidateConfiguration();

	// Saves training checkpoint.
	void SaveCheckpoint(uint currEpoch, size_t dataTrainedCount, bool finalCheckpoint);

	// Loads saved checkpoint.
	void LoadCheckpoint();
	
	// Loads image data for training from certain folder;
	void LoadImageData(string folder, vector<pair<string, uint> >& data);

	// Loads image data for training.
	void LoadImageData();

	// Loads data info.
	void LoadDataInfo();

	// Loads text data for training from certain instances file.
	void LoadTextData(string instancesFile, vector<pair<string, uint> >& data);

	// Loads text data for training.
	void LoadTextData();

	// Loads data for training.
	void LoadData();

	// Loads data batch to input layer.
	void LoadBatch(const vector<string>& dataFiles, PropagationMode propagationMode);

	// Loads gradients to layer with data parallelism, when his next layers have model parallelism.
	void LoadGradientsToLayer(Layer* layer);

	// Checks if two layers are compatible for same split.
	bool LayersCompatibleForSplit(Layer* firstLayer, Layer* secondLayer);

	// Creates layer splits for parallel propagation.
	vector<vector<Layer*> > CreateLayerSplits(size_t currTier, size_t& nextTier, int increment, function<bool(size_t)> stopCondition);

	// Does forward propagation on layers.
	void ForwardPropagateLayers(const vector<Layer*>& layers, PropagationMode propagationMode);

	// Does backward propagation on layers.
	void BackwardPropagateLayers(const vector<Layer*>& layers);

	// Propagates loaded data batch forward through the network.
	void PropagateBatchForward(size_t currTier, size_t& nextTier, PropagationMode propagationMode);

	// Propagates loaded data batch backward through the network.
	void PropagateBatchBackward(uint currEpoch, size_t currTier, size_t& nextTier, Direction& direction);

	// Prints result metrics.
	void PrintResults(uint percentDone, size_t dataCount, PropagationMode propagationMode);

	// Updates parameters of layers in specified tiers.
	void UpdateTiersParameters(uint currEpoch, size_t beginTier, size_t endTier);

	// Trains network on loaded data batch.
	void TrainBatch(uint currEpoch);

	// Trains network.
	void TrainNetwork();

	// Tests network on loaded data batch.
	void TestBatch();

	// Tests network.
	void TestNetwork();

	// Resets GPU devices used for training.
	void ResetDevices();

public:
	// Default constructor.
	Trainer();

	// Destructor.
	~Trainer();

	// Parameters signatures.
	static const string c_configurationSignature;
	static const string c_dataFolderSignature;
	static const string c_workFolderSignature;
	static const string c_numEpochsSignature;
	static const string c_batchSizeSignature;
	static const string c_loadFromCheckpointSignature;
	static const string c_defaultGpuSignature;
	static const string c_noTestSignature;

	// Other constants.
	static const string c_configurationFileName;
	static const string c_resultsFileName;
	static const string c_oldCheckpointModelName;
	static const string c_checkpointModelName;
	static const string c_predictionModelName;

	// Parses arguments for training.
	bool ParseArguments(int argc, char *argv[]);

	// Runs training of network.
	void RunTraining();
};