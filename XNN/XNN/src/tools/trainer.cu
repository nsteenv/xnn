// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Network trainer.
// Created: 12/27/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/trainer.cuh"

const string Trainer::c_configurationSignature = "-configfile";
const string Trainer::c_dataFolderSignature = "-datafolder";
const string Trainer::c_workFolderSignature = "-workfolder";
const string Trainer::c_numEpochsSignature = "-numepochs";
const string Trainer::c_batchSizeSignature = "-batchsize";
const string Trainer::c_loadFromCheckpointSignature = "-continue";
const string Trainer::c_defaultGpuSignature = "-gpu";
const string Trainer::c_noTestSignature = "-notest";

const string Trainer::c_configurationFileName = "configuration.xnn";
const string Trainer::c_resultsFileName = "results.txt";
const string Trainer::c_oldCheckpointModelName = "old_model_checkpoint.xnnm";
const string Trainer::c_checkpointModelName = "model_checkpoint.xnnm";
const string Trainer::c_predictionModelName = "model.xnnm";

Trainer::Trainer()
{
	m_neuralNet = NULL;
	m_defaultGPU = -1;
}

bool Trainer::ParseArguments(int argc, char *argv[])
{
	ParseArgument(argc, argv, c_loadFromCheckpointSignature, m_loadFromCheckpoint);
	ParseArgument(argc, argv, c_noTestSignature, m_noTest);

	if (!ParseArgument(argc, argv, c_dataFolderSignature, m_dataFolder) ||
		!ParseArgument(argc, argv, c_workFolderSignature, m_workFolder) ||
		!ParseArgument(argc, argv, c_numEpochsSignature, m_numEpochs) ||
		!ParseArgument(argc, argv, c_batchSizeSignature, m_batchSize))
	{
		return false;
	}

	if (m_loadFromCheckpoint)
	{
		m_networkConfigurationFile = m_workFolder + "\\" + c_configurationFileName;
	}
	else if (!ParseArgument(argc, argv, c_configurationSignature, m_networkConfigurationFile))
	{
		return false;
	}

	uint defaultGpu;
	if (ParseArgument(argc, argv, c_defaultGpuSignature, defaultGpu))
	{
		m_defaultGPU = (int)defaultGpu;
	}

	return true;
}

Trainer::~Trainer()
{
	if (m_neuralNet != NULL)
	{
		delete m_neuralNet;
	}

	if (!m_dataParallelTiersWeightsGradientBuffers.empty() && m_dataParallelTiersWeightsGradientBuffers[0] != NULL)
	{
		for (size_t i = 0; i < m_dataParallelTiersWeightsGradientBuffers.size(); ++i)
		{
			CudaAssert(cudaFree(m_dataParallelTiersWeightsGradientBuffers[i]));
			CudaAssert(cudaFree(m_dataParallelTiersBiasesGradientBuffers[i]));
		}
	}
}

void Trainer::SetDefaultDevice()
{
	if (m_defaultGPU != -1)
	{
		// It's been set explicitely with run parameters.
		return;
	}

	// TODO: This would require to change whole logic of tiers, currently we assume everywhere that index in tier is
	// equal to index of GPU used. Make this work one day when you find time...

	//// Find one with most amount of free memory.
	//int numDevices;
	//cudaGetDeviceCount(&numDevices);
	//size_t maxFreeMemorySize = 0;
	//// By default it is zero.
	//int maxFreeMemoryDevice = 0;
	//for (int currDevice = 0; currDevice < numDevices; ++currDevice)
	//{
	//	CudaAssert(cudaSetDevice(currDevice));
	//	size_t freeMemorySize = GetSizeOfAvailableGpuMemory();
	//	if (freeMemorySize > maxFreeMemorySize)
	//	{
	//		maxFreeMemorySize = freeMemorySize;
	//		maxFreeMemoryDevice = currDevice;
	//	}
	//}
	//m_defaultGPU = maxFreeMemoryDevice;

	m_defaultGPU = 0;

	CudaAssert(cudaSetDevice(m_defaultGPU));
}

void Trainer::InitializeNetwork(ParsingMode parsingMode)
{
	m_neuralNet = m_configurationParser.ParseNetworkFromConfiguration(parsingMode, m_networkConfigurationFile, m_dataFolder, m_batchSize, parsingMode == ParsingMode::Training);
}

void Trainer::InitializeTrainer()
{
	// Find input batch size.
	m_inputBatchSize = m_neuralNet->GetInputLayer()->GetInputDataCount();
	m_numFirstTierLayersFpropped = 0;

	// Setup helper buffers.
	vector<vector<Layer*> >& layerTiers = m_neuralNet->GetLayerTiers();
	for (size_t currTier = 0; currTier < layerTiers.size(); ++currTier)
	{
		m_dataParallelTiersWeightsGradientBuffers.push_back(NULL);
		m_dataParallelTiersBiasesGradientBuffers.push_back(NULL);
	}

	if (!m_loadFromCheckpoint)
	{
		// Copy configuration over to work folder.
		ifstream inputConfigFile(m_networkConfigurationFile);
		ofstream outputConfigFile(m_workFolder + "\\" + c_configurationFileName);
		string line;
		while (getline(inputConfigFile, line))
		{
			outputConfigFile << line << endl;
		}
		outputConfigFile.close();
		inputConfigFile.close();

		m_startEpoch = 1;
	}
}

void Trainer::ValidateConfiguration()
{
	// TODO: Finish.
}

void Trainer::SaveCheckpoint(uint currEpoch, size_t dataTrainedCount, bool finalCheckpoint)
{
	// Saving model.
	remove((m_workFolder + "\\" + c_oldCheckpointModelName).c_str());
	rename((m_workFolder + "\\" + c_checkpointModelName).c_str(), (m_workFolder + "\\" + c_oldCheckpointModelName).c_str());
	remove((m_workFolder + "\\" + c_checkpointModelName).c_str());
	if (finalCheckpoint)
	{
		m_neuralNet->SaveModelForPrediction(m_workFolder + "\\" + c_predictionModelName);
		remove((m_workFolder + "\\" + c_oldCheckpointModelName).c_str());
	}
	else
	{
		m_neuralNet->SaveModelCheckpoint(m_workFolder + "\\" + c_checkpointModelName);
	}

	// Saving results.
	ofstream resultsFile(m_workFolder + "\\" + c_resultsFileName);
	resultsFile << "Trained epochs: " << currEpoch << endl;
	resultsFile << "Batch size: " << m_batchSize << endl << endl;
	resultsFile << "Training results:    Loss: " << m_loss / dataTrainedCount << "  Accuracy: " << m_accuracy / dataTrainedCount;
	if (m_neuralNet->GetOutputLayer()->ShouldCalculateMultipleGuessAccuracy())
	{
		resultsFile << "  Multiple guess accuracy: " << m_multipleGuessAccuracy / dataTrainedCount;
	}
	resultsFile << endl << endl;
}

void Trainer::LoadCheckpoint()
{
	// Loading model.
	m_neuralNet->LoadModelCheckpoint(m_workFolder + "\\" + c_checkpointModelName);

	// Parse epoch.
	ifstream resultsFile(m_workFolder + "\\" + c_resultsFileName);
	string line;
	getline(resultsFile, line);
	size_t valuePosition = line.find_last_of(":");
	string epochValue = line.substr(valuePosition + 1);
	m_startEpoch = (uint)stoi(epochValue) + 1;
	// Parse results.
	getline(resultsFile, line);
	getline(resultsFile, line);
	getline(resultsFile, line);
	line = line.substr(line.find_first_of(":") + 1);
	line = line.substr(line.find_first_of(":") + 2);
	size_t accuracyPosition = line.find_first_of(":");
	string lossValue = line.substr(0, accuracyPosition - string("Accuracy:").length() - 1);
	m_loss = stof(lossValue);
	line = line.substr(accuracyPosition + 2);
	if (m_neuralNet->GetOutputLayer()->ShouldCalculateMultipleGuessAccuracy())
	{
		size_t mulAccuracyPosition = line.find_first_of(":");
		string accuracyValue = line.substr(0, mulAccuracyPosition - string("Multiple guess accuracy:").length() - 1);
		m_accuracy = stof(accuracyValue);
		line = line.substr(mulAccuracyPosition + 2);
		m_multipleGuessAccuracy = stof(line);
	}
	else
	{
		line = line.substr(accuracyPosition + 2);
		m_accuracy = stof(line);
	}
}

void Trainer::LoadImageData(string folder, vector<pair<string, uint> >& data)
{
	ifstream labelsFile(m_dataFolder + "\\" + folder + "\\" + DataMaker::c_labelsFileName);
	string imageName;
	uint label;
	while (labelsFile >> imageName >> label)
	{
		data.push_back(make_pair(imageName, label - 1));
	}
}

void Trainer::LoadImageData()
{
	// Loading data.
	LoadImageData("train", m_trainData);
	ShipAssert(m_trainData.size() >= m_inputBatchSize, "There is not enough train data!");
	random_shuffle(m_trainData.begin(), m_trainData.end());
	if (!m_noTest)
	{
		LoadImageData("test", m_testData);
		ShipAssert(m_testData.size() >= m_inputBatchSize, "There is not enough test data!");
	}
}

void Trainer::LoadDataInfo()
{
	if (m_neuralNet->GetInputLayer()->GetDataType() == DataType::Image)
	{
		ifstream dataInfoFile(m_dataFolder + "\\" + DataMaker::c_dataInfoFileName);
		// Load mean values.
		string temp;
		uint value;
		vector<uchar> meanValues;
		dataInfoFile >> temp /*Mean*/ >> temp /*pixel*/ >> temp /*values:*/;
		for (size_t i = 0; i < m_neuralNet->GetInputLayer()->GetActivationNumChannels(); ++i)
		{
			dataInfoFile >> value >> temp /*,*/;
			ShipAssert(value != 0, "Can't read mean channel values!");
			meanValues.push_back((uchar)value);
		}
		m_neuralNet->GetInputLayer()->SetChannelMeanValues(meanValues);
	}
}

void Trainer::LoadTextData(string instancesFile, vector<pair<string, uint> >& data)
{
	ifstream labelsFile(m_dataFolder + "\\" + instancesFile);
	string instance;
	string features;
	string label;
	while (getline(labelsFile, instance))
	{
		size_t split = instance.find_first_of(' ');
		label = instance.substr(0, split);
		features = instance.substr(split + 1);
		data.push_back(make_pair(features, stoul(label)));
	}
}

void Trainer::LoadTextData()
{
	LoadTextData("trainSet.txt", m_trainData);
	ShipAssert(m_trainData.size() >= m_inputBatchSize, "There is not enough train data!");
	random_shuffle(m_trainData.begin(), m_trainData.end());
	if (!m_noTest)
	{
		LoadTextData("testSet.txt", m_testData);
		ShipAssert(m_testData.size() >= m_inputBatchSize, "There is not enough test data!");
	}
}

void Trainer::LoadData()
{
	InputLayer* inputLayer = m_neuralNet->GetInputLayer();
	if (inputLayer->GetDataType() == DataType::Image)
	{
		LoadImageData();
	}
	else
	{
		LoadTextData();
	}
}

void Trainer::LoadBatch(const vector<string>& dataFiles, PropagationMode propagationMode)
{
	InputLayer* inputLayer = m_neuralNet->GetInputLayer();

	inputLayer->SetDataFilesToLoad(dataFiles, propagationMode);
	// Load data from disk to host memory.
	inputLayer->LoadInputs();

	// Wait for first layer to fprop.
	unique_lock<mutex> lock(m_firstTierLayersFropMutex);
	while (!m_allFirstTierLayersFpropped)
	{
		m_firstTierLayersFpropSync.wait(lock);
	}

	// Load data from host to GPU memory.
	inputLayer->DoForwardProp(propagationMode);
}

void Trainer::LoadGradientsToLayer(Layer* layer)
{
	CudaAssert(cudaSetDevice(layer->GetIndexInTier()));
	layer->LoadActivationGradients();
	layer->SynchronizeMemoryOperations();
}

void Trainer::UpdateTiersParameters(uint currEpoch, size_t beginTier, size_t endTier)
{
	vector<vector<Layer*> >& layerTiers = m_neuralNet->GetLayerTiers();

	for (size_t currTier = beginTier; currTier < endTier; ++currTier)
	{
		// Always operating on computational stream of first layer in tier.
		CudaAssert(cudaSetDevice(0));

		if (layerTiers[currTier][0]->GetLayerType() == LayerType::Convolutional || layerTiers[currTier][0]->GetLayerType() == LayerType::Standard)
		{
			// In tiers with data parallelism we need to sync gradients between layers.
			if (layerTiers[currTier][0]->GetParallelismMode() == ParallelismMode::Data)
			{
				// Casting layers to appropriate type to determine buffers to work on.
				vector<float*> weightsGradientsBuffers, biasesGradientsBuffers;
				size_t weightsGradientsBufferSize, biasesGradientsBufferSize;
				if (layerTiers[currTier][0]->GetLayerType() == LayerType::Convolutional)
				{
					for (size_t layerIndex = 0; layerIndex < layerTiers[currTier].size(); ++layerIndex)
					{
						ConvolutionalLayer* convLayer = static_cast<ConvolutionalLayer*>(layerTiers[currTier][layerIndex]);
						if (layerIndex == 0)
						{
							weightsGradientsBufferSize = convLayer->GetFiltersBufferSize();
							biasesGradientsBufferSize = convLayer->GetBiasesBufferSize();
						}
						weightsGradientsBuffers.push_back(convLayer->GetFiltersGradientsBuffer());
						biasesGradientsBuffers.push_back(convLayer->GetBiasesGradientsBuffer());
					}
				}
				else
				{
					for (size_t layerIndex = 0; layerIndex < layerTiers[currTier].size(); ++layerIndex)
					{
						StandardLayer* standardLayer = static_cast<StandardLayer*>(layerTiers[currTier][layerIndex]);
						if (layerIndex == 0)
						{
							weightsGradientsBufferSize = standardLayer->GetWeightsBufferSize();
							biasesGradientsBufferSize = standardLayer->GetBiasesBufferSize();
						}
						weightsGradientsBuffers.push_back(standardLayer->GetWeightsGradientsBuffer());
						biasesGradientsBuffers.push_back(standardLayer->GetBiasesGradientsBuffer());
					}
				}

				// Allocating helper buffers if they are used first time.
				if (m_dataParallelTiersWeightsGradientBuffers[currTier] == NULL)
				{
					CudaAssert(cudaMalloc<float>(&m_dataParallelTiersWeightsGradientBuffers[currTier], weightsGradientsBufferSize));
					CudaAssert(cudaMalloc<float>(&m_dataParallelTiersBiasesGradientBuffers[currTier], biasesGradientsBufferSize));
				}

				// Sum up weights and biases gradients from all layers into first layer's gradients buffers.
				for (int layerIndex = 1; layerIndex < layerTiers[currTier].size(); ++layerIndex)
				{
					// Copy over weights gradients buffer.
					CudaAssert(cudaMemcpyPeer(m_dataParallelTiersWeightsGradientBuffers[currTier], 0, weightsGradientsBuffers[layerIndex],
						layerIndex, weightsGradientsBufferSize));

					// Add them to first layer's weights gradients buffer.
					CalculateElementWiseSum(weightsGradientsBuffers[0], m_dataParallelTiersWeightsGradientBuffers[currTier],
						(uint)(weightsGradientsBufferSize / sizeof(float)), weightsGradientsBuffers[0], 0);

					// Copy over biases gradients buffer.
					CudaAssert(cudaMemcpyPeer(m_dataParallelTiersBiasesGradientBuffers[currTier], 0, biasesGradientsBuffers[layerIndex],
						layerIndex, biasesGradientsBufferSize));

					// Add them to first layer's biases gradients buffer.
					CalculateElementWiseSum(biasesGradientsBuffers[0], m_dataParallelTiersBiasesGradientBuffers[currTier],
						(uint)(biasesGradientsBufferSize / sizeof(float)), biasesGradientsBuffers[0], 0);
				}

				cudaStreamSynchronize(0);

				// Copy summed gradients from first layer's buffers to other layers' buffers.
				for (int layerIndex = 1; layerIndex < layerTiers[currTier].size(); ++layerIndex)
				{
					// Copy over weights gradients buffer.
					CudaAssert(cudaMemcpyPeer(weightsGradientsBuffers[layerIndex], layerIndex, weightsGradientsBuffers[0],
						0, weightsGradientsBufferSize));

					// Copy over biases gradients buffer.
					CudaAssert(cudaMemcpyPeer(biasesGradientsBuffers[layerIndex], layerIndex, biasesGradientsBuffers[0],
						0, biasesGradientsBufferSize));
				}
			}

			// Update each layer's parameters.
			for (Layer* layer : layerTiers[currTier])
			{
				CudaAssert(cudaSetDevice(layer->GetIndexInTier()));
				layer->UpdateLayerParameters((float)currEpoch / m_numEpochs);
			}
		}
	}

	// Reverting back to default device since this work is not done in separate thread.
	CudaAssert(cudaSetDevice(0));
}

bool Trainer::LayersCompatibleForSplit(Layer* firstLayer, Layer* secondLayer)
{
	return firstLayer->GetParallelismMode() == secondLayer->GetParallelismMode() &&
		firstLayer->GetNextLayers().size() == 1 && secondLayer->GetPrevLayers().size() == 1;
}

void Trainer::ForwardPropagateLayers(const vector<Layer*>& layers, PropagationMode propagationMode)
{
	uint splitIndexInTier = layers[0]->GetIndexInTier();
	CudaAssert(cudaSetDevice(splitIndexInTier));

	Layer* prevSplitLastLayer = layers[0]->GetPrevLayers().size() <= splitIndexInTier ? NULL : layers[0]->GetPrevLayers()[splitIndexInTier];
	Layer* prevTierFirstLayer = layers[0]->GetPrevLayers()[0];

	for (size_t layerIndex = 0; layerIndex < layers.size(); ++layerIndex)
	{
		Layer* layer = layers[layerIndex];

		// Check if we should start preloading inputs for next propagation of layers with model parallelism connected with
		// layers with data parallelism, in parallel with computation in these layers.
		if (layerIndex == 0 && layer->GetParallelismMode() == ParallelismMode::Model &&
			prevSplitLastLayer != NULL && prevSplitLastLayer->GetParallelismMode() == ParallelismMode::Model)
		{
			Layer* prevSplitFirstLayer = prevSplitLastLayer;
			while (prevSplitFirstLayer->GetLayerType() != LayerType::Input && LayersCompatibleForSplit(prevSplitFirstLayer->GetPrevLayers()[0], prevSplitFirstLayer))
			{
				prevSplitFirstLayer = prevSplitFirstLayer->GetPrevLayers()[0];
			}

			if (prevSplitFirstLayer->GetLayerType() != LayerType::Input && prevSplitFirstLayer->GetPrevLayers()[0]->GetParallelismMode() == ParallelismMode::Data &&
				prevSplitFirstLayer->GetInputLayerIndexInTier() < (int)layer->GetPrevLayers().size() - 1)
			{
				// Start preloading inputs for next propagation through this layer.
				prevSplitFirstLayer->LoadInputs();
			}
		}

		// Load input if needed.
		if (layer->GetInputLayerIndexInTier() < 0)
		{
			layer->LoadInputs();

			// No need for sync with input layer, since it's propagation is worked on in different thread
			// which is joined before computation thread starts.
			if (layer->HoldsInputData())
			{
				// Making sure inputs are loaded before computation.
				layer->SynchronizeMemoryOperations();
			}
		}
		else
		{
			// Skipping load of inputs if they are preloaded, but need to sync to make sure they are finished preloading.
			if (layer->GetInputLayerIndexInTier() != layer->GetIndexInTier())
			{
				layer->SynchronizeMemoryOperations();
			}
		}
		layer->IncreaseInputLayerIndexInTier();

		// Do forward propagation.
		layer->DoForwardProp(propagationMode);

		// Check if we should start preloading input data for next batch propagation.
		if (layerIndex == 0 && prevTierFirstLayer->GetLayerType() == LayerType::Input)
		{
			lock_guard<mutex> lock(m_firstTierLayersFropMutex);
			++m_numFirstTierLayersFpropped;
			if (m_numFirstTierLayersFpropped == prevTierFirstLayer->GetNextLayers().size())
			{
				// Notify that all first tier layers fpropped.
				m_allFirstTierLayersFpropped = true;
				m_numFirstTierLayersFpropped = 0;
				m_firstTierLayersFpropSync.notify_one();
			}
		}

		// Gathering batch results if this is output layer.
		if (propagationMode == PropagationMode::Train && layer->GetLayerType() == LayerType::Output)
		{
			OutputLayer* outputLayer = static_cast<OutputLayer*>(layer);
			// No need for locking here since there can be only one output layer.
			m_loss += outputLayer->GetLoss();
			m_accuracy += outputLayer->GetAccuracy();
			if (outputLayer->ShouldCalculateMultipleGuessAccuracy())
			{
				m_multipleGuessAccuracy += outputLayer->GetMultipleGuessAccuracy();
			}
		}
	}

	// Making sure calculations are finished before moving on to next tier.
	layers.back()->SynchronizeCalculations();
}

void Trainer::BackwardPropagateLayers(const vector<Layer*>& layers)
{
	CudaAssert(cudaSetDevice(layers[0]->GetIndexInTier()));

	// Layers are sorted from last to first, imagine everything in this method written as if we are going backwards through the network.
	Layer* prevSplitFirstLayer = layers[0]->GetNextLayers().empty() ? NULL : layers[0]->GetNextLayers()[0];

	for (size_t layerIndex = 0; layerIndex < layers.size(); ++layerIndex)
	{
		Layer* layer = layers[layerIndex];

		// Skipping load of activation gradients if they are preloaded.
		if (!(layerIndex == 0 && layer->GetParallelismMode() == ParallelismMode::Data && prevSplitFirstLayer != NULL &&
			prevSplitFirstLayer->GetParallelismMode() == ParallelismMode::Model && layer->GetIndexInTier() < layer->GetTierSize() - 1))
		{
			layer->LoadActivationGradients();

			if (layer->HoldsActivationGradients())
			{
				// Making sure activation gradients are loaded before computation.
				layer->SynchronizeMemoryOperations();
			}
		}

		// We need to delay doing backward prop on model parallelized layer which is currently sending gradients to data parallelized layer before it.
		if (layerIndex == layers.size() - 1 && layer->GetParallelismMode() == ParallelismMode::Model &&
			layer->GetPrevLayers()[0]->GetParallelismMode() == ParallelismMode::Data)
		{
			lock_guard<mutex> lock(m_gradientsToDataLayerLoadMutex);
			if (!m_gradientsToDataLayerLoaded)
			{
				m_gradientsToDataLayerLoadThread.join();
				m_gradientsToDataLayerLoaded = true;
			}
		}

		// Do backward propagation.
		layer->DoBackwardProp();
	}

	// Making sure calculations are finished before moving on to next tier.
	layers.back()->SynchronizeCalculations();
}

vector<vector<Layer*> > Trainer::CreateLayerSplits(size_t currTier, size_t& nextTier, int increment, function<bool(size_t)> stopCondition)
{
	vector<vector<Layer*> >& layerTiers = m_neuralNet->GetLayerTiers();

	// Split is group of layers from adjacent tiers that can work together in one pass.
	// If those adjacent tiers contain multiple layers, than we have multiple splits to work on in parallel during one pass.
	vector<vector<Layer*> > layerSplits;
	for (size_t split = 0; split < layerTiers[currTier].size(); ++split)
	{
		vector<Layer*> layerSplit;
		layerSplit.push_back(layerTiers[currTier][split]);
		layerSplits.push_back(layerSplit);
	}
	for (nextTier = currTier + increment; stopCondition(nextTier); nextTier += increment)
	{
		bool layersCompatible = currTier < nextTier ? LayersCompatibleForSplit(layerTiers[currTier][0], layerTiers[nextTier][0]) :
			LayersCompatibleForSplit(layerTiers[nextTier][0], layerTiers[currTier][0]);

		if (layerTiers[currTier].size() == layerTiers[nextTier].size() && layersCompatible)
		{
			for (size_t split = 0; split < layerSplits.size(); ++split)
			{
				layerSplits[split].push_back(layerTiers[nextTier][split]);
			}
		}
		else
		{
			break;
		}
	}

	return layerSplits;
}

void Trainer::PropagateBatchForward(size_t currTier, size_t& nextTier, PropagationMode propagationMode)
{
	vector<vector<Layer*> >& layerTiers = m_neuralNet->GetLayerTiers();

	// Make splits.
	vector<vector<Layer*> > layerSplits = CreateLayerSplits(currTier, nextTier, 1, [layerTiers](size_t tier) { return tier < layerTiers.size(); });

	// Propagate on split.
	vector<thread> propagationThreads;
	for (size_t split = 0; split < layerSplits.size(); ++split)
	{
		propagationThreads.push_back(thread([this, layerSplits, split, propagationMode] { this->ForwardPropagateLayers(layerSplits[split], propagationMode); }));
	}
	for (size_t split = 0; split < layerSplits.size(); ++split)
	{
		propagationThreads[split].join();
	}
}

void Trainer::PropagateBatchBackward(uint currEpoch, size_t currTier, size_t& nextTier, Direction& direction)
{
	vector<vector<Layer*> >& layerTiers = m_neuralNet->GetLayerTiers();

	// In case where layers with data parallelism are input to layers with model parallelism,
	// we need to propagate multiple times through layers with model parallelism.
	if (currTier < layerTiers.size() - 1 && layerTiers[currTier][0]->GetParallelismMode() == ParallelismMode::Data &&
		layerTiers[currTier + 1][0]->GetParallelismMode() == ParallelismMode::Model)
	{
		if (layerTiers[currTier + 1][0]->GetInputLayerIndexInTier() < (int)layerTiers[currTier].size() - 1)
		{
			// We have to propagate again through previous tiers since not all data has been propagated through them.
			nextTier = currTier + 1;
			direction = FORWARD;
			m_neuralNet->GetOutputLayer()->MoveLabelsOffset();

			// Start loading gradients into layer whose activities we propagated on in model layers.
			Layer* propagatedLayer = layerTiers[currTier][layerTiers[currTier + 1][0]->GetInputLayerIndexInTier()];
			m_gradientsToDataLayerLoaded = false;
			m_gradientsToDataLayerLoadThread = thread([this, propagatedLayer] { this->LoadGradientsToLayer(propagatedLayer); });

			// Update parameters in previous tiers.
			UpdateTiersParameters(currEpoch, nextTier, layerTiers.size() - 1);

			return;
		}
		else
		{
			// All data has been propagated through previous tiers, clearing their track record.
			for (Layer* layer : layerTiers[currTier + 1])
			{
				layer->ResetInputLayerIndexInTier();
			}
		}
	}

	// Make splits.
	vector<vector<Layer*> > layerSplits = CreateLayerSplits(currTier, nextTier, -1, [](size_t tier) { return tier > 0; });

	// Propagate on split.
	vector<thread> propagationThreads;
	for (size_t split = 0; split < layerSplits.size(); ++split)
	{
		propagationThreads.push_back(thread([this, layerSplits, split] { this->BackwardPropagateLayers(layerSplits[split]); }));
	}
	for (size_t split = 0; split < layerSplits.size(); ++split)
	{
		propagationThreads[split].join();
	}
}

void Trainer::TrainBatch(uint currEpoch)
{
	vector<vector<Layer*> >& layerTiers = m_neuralNet->GetLayerTiers();
	Direction direction = FORWARD;
	size_t currTier = 1;
	while (currTier != 0)
	{
		size_t nextTier;

		// Propagate.
		if (direction == FORWARD)
		{
			PropagateBatchForward(currTier, nextTier, PropagationMode::Train);
		}
		else
		{
			PropagateBatchBackward(currEpoch, currTier, nextTier, direction);
		}

		// Move on to next tier.
		if (nextTier == layerTiers.size())
		{
			currTier = layerTiers.size() - 1;
			direction = BACKWARD;
		}
		else
		{
			currTier = nextTier;
		}
	}

	// Update parameters in all tiers.
	UpdateTiersParameters(currEpoch, 0, layerTiers.size() - 1);
}

void Trainer::PrintResults(uint percentDone, size_t dataCount, PropagationMode propagationMode)
{
	if (percentDone < 100)
	{
		s_consoleHelper.SetConsoleForeground(ConsoleForeground::GRAY);
	}
	else
	{
		s_consoleHelper.SetConsoleForeground(ConsoleForeground::GREEN);
	}

	cout << percentDone << "% done, results:   ";
	if (percentDone < 100)
	{
		cout << " ";
	}
	if (propagationMode == PropagationMode::Train)
	{
		cout << "Loss: " << m_loss / dataCount << "  ";
	}
	cout << "Accuracy: " << m_accuracy / dataCount;
	if (m_neuralNet->GetOutputLayer()->ShouldCalculateMultipleGuessAccuracy())
	{
		cout << "  Multiple guess accuracy: " << m_multipleGuessAccuracy / dataCount;
	}
	if (percentDone == 100)
	{
		cout << "    [" << GetCurrentTimeStamp() << "]";
	}
	cout << endl;
}

void Trainer::TrainNetwork()
{
	s_consoleHelper.SetConsoleForeground(ConsoleForeground::WHITE);
	cout << endl;
	cout << "**********************************************************************************************************************************" << endl;
	if (m_loadFromCheckpoint)
	{
		cout << "    Network training from checkpoint started  [" << GetCurrentTimeStamp() << "]" << endl;
		cout << "    Last results:   Loss: " << m_loss << "  Accuracy: " << m_accuracy;
		if (m_neuralNet->GetOutputLayer()->ShouldCalculateMultipleGuessAccuracy())
		{
			cout << "  Multiple guess accuracy: " << m_multipleGuessAccuracy;
		}
		cout << endl;
	}
	else
	{
		cout << "    Network training started  [" << GetCurrentTimeStamp() << "]" << endl;
	}
	cout << "**********************************************************************************************************************************" << endl;

	vector<string> dataFiles;
	vector<uint> dataLabels;
	vector<string> nextDataFiles;
	vector<uint> nextDataLabels;
	thread mainTrainThread;
	InputLayer* inputLayer = m_neuralNet->GetInputLayer();
	OutputLayer* outputLayer = m_neuralNet->GetOutputLayer();

	for (uint currEpoch = m_startEpoch; currEpoch <= m_numEpochs; ++currEpoch)
	{
		s_consoleHelper.SetConsoleForeground(ConsoleForeground::DARKCYAN);
		cout << endl << "Training epoch " << currEpoch << endl;
		cout << "------------------------------------------------------------------------------------------------" << endl;

		m_loss = 0.f;
		m_accuracy = 0.f;
		m_multipleGuessAccuracy = 0.f;
		size_t dataTrainedCount = 0;
		uint percentDone = 0;
		uint percentStep = 10;

		// Load first batch of data.
		for (size_t dataIndex = 0; dataIndex < min(m_inputBatchSize, m_trainData.size()); ++dataIndex)
		{
			dataFiles.push_back(m_trainData[dataIndex].first);
			dataLabels.push_back(m_trainData[dataIndex].second);
		}
		m_allFirstTierLayersFpropped = true;
		LoadBatch(dataFiles, PropagationMode::Train);

		// Train on data, batch per batch.
		for (size_t dataIndex = m_inputBatchSize; dataIndex < m_trainData.size(); ++dataIndex)
		{
			nextDataFiles.push_back(m_trainData[dataIndex].first);
			nextDataLabels.push_back(m_trainData[dataIndex].second);

			if ((dataIndex + 1 ) % m_inputBatchSize == 0 || dataIndex == m_trainData.size() - 1)
			{
				// Upload data labels to output layer.
				outputLayer->LoadDataLabels(dataLabels);

				// Run training on current batch.
				m_allFirstTierLayersFpropped = false;
				mainTrainThread = thread([this, currEpoch] { this->TrainBatch(currEpoch); });
				dataTrainedCount += m_inputBatchSize;

				dataFiles = nextDataFiles;
				dataLabels = nextDataLabels;
				// Run preloading of next batch, if it's not last.
				if (dataIndex < m_trainData.size() - 1)
				{
					LoadBatch(dataFiles, PropagationMode::Train);
					nextDataFiles.clear();
					nextDataLabels.clear();
				}

				mainTrainThread.join();

				// Printing results.				
				if (dataTrainedCount > (percentDone + percentStep) / 100.f * m_trainData.size())
				{
					percentDone += percentStep;
					PrintResults(percentDone, dataTrainedCount, PropagationMode::Train);
				}
			}
		}

		// Train on last batch, if there are enough data loaded.
		if (dataFiles.size() >= m_neuralNet->GetMaxNetworkTierSize())
		{
			// We need to have equal number of data trained in each split in case of data parallelism.
			size_t numDataToExclude = dataFiles.size() % m_neuralNet->GetMaxNetworkTierSize();
			for (size_t i = 0; i < numDataToExclude; ++i)
			{
				dataFiles.pop_back();
				dataLabels.pop_back();
			}

			outputLayer->LoadDataLabels(dataLabels);
			LoadBatch(dataFiles, PropagationMode::Train);
			TrainBatch(currEpoch);
			dataTrainedCount += dataFiles.size();
		}
		dataFiles.clear();
		dataLabels.clear();
		nextDataFiles.clear();
		nextDataLabels.clear();

		PrintResults(100, dataTrainedCount, PropagationMode::Train);

		SaveCheckpoint(currEpoch, dataTrainedCount, currEpoch == m_numEpochs);
	}

	cout << endl << endl;
}

void Trainer::TestBatch()
{
	vector<vector<Layer*> >& layerTiers = m_neuralNet->GetLayerTiers();
	size_t currTier = 1;
	while (currTier != layerTiers.size())
	{
		size_t nextTier;

		// Propagate.
		PropagateBatchForward(currTier, nextTier, PropagationMode::Test);

		// Move on to next tier.
		currTier = nextTier;
	}
}

void Trainer::TestNetwork()
{
	s_consoleHelper.SetConsoleForeground(ConsoleForeground::WHITE);
	cout << endl;
	cout << "**********************************************************************************************************************************" << endl;
	cout << "    Network testing started  [" << GetCurrentTimeStamp() << "]" << endl;
	cout << "**********************************************************************************************************************************" << endl;

	vector<string> dataFiles;
	vector<uint> dataLabels;
	vector<string> nextDataFiles;
	vector<uint> nextDataLabels;
	thread mainTrainThread;
	InputLayer* inputLayer = m_neuralNet->GetInputLayer();
	OutputLayer* outputLayer = m_neuralNet->GetOutputLayer();
	bool calculateMultipleGuessAccuracy = outputLayer->ShouldCalculateMultipleGuessAccuracy();

	m_loss = 0.f;
	m_accuracy = 0.f;
	m_multipleGuessAccuracy = 0.f;
	size_t dataTestedCount = 0;
	uint percentDone = 0;
	uint percentStep = 10;
	uint numTestPasses = inputLayer->GetNumTestPasses();

	// Load first batch of data.
	size_t batchSize = (size_t)m_batchSize;
	for (size_t dataIndex = 0; dataIndex < min(batchSize, m_testData.size()); ++dataIndex)
	{
		dataFiles.push_back(m_testData[dataIndex].first);
		dataLabels.push_back(m_testData[dataIndex].second);
	}
	m_allFirstTierLayersFpropped = true;
	LoadBatch(dataFiles, PropagationMode::Test);

	// Test on data, batch per batch.
	for (size_t dataIndex = batchSize; dataIndex < m_testData.size(); ++dataIndex)
	{
		nextDataFiles.push_back(m_testData[dataIndex].first);
		nextDataLabels.push_back(m_testData[dataIndex].second);
		if ((dataIndex + 1) % batchSize == 0 || dataIndex == m_testData.size() - 1)
		{
			// Upload data labels to output layer.
			outputLayer->LoadDataLabels(dataLabels);
			dataTestedCount += batchSize;

			for (uint testPass = 0; testPass < numTestPasses; ++testPass)
			{
				// Run testing on current batch.
				m_allFirstTierLayersFpropped = false;
				mainTrainThread = thread([this] { this->TestBatch(); });

				if (testPass < numTestPasses - 1)
				{
					LoadBatch(dataFiles, PropagationMode::Test);
				}
				else
				{
					dataFiles = nextDataFiles;
					dataLabels = nextDataLabels;

					// Run preloading of next batch, if if it's not last.
					if (dataIndex < m_testData.size() - 1)
					{
						LoadBatch(dataFiles, PropagationMode::Test);
						nextDataFiles.clear();
						nextDataLabels.clear();
					}
				}				

				mainTrainThread.join();
			}

			// Gathering batch results.
			m_loss += outputLayer->GetLoss();
			m_accuracy += outputLayer->GetAccuracy();
			if (calculateMultipleGuessAccuracy)
			{
				m_multipleGuessAccuracy += outputLayer->GetMultipleGuessAccuracy();
			}
			if (dataTestedCount > (percentDone + percentStep) / 100.f * m_testData.size())
			{
				percentDone += percentStep;
				PrintResults(percentDone, dataTestedCount, PropagationMode::Test);
			}
		}
	}

	// Test on last batch, if there are enough data loaded.
	if (dataFiles.size() >= m_neuralNet->GetMaxNetworkTierSize())
	{
		outputLayer->LoadDataLabels(dataLabels);
		LoadBatch(dataFiles, PropagationMode::Test);
		
		for (uint testPass = 0; testPass < numTestPasses; ++testPass)
		{
			// Run testing on current batch.
			m_allFirstTierLayersFpropped = false;
			mainTrainThread = thread([this] { this->TestBatch(); });

			if (testPass < numTestPasses - 1)
			{
				LoadBatch(dataFiles, PropagationMode::Test);
			}
			
			mainTrainThread.join();
		}

		dataTestedCount += dataFiles.size();
		m_loss += outputLayer->GetLoss();
		m_accuracy += outputLayer->GetAccuracy();
		if (calculateMultipleGuessAccuracy)
		{
			m_multipleGuessAccuracy += outputLayer->GetMultipleGuessAccuracy();
		}
	}

	PrintResults(100, dataTestedCount, PropagationMode::Test);
	cout << endl << endl;

	// Saving test results.
	ofstream resultsFile(m_workFolder + "\\" + c_resultsFileName, ios::app);
	resultsFile << "Test results:    Accuracy: " << m_accuracy / dataTestedCount;
	if (calculateMultipleGuessAccuracy)
	{
		resultsFile << "  Multiple guess accuracy: " << m_multipleGuessAccuracy / dataTestedCount;
	}
	resultsFile << endl;
	resultsFile.close();
}

void Trainer::ResetDevices()
{
	int numGpus;
	CudaAssert(cudaGetDeviceCount(&numGpus));
	for (int i = 0; i < numGpus; ++i)
	{
		CudaAssert(cudaSetDevice(i));
		CudaAssert(cudaDeviceReset());
	}

	CudaAssert(cudaSetDevice(0));
}

void Trainer::RunTraining()
{
	// Creating work folder.
	ShipAssert(_mkdir(m_workFolder.c_str()) == 0 || errno == EEXIST, "Problem creating work directory \"" + m_workFolder + "\".");

	// Training network.
	SetDefaultDevice();
	InitializeNetwork(ParsingMode::Training);
	if (m_loadFromCheckpoint)
	{
		LoadCheckpoint();
	}
	InitializeTrainer();
	ValidateConfiguration();
	LoadData();
	LoadDataInfo();
	TrainNetwork();
	
	if (!m_noTest)
	{
		// Reseting devices and network for testing.
		delete m_neuralNet;
		ResetDevices();

		// Testing network.
		InitializeNetwork(ParsingMode::Prediction);
		m_neuralNet->LoadModelForPrediction(m_workFolder + "\\" + c_predictionModelName);
		LoadDataInfo();
		TestNetwork();
	}
}