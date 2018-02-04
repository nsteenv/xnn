// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network features extractor.
// Created: 04/03/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/featurizer.cuh"

const string Featurizer::c_configurationSignature = "-configfile";
const string Featurizer::c_dataFolderSignature = "-datafolder";
const string Featurizer::c_modelFileSignature = "-model";
const string Featurizer::c_batchSizeSignature = "-batchsize";
const string Featurizer::c_targetLayerSignature = "-layer";

Featurizer::Featurizer()
{
	m_neuralNet = NULL;
	m_hostTargetLayerActivations = NULL;
}

Featurizer::~Featurizer()
{
	if (m_neuralNet != NULL)
	{
		delete m_neuralNet;
	}
	if (m_hostTargetLayerActivations != NULL)
	{
		CudaAssert(cudaFreeHost(m_hostTargetLayerActivations));
	}
}

bool Featurizer::ParseArguments(int argc, char *argv[])
{
	uint targetLayer;
	if (!ParseArgument(argc, argv, c_configurationSignature, m_networkConfigurationFile) ||
		!ParseArgument(argc, argv, c_modelFileSignature, m_modelFile) ||
		!ParseArgument(argc, argv, c_dataFolderSignature, m_dataFolder) ||
		!ParseArgument(argc, argv, c_batchSizeSignature, m_batchSize) ||
		!ParseArgument(argc, argv, c_targetLayerSignature, targetLayer))
	{
		return false;
	}

	m_targetLayerIndex = targetLayer;

	return true;
}

void Featurizer::InitializeNetwork()
{
	m_neuralNet = m_configurationParser.ParseNetworkFromConfiguration(ParsingMode::Prediction, m_networkConfigurationFile, m_dataFolder, m_batchSize, false);
	m_neuralNet->LoadModelForPrediction(m_modelFile);
}

void Featurizer::LoadData()
{
	// Loading data.
	ifstream dataList(m_dataFolder + "\\datalist.txt");
	string fileName;
	while (getline(dataList, fileName))
	{
		m_data.push_back(fileName);
	}
	
	// Load data info.
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

void Featurizer::LoadBatch(const vector<string>& dataFiles)
{
	InputLayer* inputLayer = m_neuralNet->GetInputLayer();

	inputLayer->SetDataFilesToLoad(dataFiles, PropagationMode::Featurization);
	
	// Load data from disk to host memory.
	inputLayer->LoadInputs();

	// Load data from host to GPU memory.
	inputLayer->DoForwardProp(PropagationMode::Featurization);
}

void Featurizer::ExtractFeaturesOnBatch()
{
	vector<vector<Layer*> >& layerTiers = m_neuralNet->GetLayerTiers();
	for (size_t currTier = 1; currTier <= m_targetLayerIndex; ++currTier)
	{
		Layer* layer = layerTiers[currTier][0];
		
		layer->LoadInputs();
		if (layer->HoldsInputData())
		{
			// Making sure inputs are loaded before computation.
			layer->SynchronizeMemoryOperations();
		}

		layer->DoForwardProp(PropagationMode::Featurization);
	}

	Layer* targetLayer = layerTiers[m_targetLayerIndex][0];
	targetLayer->SynchronizeCalculations();

	// Copy results to host.
	if (m_hostTargetLayerActivations == NULL)
	{
		CudaAssert(cudaMallocHost<float>(&m_hostTargetLayerActivations, targetLayer->GetActivationBufferSize(), cudaHostAllocPortable));
	}
	CudaAssert(cudaMemcpy(m_hostTargetLayerActivations, targetLayer->GetActivationDataBuffer(), targetLayer->GetActivationBufferSize(), cudaMemcpyDeviceToHost));

	// Append results to file.
	ofstream featuresFile(m_dataFolder + "\\" + "features.txt", ios::app);
	uint dataCount = targetLayer->GetActivationDataCount();
	uint activationBufferLength = (uint)(targetLayer->GetActivationBufferSize() / sizeof(float));
	for (uint i = 0; i < dataCount; ++i)
	{
		featuresFile << m_hostTargetLayerActivations[i];
		for (uint j = i + dataCount; j < activationBufferLength; j += dataCount)
		{
			featuresFile << " " << m_hostTargetLayerActivations[j];
		}
		featuresFile << endl;
	}
}

void Featurizer::ExtractFeatures()
{
	s_consoleHelper.SetConsoleForeground(ConsoleForeground::WHITE);
	cout << endl;
	cout << "**********************************************************************************************************************************" << endl;
	cout << "    Features extraction started  [" << GetCurrentTimeStamp() << "]" << endl;
	cout << "**********************************************************************************************************************************" << endl;

	vector<string> dataFiles;
	InputLayer* inputLayer = m_neuralNet->GetInputLayer();
	size_t batchSize = (size_t)m_batchSize;
	size_t stepSize = 10;
	size_t currStep = 1;
	size_t progress = 0;

	// Run features extraction on data, batch per batch.
	s_consoleHelper.SetConsoleForeground(ConsoleForeground::GRAY);
	for (size_t dataIndex = 0; dataIndex < m_data.size(); ++dataIndex)
	{
		dataFiles.push_back(m_data[dataIndex]);
		if ((dataIndex + 1) % batchSize == 0 || dataIndex == m_data.size() - 1)
		{
			// Load data for current batch.
			LoadBatch(dataFiles);

			// Run extraction on current batch.
			ExtractFeaturesOnBatch();

			dataFiles.clear();
		}

		if (currStep * m_data.size() / stepSize < dataIndex)
		{
			++currStep;
			progress += stepSize;
			cout << "Done: " << progress << "%" << endl;
		}
	}

	s_consoleHelper.SetConsoleForeground(ConsoleForeground::GREEN);
	cout << endl << "Features extraction finished!" << endl << endl << endl;
}

void Featurizer::RunExtraction()
{
	InitializeNetwork();
	LoadData();
	ExtractFeatures();
}