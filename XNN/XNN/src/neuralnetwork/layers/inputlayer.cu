// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network input layer.
// Created: 12/30/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/inputlayer.cuh"

InputLayer::InputLayer(string dataFolder, DataType dataType, vector<cudaStream_t> deviceMemoryStreams, uint inputNumChannels, uint inputDataWidth,
	uint inputDataHeight, uint inputDataCount, uint dataWidth, uint dataHeight, uint numInputs, uint numTestPatches, bool testOnFlips)
{
	m_layerType = LayerType::Input;
	m_parallelismMode = ParallelismMode::Model;
	m_dataFolder = dataFolder;
	m_dataType = dataType;
	m_deviceMemoryStreams = deviceMemoryStreams;
	m_indexInTier = 0;
	m_tierSize = 1;
	m_numInputs = numInputs;

	m_numTestPatches = numTestPatches;
	m_testOnFlips = testOnFlips;
	m_numTestPasses = m_numTestPatches * (m_testOnFlips ? 2 : 1);
	m_testPassCounter = 0;

	m_inputNumChannels = m_activationNumChannels = inputNumChannels;
	m_inputDataWidth = m_activationDataWidth = inputDataWidth;
	m_inputDataHeight = m_activationDataHeight = inputDataHeight;
	m_inputDataSize = m_activationDataSize = m_inputDataWidth * m_inputDataHeight;
	m_inputDataCount = inputDataCount;
	m_dataWidth = dataWidth;
	m_dataHeight = dataHeight;
	m_holdsInputData = true;

	if (dataType == DataType::Image)
	{
		m_cropPositionXGenerator = default_random_engine((uint)chrono::system_clock::now().time_since_epoch().count());
		m_cropPositionXDistribution = uniform_int_distribution<uint>(0, m_dataWidth - m_inputDataWidth);
		m_cropPositionYGenerator = default_random_engine((uint)chrono::system_clock::now().time_since_epoch().count());
		m_cropPositionYDistribution = uniform_int_distribution<uint>(0, m_dataHeight - m_inputDataHeight);
		m_cropFlipGenerator = default_random_engine((uint)chrono::system_clock::now().time_since_epoch().count());
		m_cropFlipDistribution = uniform_int_distribution<uint>(1, 100);
	}

	// Allocating input data buffer.
	m_inputBufferSize = m_inputNumChannels * m_inputDataSize * m_inputDataCount * sizeof(float);
	CudaAssert(cudaMallocHost<float>(&m_inputDataBuffer, m_inputBufferSize, cudaHostAllocPortable));

	// Allocating activation data buffers.
	uint dataPerInput = m_inputDataCount / (uint)numInputs;
	for (uint i = 0; i < m_numInputs; ++i)
	{
		m_activationDataCounts.push_back(dataPerInput);
		CudaAssert(cudaSetDevice((int)i));
		float* activationDataBuffer;
		CudaAssert(cudaMalloc<float>(&activationDataBuffer, dataPerInput * m_inputNumChannels * m_inputDataSize * sizeof(float)));
		m_activationDataBuffers.push_back(activationDataBuffer);
	}
	// Reverting device back to 0.
	CudaAssert(cudaSetDevice(0));

	m_holdsActivationGradients = false;
}

InputLayer::~InputLayer()
{
	CudaAssert(cudaFreeHost(m_inputDataBuffer));
	m_inputDataBuffer = NULL;

	for (size_t i = 0; i < m_activationDataBuffers.size(); ++i)
	{
		CudaAssert(cudaFree(m_activationDataBuffers[i]));
	}
	m_activationDataBuffer = NULL;
}

void InputLayer::SetDataFilesToLoad(const vector<string>& dataFiles, PropagationMode propagationMode)
{
	m_dataFilesToLoad = dataFiles;
	m_propagationMode = propagationMode;
}

void InputLayer::CalculatePatch(uint& cropPositionX, uint numPatchesX, uint patchX, uint& cropPositionY, uint numPatchesY, uint patchY)
{
	cropPositionX = (patchX - 1) * (m_dataWidth - m_inputDataWidth) / (numPatchesX - 1);
	cropPositionY = (patchY - 1) * (m_dataHeight - m_inputDataHeight) / (numPatchesY - 1);
}

void InputLayer::CalculateTestPatchPosition(uint& cropPositionX, uint& cropPositionY, bool& flip)
{
	++m_testPassCounter;

	flip = false;
	if (m_testOnFlips)
	{
		flip = m_testPassCounter > m_numTestPatches;
	}

	uint pass = m_testPassCounter;
	if (flip)
	{
		pass -= m_numTestPatches;
	}

	if (m_numTestPatches == 1)
	{
		CalculatePatch(cropPositionX, 3, 2, cropPositionY, 3, 2);
	}
	else if (m_numTestPatches == 2)
	{
		CalculatePatch(cropPositionX, 2, pass, cropPositionY, 3, 2);
	}
	else if (m_numTestPatches == 3)
	{
		CalculatePatch(cropPositionX, 3, pass, cropPositionY, 3, 2);
	}
	else if (m_numTestPatches == 4)
	{
		if (pass <= 2)
		{
			CalculatePatch(cropPositionX, 2, pass, cropPositionY, 2, 1);
		}
		else
		{
			CalculatePatch(cropPositionX, 2, pass - 2, cropPositionY, 2, 2);
		}
	}
	else if (m_numTestPatches == 5)
	{
		if (pass <= 2)
		{
			CalculatePatch(cropPositionX, 2, pass, cropPositionY, 2, 1);
		}
		else if (pass <= 4)
		{
			CalculatePatch(cropPositionX, 2, pass - 2, cropPositionY, 2, 2);
		}
		else
		{
			CalculatePatch(cropPositionX, 3, 2, cropPositionY, 3, 2);
		}
	}
	else if (m_numTestPatches == 6)
	{
		if (pass <= 3)
		{
			CalculatePatch(cropPositionX, 3, pass, cropPositionY, 2, 1);
		}
		else
		{
			CalculatePatch(cropPositionX, 3, pass - 3, cropPositionY, 2, 2);
		}
	}
	else if (m_numTestPatches == 7)
	{
		if (pass <= 3)
		{
			CalculatePatch(cropPositionX, 3, pass, cropPositionY, 2, 1);
		}
		else if (pass <= 6)
		{
			CalculatePatch(cropPositionX, 3, pass - 3, cropPositionY, 2, 2);
		}
		else
		{
			CalculatePatch(cropPositionX, 3, 2, cropPositionY, 3, 2);
		}
	}
	else if (m_numTestPatches == 8)
	{
		if (pass <= 4)
		{
			CalculatePatch(cropPositionX, 4, pass, cropPositionY, 2, 1);
		}
		else
		{
			CalculatePatch(cropPositionX, 4, pass - 4, cropPositionY, 2, 2);
		}
	}
	else if (m_numTestPatches == 9)
	{
		if (pass <= 3)
		{
			CalculatePatch(cropPositionX, 3, pass, cropPositionY, 3, 1);
		}
		else if (pass <= 6)
		{
			CalculatePatch(cropPositionX, 3, pass - 3, cropPositionY, 3, 2);
		}
		else
		{
			CalculatePatch(cropPositionX, 3, pass - 6, cropPositionY, 3, 3);
		}
	}
	else
	{
		ShipAssert(false, "Currently not supported!");
	}

	if (m_testPassCounter == m_numTestPasses)
	{
		m_testPassCounter = 0;
	}
}

void InputLayer::SetupDataPositions(int partIndex, size_t inputIndex, size_t& startIndex, size_t& endIndex, float** inputDataBuffer, vector<string>& dataFilesToLoad)
{
	DebugAssert(partIndex < Config::NUM_DATA_LOAD_CPU_CORES, "There is more data parts than CPU cores!");

	// Setting up data positions.
	auto beginIterator = m_dataFilesToLoad.begin();
	for (size_t i = 0; i < inputIndex; ++i)
	{
		beginIterator += m_activationDataCounts[i];
	}
	auto endIterator = beginIterator + m_activationDataCounts[inputIndex];
	dataFilesToLoad = vector<string>(beginIterator, endIterator);
	size_t dataPerCore = dataFilesToLoad.size() / Config::NUM_DATA_LOAD_CPU_CORES;
	startIndex = partIndex * dataPerCore;
	endIndex = partIndex + 1 < Config::NUM_DATA_LOAD_CPU_CORES ? (partIndex + 1) * dataPerCore : dataFilesToLoad.size();
	*inputDataBuffer = m_inputDataBuffer + (beginIterator - m_dataFilesToLoad.begin()) * m_inputNumChannels * m_inputDataSize;
}

void InputLayer::LoadImageInputsPart(int partIndex, size_t inputIndex, uint cropPositionX, uint cropPositionY, bool flip)
{
	// Setting data positions.
	size_t startIndex;
	size_t endIndex;
	float* inputDataBuffer;
	vector<string> dataFilesToLoad;
	SetupDataPositions(partIndex, inputIndex, startIndex, endIndex, &inputDataBuffer, dataFilesToLoad);

	// Preparing data.
	string dataFolder = m_dataFolder + "\\";
	if (m_propagationMode == PropagationMode::Train)
	{
		dataFolder += "train\\";
	}
	else if (m_propagationMode == PropagationMode::Test)
	{
		dataFolder += "test\\";
	}
	DataParserFactory dataParserFactory;
	DataParser* dataParser;
	ImageData* image;
	ImageData* inputImage;
	for (size_t i = startIndex; i < endIndex; ++i)
	{
		// Finding file extension.
		string dataExtension = GetExtension(dataFilesToLoad[i]);
		if (dataExtension == "")
		{
			cout << endl << "Encountered data without extension! File name: " << dataFilesToLoad[i] << endl;
			continue;
		}

		// Finding appropriate parser for that extension.
		dataParser = dataParserFactory.GetDataParser(dataExtension);

		// Parsing image.
		image = dataParser->LoadImage(dataFolder + dataFilesToLoad[i]);
		ShipAssert(image->GetNumOfChannels() == m_inputNumChannels, "Encountered image with invalid number of channels! File name: " + dataFilesToLoad[i]);
		
		// Cropping image.
		inputImage = dataParser->CropImage(*image, cropPositionX, cropPositionY, m_inputDataWidth, m_inputDataHeight, flip);
		
		// Copying image data to buffer.
		uchar* croppedImageRowMajorPixels = inputImage->GetRowMajorPixels();
		uint totalPixelsPerChannel = m_activationDataCounts[inputIndex] * m_inputDataSize;
		size_t inputImageBufferLength = inputImage->GetBufferSize() / sizeof(uchar);
		for (size_t pixel = 0; pixel < inputImageBufferLength; ++pixel)
		{
			size_t channel = pixel % 3;
			inputDataBuffer[i + (pixel / m_inputNumChannels) * m_activationDataCounts[inputIndex] + channel * totalPixelsPerChannel] =
				(float)croppedImageRowMajorPixels[pixel] - m_channelMeanValues[channel];
		}

		delete image;
		delete inputImage;
	}
}

void InputLayer::LoadTextInputsPart(int partIndex, size_t inputIndex)
{
	// Setting data positions.
	size_t startIndex;
	size_t endIndex;
	float* inputDataBuffer;
	vector<string> dataFilesToLoad;
	SetupDataPositions(partIndex, inputIndex, startIndex, endIndex, &inputDataBuffer, dataFilesToLoad);

	// Parsing data.
	for (size_t i = startIndex; i < endIndex; ++i)
	{
		istringstream dataParser(dataFilesToLoad[i]);
		float inputFeature;
		for (size_t j = 0; j < m_inputDataWidth; ++j)
		{
			dataParser >> inputFeature;
			inputDataBuffer[i + j * m_activationDataCounts[inputIndex]] = inputFeature;
		}
	}
}

void InputLayer::LoadInputs()
{
	DebugAssert(!m_dataFilesToLoad.empty(), "Data files to load must be set first!");

	// Reinitialize layer if needed.
	if (m_dataFilesToLoad.size() != m_inputDataCount)
	{
		Reinitialize((uint)m_dataFilesToLoad.size());

		uint dataPerInput = m_inputDataCount / m_numInputs;
		for (size_t i = 0; i < m_nextLayers.size(); ++i)
		{
			m_activationDataCounts[i] = dataPerInput;
			if (i < m_inputDataCount % m_numInputs)
			{
				++m_activationDataCounts[i];
			}
		}
	}

	if (m_dataType == DataType::Image)
	{
		uint cropPositionX = 0;
		uint cropPositionY = 0;
		bool flip = false;
		if (m_propagationMode == PropagationMode::Test)
		{
			CalculateTestPatchPosition(cropPositionX, cropPositionY, flip);
		}

		// Loading inputs in parts.
		for (size_t inputIndex = 0; inputIndex < m_numInputs; ++inputIndex)
		{
			vector<thread> dataLoadThreads;
			for (int i = 0; i < Config::NUM_DATA_LOAD_CPU_CORES; ++i)
			{
				if (m_propagationMode == PropagationMode::Train)
				{
					cropPositionX = m_cropPositionXDistribution(m_cropPositionXGenerator);
					cropPositionY = m_cropPositionYDistribution(m_cropPositionYGenerator);
					flip = m_cropFlipDistribution(m_cropFlipGenerator) % 2 == 0;
				}

				dataLoadThreads.push_back(thread([this, i, inputIndex, cropPositionX, cropPositionY, flip]
				{
					this->LoadImageInputsPart(i, inputIndex, cropPositionX, cropPositionY, flip);
				}));
			}
			for (int i = 0; i < Config::NUM_DATA_LOAD_CPU_CORES; ++i)
			{
				dataLoadThreads[i].join();
			}
		}
	}
	else
	{
		// Loading inputs in parts.
		for (size_t inputIndex = 0; inputIndex < m_numInputs; ++inputIndex)
		{
			vector<thread> dataLoadThreads;
			for (int i = 0; i < Config::NUM_DATA_LOAD_CPU_CORES; ++i)
			{
				dataLoadThreads.push_back(thread([this, i, inputIndex]
				{
					this->LoadTextInputsPart(i, inputIndex);
				}));
			}
			for (int i = 0; i < Config::NUM_DATA_LOAD_CPU_CORES; ++i)
			{
				dataLoadThreads[i].join();
			}
		}
	}
}

void InputLayer::DoForwardProp(PropagationMode propagationMode)
{
	// Forwarding loaded input data to each of activation buffers.
	uint inputOffset = 0;
	for (size_t i = 0; i < m_activationDataBuffers.size(); ++i)
	{
		CudaAssert(cudaSetDevice((int)i));
		uint m_activationDataBufferLength = m_activationDataCounts[i] * m_inputNumChannels * m_inputDataSize;
		CudaAssert(cudaMemcpyAsync(m_activationDataBuffers[i], m_inputDataBuffer + inputOffset, m_activationDataBufferLength * sizeof(float),
			cudaMemcpyHostToDevice, m_deviceMemoryStreams[i]));
		CudaAssert(cudaStreamSynchronize(m_deviceMemoryStreams[i]));
		inputOffset += m_activationDataBufferLength;
	}

	// Reverting device back to 0.
	CudaAssert(cudaSetDevice(0));
}

void InputLayer::DoBackwardProp()
{
	ShipAssert(false, "Shouldn't backpropagate on input layer!");
}