// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Prepares data for training or featurization.
// Created: 11/24/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <chrono>
#include <direct.h>
#include <fstream>
#include <thread>
#include <vector>

#include "../../dataparsers/include/dataparserfactory.cuh"
#include "../../utils/include/config.cuh"

using namespace std;
using namespace std::chrono;

// Data making modes.
enum class DataMakerMode
{
	Training,
	Featurization,
	DatasetExtension
};

class DataMaker
{
private:
	// Input folder path.
	string m_inputFolder;

	// Input data list file path.
	string m_inputDataListFile;

	// Output folder path.
	string m_outputFolder;

	// Size of output images.
	uint m_imageSize;

	// Number of image channels.
	uint m_numImageChannels;

	// Device buffers for calculating mean image.
	vector<uint*> m_deviceMeanImageBuffers;

	// Device buffers length.
	uint m_deviceMeanImageBufferLength;

	// Count of images applied to calculate each mean image.
	vector<uint> m_meanImagesAppliedCounts;

	// Mutex used for output from multiple worker threads.
	static mutex s_outputMutex;

	// Initializes data maker.
	void Initialize(DataMakerMode dataMakerMode);

	// Prepares part of data.
	void MakeDataPart(const vector<string>& data, string folder, int partIndex, DataMakerMode dataMakerMode);

	// Handles image for training purposes.
	void HandleImageForTraining(ImageData* image, string imageName, DataParser* dataParser, cudaStream_t stream, int partIndex, string folder);

	// Handles image for featurization purposes.
	void HandleImageForFeaturization(ImageData* image, string imageName, DataParser* dataParser, cudaStream_t stream);

	// Handles image for dataset extension purposes.
	void HandleImageForDatasetExtension(ImageData* image, string imageName, DataParser* dataParser, cudaStream_t stream);

	// Prepares data in certain folder.
	void MakeData(string folder, DataMakerMode dataMakerMode);

	// Makes data info file.
	void MakeDataInfo();

	// Prints timing of certain operation.
	void PrintTiming(string operationMessage, high_resolution_clock::time_point operationStartTime,
		high_resolution_clock::time_point operationEndTime);

public:
	// Argument parameters signatures.
	static const string c_inputFolderSignature;
	static const string c_inputDataListSignature;
	static const string c_outputFolderSignature;
	static const string c_imageSizeSignature;
	static const string c_numImageChannelsSignature;

	// Parameters default values.
	static const uint c_defaultImageSize;
	static const uint c_defaultNumOfImageChannels;

	// Labels file name.
	static const string c_labelsFileName;

	// Data info file name.
	static const string c_dataInfoFileName;

	// Destructor.
	~DataMaker();

	// Parses arguments for data making.
	bool ParseArguments(int argc, char *argv[]);

	// Prepares data for training.
	void MakeDataForTraining();

	// Prepares data for featurization.
	void MakeDataForFeaturization();

	// Makes extended dataset by applying crops and flips.
	void MakeExtendedDataset();
};