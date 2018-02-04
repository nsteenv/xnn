// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Neural network input layer.
// Created: 12/30/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <sstream>
#include <thread>

#include "layer.cuh"
#include "../../../dataparsers/include/dataparserfactory.cuh"
#include "../../../utils/include/config.cuh"

using namespace std;

// Data types.
enum class DataType
{
	Image,
	Text
};

/*
	Input layer loads data for the network training.
*/
class InputLayer : public Layer
{
private:
	// Device streams for memory operations.
	vector<cudaStream_t> m_deviceMemoryStreams;

	// Folder with data for training.
	string m_dataFolder;

	// Data type.
	DataType m_dataType;

	// Width of data samples.
	uint m_dataWidth;

	// Height of data samples.
	uint m_dataHeight;

	// Seed engine for generating random crop horizontal positions.
	default_random_engine m_cropPositionXGenerator;

	// Random distribution for generating random crop horizontal positions.
	uniform_int_distribution<uint> m_cropPositionXDistribution;

	// Seed engine for generating random crop vertical positions.
	default_random_engine m_cropPositionYGenerator;

	// Random distribution for generating random crop vertical positions.
	uniform_int_distribution<uint> m_cropPositionYDistribution;

	// Seed engine for generating random crop flip decisions.
	default_random_engine m_cropFlipGenerator;

	// Random distribution for generating random crop flip decisions.
	uniform_int_distribution<uint> m_cropFlipDistribution;

	// Data files to load.
	vector<string> m_dataFilesToLoad;

	// Number of inputs to generate, depending on number of layers in next tier and their parallelism.
	uint m_numInputs;

	// Propagation mode.
	PropagationMode m_propagationMode;

	// Mean values of data per channel.
	vector<uchar> m_channelMeanValues;

	// Activation data buffers, one buffer for each of layers in next tier.
	vector<float*> m_activationDataBuffers;

	// Activation data counts, one count for each of layers in next tier.
	vector<uint> m_activationDataCounts;

	// Number of patches we extract from the image during test to get average prediction.
	uint m_numTestPatches;

	// Should we also test on flipped versions of patches.
	bool m_testOnFlips;

	// Number of test passes for one image.
	uint m_numTestPasses;

	// Counter of test passes.
	uint m_testPassCounter;

	// Calculates patch position.
	void CalculatePatch(uint& cropPositionX, uint numPatchesX, uint patchX, uint& cropPositionY, uint numPatchesY, uint patchY);

	// Calculates position from which to crop patch for test pass.
	void CalculateTestPatchPosition(uint& cropPositionX, uint& cropPositionY, bool& flip);

	// Setups data positions for load.
	void SetupDataPositions(int partIndex, size_t inputIndex, size_t& startIndex, size_t& endIndex, float** inputDataBuffer, vector<string>& dataFilesToLoad);

	// Loads part of input image files to input data buffer which position depends of how many inputs we are loading.
	void LoadImageInputsPart(int partIndex, size_t inputIndex, uint cropPositionX, uint cropPositionY, bool flip);

	// Loads part of input text files to input data buffer which position depends of how many inputs we are loading.
	void LoadTextInputsPart(int partIndex, size_t inputIndex);

public:
	// Constructor.
	InputLayer(string dataFolder, DataType dataType, vector<cudaStream_t> deviceMemoryStreams, uint inputNumChannels, uint inputDataWidth,
		uint inputDataHeight, uint inputDataCount, uint dataWidth, uint dataHeight, uint numInputs, uint numTestPatches, bool testOnFlips);

	// Destructor.
	virtual ~InputLayer();

	// Gets input data type.
	DataType GetDataType() const { return m_dataType; }

	// Sets data files to load.
	void SetDataFilesToLoad(const vector<string>& dataFiles, PropagationMode propagationMode);

	// Sets mean values of data per channel.
	void SetChannelMeanValues(vector<uchar> values) { m_channelMeanValues = values; }

	// Gets activation data buffer, dedicated to layer with specified index in next tier.
	float* GetActivationDataBuffer(uint indexInTier) { return m_activationDataBuffers[indexInTier]; }

	// Gets input data count.
	uint GetInputDataCount() const { return m_inputDataCount; }

	// Gets activation data counts, dedicated to layer with specified index in next tier.
	uint GetActivationDataCount(uint indexInTier) { return m_activationDataCounts[indexInTier]; }

	// Gets number of test passes for one image.
	uint GetNumTestPasses() const { return m_numTestPasses; }

	// Loads input to layer.
	virtual void LoadInputs();

	// Does forward propagation through layer.
	virtual void DoForwardProp(PropagationMode propagationMode);

	// Does backward propagation through layer.
	virtual void DoBackwardProp();
};