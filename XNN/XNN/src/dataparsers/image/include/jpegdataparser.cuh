// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Jpeg data parser.
// Created: 11/26/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#ifdef _WIN32
	#define XMD_H /* This prevents jpeglib to redefine INT32 */
#endif

#include <jpeglib.h>

#include "../../include/dataparser.cuh"
#include "../../../utils/include/config.cuh"

using namespace std;

class JpegDataParser : public DataParser
{
private:
	// Operations source image device buffer.
	uchar* m_deviceOpSrcBuffer;

	// Operations destination image device buffer.
	uchar* m_deviceOpDestBuffer;

	// Size of operations image device buffers.
	size_t m_deviceOpBufferSize;

	// Allocates memory for operations image device buffers if they are not allocated, or their size is less than needed.
	void AllocMemoryIfNeeded(size_t imageBufferSize);

	// Transfers image data to operations device source buffer.
	void TransferImageToOpDeviceBuffer(const ImageData& image, cudaStream_t stream = 0);

	// Calculates resize dimensions.
	void CalculateResizeDimensions(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode,
		NppiSize& srcImageSize, NppiRect& srcImageRect, NppiRect& destImageRect, double& scaleX, double& scaleY);

	// Resizes image from source device buffer to destination device buffer.
	void ResizeOnDeviceBuffers(NppiSize srcImageSize, NppiRect srcImageRect, NppiRect destImageRect, uint numOfChannels,
		double scaleX, double scaleY);

public:
	// Constructor.
	JpegDataParser();

	// Destructor.
	virtual ~JpegDataParser();

	// Loads image from file.
	virtual ImageData* LoadImage(string inputPath);

	// Saves image to file.
	virtual void SaveImage(const ImageData& image, string outputPath);

	// Resizes image to desired dimensions, using CUDA.
	virtual ImageData* ResizeImageCu(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode, cudaStream_t stream = 0);

	// Crops image to desired coordinates and dimensions, with option to flip.
	virtual ImageData* CropImage(const ImageData& image, uint startPixelX, uint startPixelY, uint desiredWidth, uint desiredHeight, bool flipCrop);

	// Resizes image to desired dimensions and crops them to be squared, using CUDA.
	virtual ImageData* ResizeImageWithCropCu(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode,
		CropMode cropMode, cudaStream_t stream = 0);
};