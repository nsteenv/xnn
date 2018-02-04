// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Jpeg data parser.
// Created: 11/26/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/jpegdataparser.cuh"

JpegDataParser::JpegDataParser()
{
	m_deviceOpSrcBuffer = NULL;
	m_deviceOpDestBuffer = NULL;
	// Defaultly we set this to support 16MP images with 3 channels.
	m_deviceOpBufferSize = 3 * 8192 * 8192 * sizeof(uchar);
}

JpegDataParser::~JpegDataParser()
{
	CudaAssert(cudaFree(m_deviceOpSrcBuffer));
	CudaAssert(cudaFree(m_deviceOpDestBuffer));
}

void JpegDataParser::AllocMemoryIfNeeded(size_t imageBufferSize)
{
	if (imageBufferSize > m_deviceOpBufferSize)
	{
		m_deviceOpBufferSize = imageBufferSize;
		CudaAssert(cudaFree(m_deviceOpSrcBuffer));
		m_deviceOpSrcBuffer = NULL;
		CudaAssert(cudaFree(m_deviceOpDestBuffer));
		m_deviceOpDestBuffer = NULL;
	}
	if (m_deviceOpSrcBuffer == NULL)
	{
		CudaAssert(cudaMalloc<uchar>(&m_deviceOpSrcBuffer, m_deviceOpBufferSize));
	}
	if (m_deviceOpDestBuffer == NULL)
	{
		CudaAssert(cudaMalloc<uchar>(&m_deviceOpDestBuffer, m_deviceOpBufferSize));
	}
}

ImageData* JpegDataParser::LoadImage(string inputPath)
{
	// Opening image file.
	FILE *imageFile;
	ShipAssert((fopen_s(&imageFile, inputPath.c_str(), "rb")) == 0, "Cannot open jpeg image file \"" + inputPath + "\".");

	// Parsing image.
	struct jpeg_decompress_struct imageInfo;
	struct jpeg_error_mgr errorHandler;
	imageInfo.err = jpeg_std_error(&errorHandler);
	jpeg_create_decompress(&imageInfo);
	jpeg_stdio_src(&imageInfo, imageFile);
	jpeg_read_header(&imageInfo, TRUE);
	switch (imageInfo.jpeg_color_space)
	{
		case JCS_GRAYSCALE:
		case JCS_RGB:
		case JCS_YCbCr:
			imageInfo.out_color_space = JCS_RGB;
			break;
		case JCS_CMYK:
		case JCS_YCCK:
			imageInfo.out_color_space = JCS_CMYK;
			break;
		default:
			ShipAssert(false, "Unsupported jpeg color format: " + to_string(imageInfo.out_color_space));
	}

	jpeg_start_decompress(&imageInfo);

	uint imageWidth = imageInfo.output_width;
	uint imageHeight = imageInfo.output_height;
	uint imageNumChannels = imageInfo.output_components;
	ImageData* image = new ImageData(imageWidth, imageHeight, imageNumChannels);
	uint imageStride = image->GetStride();
	uchar* rowMajorPixels = image->GetRowMajorPixels();
	while (imageInfo.output_scanline < imageHeight)
	{
		JSAMPROW writeLoc = &rowMajorPixels[imageInfo.output_scanline * imageStride];
		jpeg_read_scanlines(&imageInfo, &writeLoc, 1);
	}
	
	// Cleaning up.
	jpeg_finish_decompress(&imageInfo);
	jpeg_destroy_decompress(&imageInfo);
	fclose(imageFile);

	// Ensuring right color format.
	if (imageInfo.out_color_space == JCS_RGB)
	{
		ShipAssert(imageInfo.output_components == 3, "RGB should have 3 components, encountered: " + to_string(imageInfo.output_components));
		return image;
	}
	else if (imageInfo.out_color_space == JCS_CMYK)
	{
		ShipAssert(imageInfo.output_components == 4, "CMYK should have 4 components, encountered: " + to_string(imageInfo.output_components));

		// Transforming into RGB
		const uint rgbNumChannels = 3;
		ImageData* rgbImage = new ImageData(imageWidth, imageHeight, rgbNumChannels);
		uchar* rgbRowMajorPixels = rgbImage->GetRowMajorPixels();
		uint rgbImageStride = rgbImage->GetStride();
		for (uint row = 0; row < image->GetHeight(); ++row)
		{
			for (uint col = 0; col < image->GetWidth(); ++col)
			{
				uint pixelPos = row * imageStride + col * image->GetNumOfChannels();
				uchar cyan = rowMajorPixels[pixelPos];
				uchar magenta = rowMajorPixels[pixelPos + 1];
				uchar yellow = rowMajorPixels[pixelPos + 2];
				uchar key = rowMajorPixels[pixelPos + 3];

				// This is incorrect but pretty close aproximation. Resulting picture will be different,
				// but picture semantic will be intact, and picture semantic is only thing that matters for recognition.
				uint rgbPixelPos = row * rgbImageStride + col * rgbNumChannels;
				rgbRowMajorPixels[rgbPixelPos] = cyan * key / 255;
				rgbRowMajorPixels[rgbPixelPos + 1] = magenta * key / 255;
				rgbRowMajorPixels[rgbPixelPos + 2] = yellow * key / 255;
			}
		}

		delete image;
		return rgbImage;
	}
	else
	{
		ShipAssert(false, "Unexpected output color space: " + to_string(imageInfo.out_color_space));
		return NULL;
	}
}

void JpegDataParser::SaveImage(const ImageData& image, string outputPath)
{
	// Creating output image file.
	FILE *imageFile;
	ShipAssert((fopen_s(&imageFile, outputPath.c_str(), "wb")) == 0, "Cannot create jpeg image file \"" + outputPath + "\".");

	// Packing image.
	const int imageQuality = 95;
	uint imageWidth = image.GetWidth();
	uint imageHeight = image.GetHeight();
	uint imageNumChannels = image.GetNumOfChannels();
	uint imageStride = image.GetStride();
	struct jpeg_compress_struct imageInfo;
	struct jpeg_error_mgr errorHandler;
	imageInfo.err = jpeg_std_error(&errorHandler);
	jpeg_create_compress(&imageInfo);
	jpeg_stdio_dest(&imageInfo, imageFile);
	imageInfo.image_width = imageWidth;
	imageInfo.image_height = imageHeight;
	imageInfo.input_components = imageNumChannels;
	imageInfo.in_color_space = JCS_RGB;
	jpeg_set_defaults(&imageInfo);
	jpeg_set_quality(&imageInfo, imageQuality, TRUE);
	jpeg_start_compress(&imageInfo, TRUE);

	// Copying image data.
	uchar* rowMajorPixels = image.GetRowMajorPixels();
	while (imageInfo.next_scanline < imageInfo.image_height)
	{
		JSAMPROW readLoc = &rowMajorPixels[imageInfo.next_scanline * imageStride];
		jpeg_write_scanlines(&imageInfo, &readLoc, 1);
	}	

	// Cleaning up.
	jpeg_finish_compress(&imageInfo);
	jpeg_destroy_compress(&imageInfo);
	fclose(imageFile);
}

void JpegDataParser::TransferImageToOpDeviceBuffer(const ImageData& image, cudaStream_t stream)
{
	size_t imageBufferSize = image.GetBufferSize();
	AllocMemoryIfNeeded(imageBufferSize);

	CudaAssert(cudaMemcpyAsync(m_deviceOpSrcBuffer, image.GetRowMajorPixels(), imageBufferSize, cudaMemcpyHostToDevice, stream));
}

void JpegDataParser::CalculateResizeDimensions(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode,
	NppiSize& srcImageSize, NppiRect& srcImageRect, NppiRect& destImageRect, double& scaleX, double& scaleY)
{
	srcImageSize.width = image.GetWidth();
	srcImageSize.height = image.GetHeight();
	srcImageRect.x = 0;
	srcImageRect.y = 0;
	srcImageRect.width = srcImageSize.width;
	srcImageRect.height = srcImageSize.height;
	
	switch (resizeMode)
	{
		case ResizeMode::ResizeToSmaller:
			scaleX = scaleY = srcImageSize.width <= srcImageSize.height ?
				((double)desiredWidth / srcImageSize.width + ((double)desiredWidth + 1.0) / srcImageSize.width) / 2.0 :
				((double)desiredHeight / srcImageSize.height + ((double)desiredHeight + 1.0) / srcImageSize.height) / 2.0;
			break;
		case ResizeMode::ResizeToLarger:
			scaleX = scaleY = srcImageSize.width >= srcImageSize.height ?
				((double)desiredWidth / srcImageSize.width + ((double)desiredWidth + 1.0) / srcImageSize.width) / 2.0 :
				((double)desiredHeight / srcImageSize.height + ((double)desiredHeight + 1.0) / srcImageSize.height) / 2.0;
			break;
		case ResizeMode::ResizeToFit:
			scaleX = ((double)desiredWidth / srcImageSize.width + ((double)desiredWidth + 1.0) / srcImageSize.width) / 2.0;
			scaleY = ((double)desiredHeight / srcImageSize.height + ((double)desiredHeight + 1.0) / srcImageSize.height) / 2.0;
			break;
		default:
			ShipAssert(false, "Unknown resize mode encountered!");
	}
	nppiGetResizeRect(srcImageRect, &destImageRect, scaleX, scaleY, 0, 0, NPPI_INTER_LANCZOS);
	switch (resizeMode)
	{
		case ResizeMode::ResizeToSmaller:
			ShipAssert((destImageRect.width <= destImageRect.height && destImageRect.width == desiredWidth) ||
				(destImageRect.height < destImageRect.width && destImageRect.height == desiredHeight), "Calculated wrong crop dimensions!");
			break;
		case ResizeMode::ResizeToLarger:
			ShipAssert((destImageRect.width >= destImageRect.height && destImageRect.width == desiredWidth) ||
				(destImageRect.height > destImageRect.width && destImageRect.height == desiredHeight), "Calculated wrong crop dimensions!");
			break;
		case ResizeMode::ResizeToFit:
			ShipAssert(destImageRect.width == desiredWidth && destImageRect.height == desiredHeight, "Calculated wrong crop dimensions!");
	}
}

void JpegDataParser::ResizeOnDeviceBuffers(NppiSize srcImageSize, NppiRect srcImageRect, NppiRect destImageRect, uint numOfChannels,
	double scaleX, double scaleY)
{
	uint destImageBufferSize = numOfChannels * destImageRect.height * destImageRect.width * sizeof(uchar);
	AllocMemoryIfNeeded(destImageBufferSize);
	
	CudaNppAssert(nppiResizeSqrPixel_8u_C3R(m_deviceOpSrcBuffer, srcImageSize, srcImageSize.width * numOfChannels, srcImageRect,
		m_deviceOpDestBuffer, destImageRect.width * numOfChannels, destImageRect, scaleX, scaleY, 0, 0, NPPI_INTER_LANCZOS));
}

ImageData* JpegDataParser::ResizeImageCu(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode, cudaStream_t stream)
{
	// Setting up the device source buffer and transfering image data to it.
	TransferImageToOpDeviceBuffer(image, stream);

	// Calculating dimensions.
	NppiSize srcImageSize;
	NppiRect srcImageRect, destImageRect;
	double scaleX, scaleY;
	CalculateResizeDimensions(image, desiredWidth, desiredHeight, resizeMode, srcImageSize, srcImageRect, destImageRect, scaleX, scaleY);

	// Resizing on device.
	ResizeOnDeviceBuffers(srcImageSize, srcImageRect, destImageRect, image.GetNumOfChannels(), scaleX, scaleY);
	
	// Creating output image.
	ImageData* resizedImage = new ImageData(destImageRect.width, destImageRect.height, image.GetNumOfChannels());
	CudaAssert(cudaMemcpyAsync(resizedImage->GetRowMajorPixels(), m_deviceOpDestBuffer, resizedImage->GetBufferSize(), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	return resizedImage;
}

ImageData* JpegDataParser::CropImage(const ImageData& image, uint startPixelX, uint startPixelY, uint desiredWidth, uint desiredHeight, bool flipCrop)
{
	uint endPixelX = startPixelX + desiredWidth;
	uint endPixelY = startPixelY + desiredHeight;
	uint numOfChannels = image.GetNumOfChannels();
	uint imageStride = image.GetStride();
	uint croppedImageStride = numOfChannels * desiredWidth;

	ShipAssert(endPixelX <= image.GetWidth(), "Can't crop jpeg image, bad crop width dimensions.");
	ShipAssert(endPixelY <= image.GetHeight(), "Can't crop jpeg image, bad crop height dimensions.");
	
	ImageData* croppedImage = new ImageData(desiredWidth, desiredHeight, numOfChannels);
	uchar* imageRowMajorPixels = image.GetRowMajorPixels();
	uchar* croppedImageRowMajorPixels = croppedImage->GetRowMajorPixels();

	// Redundant code to avoid checking for flip deep inside the loop which hurts the performance.
	if (flipCrop)
	{
		for (uint row = startPixelY; row < endPixelY; ++row)
		{
			const uint croppedImageRowOffset = (row - startPixelY) * croppedImageStride;
			const uint imageRowOffset = row * imageStride;
			for (uint col = startPixelX; col < endPixelX; ++col)
			{
				const uint croppedImageTotalOffset = croppedImageRowOffset + (col - startPixelX) * numOfChannels;
				const uint imageTotalOffset = imageRowOffset + (startPixelX + endPixelX - col - 1) * numOfChannels;
				for (uint channel = 0; channel < numOfChannels; ++channel)
				{
					croppedImageRowMajorPixels[croppedImageTotalOffset + channel] = imageRowMajorPixels[imageTotalOffset + channel];
				}
			}
		}
	}
	else
	{
		for (uint row = startPixelY; row < endPixelY; ++row)
		{
			const uint croppedImageRowOffset = (row - startPixelY) * croppedImageStride;
			const uint imageRowOffset = row * imageStride;
			for (uint col = startPixelX; col < endPixelX; ++col)
			{
				const uint croppedImageTotalOffset = croppedImageRowOffset + (col - startPixelX) * numOfChannels;
				const uint imageTotalOffset = imageRowOffset + col * numOfChannels;
				for (uint channel = 0; channel < numOfChannels; ++channel)
				{
					croppedImageRowMajorPixels[croppedImageTotalOffset + channel] = imageRowMajorPixels[imageTotalOffset + channel];
				}
			}
		}
	}

	return croppedImage;
}

// Crop kernel for images of small size (less or equal than maximum number of threads per block).
__global__ void CropKernelSmall(uchar* srcBuffer, uchar* destBuffer, uint startPixelX, uint startPixelY, uint imageStride, uint croppedSize, uint numOfChannels)
{
	uint startPosX = startPixelX * numOfChannels;
	uint endPixelY = startPixelY + croppedSize;
	uint croppedImageStride = croppedSize * numOfChannels;

	for (uint row = startPixelY; row < endPixelY; row += gridDim.x)
	{
		uint posY = row + blockIdx.x;
		if (posY < endPixelY && threadIdx.x < croppedImageStride)
		{			
			destBuffer[(posY - startPixelY) * croppedImageStride + threadIdx.x] = srcBuffer[posY * imageStride + startPosX + threadIdx.x];
		}
	}
}

// Crop kernel for images of medium size (larger than maximum number of threads per block, but smaller or equal than maximum number of threads per SM).
__global__ void CropKernelMedium(uchar* srcBuffer, uchar* destBuffer, uint startPixelX, uint startPixelY, uint imageStride, uint croppedSize, uint numOfChannels)
{
	uint startPosX = startPixelX * numOfChannels;
	uint endPixelY = startPixelY + croppedSize;
	uint croppedImageStride = croppedSize * numOfChannels;

	for (uint row = startPixelY; row < endPixelY; ++row)
	{
		uint posX = blockIdx.x * blockDim.x + threadIdx.x;
		if (posX < croppedImageStride)
		{
			destBuffer[(row - startPixelY) * croppedImageStride + posX] = srcBuffer[row * imageStride + startPosX + posX];
		}
	}
}

// Crop kernel for images of large size (larger than maximum number of threads per SM).
__global__ void CropKernelLarge(uchar* srcBuffer, uchar* destBuffer, uint startPixelX, uint startPixelY, uint imageStride, uint croppedSize, uint numOfChannels)
{
	uint startPosX = startPixelX * numOfChannels;
	uint endPixelY = startPixelY + croppedSize;
	uint croppedImageStride = croppedSize * numOfChannels;

	for (uint row = startPixelY; row < endPixelY; ++row)
	{
		for (uint posX = blockIdx.x * blockDim.x + threadIdx.x; posX < croppedImageStride; posX += gridDim.x * blockDim.x)
		{
			destBuffer[(row - startPixelY) * croppedImageStride + posX] = srcBuffer[row * imageStride + startPosX + posX];
		}
	}
}

ImageData* JpegDataParser::ResizeImageWithCropCu(const ImageData& image, uint desiredWidth, uint desiredHeight, ResizeMode resizeMode,
	CropMode cropMode, cudaStream_t stream)
{
	// Setting up the device source buffer and transfering image data to it.
	TransferImageToOpDeviceBuffer(image, stream);

	// Calculating dimensions.
	NppiSize srcImageSize;
	NppiRect srcImageRect, destImageRect;
	double scaleX, scaleY;
	CalculateResizeDimensions(image, desiredWidth, desiredHeight, resizeMode, srcImageSize, srcImageRect, destImageRect, scaleX, scaleY);

	// Resizing on device.
	ResizeOnDeviceBuffers(srcImageSize, srcImageRect, destImageRect, image.GetNumOfChannels(), scaleX, scaleY);

	// Cropping on device.
	uint croppedSize = min(destImageRect.width, destImageRect.height);
	uchar* destinationBuffer = m_deviceOpDestBuffer;
	bool skipCropping = (destImageRect.width == destImageRect.height) || (cropMode == CropMode::CropLeft && destImageRect.width < destImageRect.height);
	if (!skipCropping)
	{
		uint startPixelX, startPixelY;
		switch (cropMode)
		{
			case CropMode::CropLeft:
				startPixelX = 0;
				startPixelY = 0;
				break;
			case CropMode::CropCentral:
				startPixelX = (destImageRect.width - croppedSize) / 2;
				startPixelY = (destImageRect.height - croppedSize) / 2;
				break;
			case CropMode::CropRight:
				startPixelX = destImageRect.width - croppedSize;
				startPixelY = destImageRect.height - croppedSize;
				break;
			default:
				ShipAssert(false, "Unknown crop mode encountered!");
		}
		// Reversing src and dest buffers, to save time of calling memcpy.
		destinationBuffer = m_deviceOpSrcBuffer;
		uint croppedImageStride = croppedSize * image.GetNumOfChannels();
		if (croppedImageStride <= Config::MAX_NUM_THREADS)
		{
			uint numThreads = RoundUp(croppedImageStride, Config::WARP_SIZE);
			uint numBlocks = min(Config::MAX_NUM_FULL_BLOCKS * Config::MAX_NUM_THREADS / croppedImageStride, (uint)Config::MAX_NUM_BLOCKS);
			LAUNCH_KERNEL_ASYNC(CropKernelSmall, numBlocks, numThreads, stream)(m_deviceOpDestBuffer, m_deviceOpSrcBuffer, startPixelX, startPixelY,
				destImageRect.width * image.GetNumOfChannels(), croppedSize, image.GetNumOfChannels());
		}
		else if (croppedImageStride <= Config::MAX_NUM_FULL_BLOCKS * Config::MAX_NUM_THREADS)
		{
			uint numThreads = RoundUp(DivideUp(croppedImageStride, Config::MAX_NUM_FULL_BLOCKS), Config::WARP_SIZE);
			uint numBlocks = Config::MAX_NUM_FULL_BLOCKS;
			LAUNCH_KERNEL_ASYNC(CropKernelMedium, numBlocks, numThreads, stream)(m_deviceOpDestBuffer, m_deviceOpSrcBuffer, startPixelX, startPixelY,
				destImageRect.width * image.GetNumOfChannels(), croppedSize, image.GetNumOfChannels());
		}
		else
		{
			uint numThreads = Config::MAX_NUM_THREADS;
			uint numBlocks = Config::MAX_NUM_FULL_BLOCKS;
			LAUNCH_KERNEL_ASYNC(CropKernelLarge, numBlocks, numThreads, stream)(m_deviceOpDestBuffer, m_deviceOpSrcBuffer, startPixelX, startPixelY,
				destImageRect.width * image.GetNumOfChannels(), croppedSize, image.GetNumOfChannels());
		}
		CudaAssert(cudaGetLastError());
	}

	// Creating output image.
	ImageData* resizedImage = new ImageData(croppedSize, croppedSize, image.GetNumOfChannels());
	CudaAssert(cudaMemcpyAsync(resizedImage->GetRowMajorPixels(), destinationBuffer, resizedImage->GetBufferSize(), cudaMemcpyDeviceToHost, stream));
	resizedImage->m_deviceImageBuffer = destinationBuffer;
	cudaStreamSynchronize(stream);

	return resizedImage;
}