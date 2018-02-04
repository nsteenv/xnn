// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Image data class.
// Created: 11/29/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/imagedata.cuh"

ImageData::ImageData(uint width, uint height, uint numberOfChannels)
{
	m_width = width;
	m_height = height;
	m_numberOfChannels = numberOfChannels;
	m_stride = m_width * m_numberOfChannels;
	m_bufferSize = m_width * m_height * m_numberOfChannels * sizeof(uchar);

	CudaAssert(cudaMallocHost<uchar>(&m_rowMajorPixels, m_bufferSize, cudaHostAllocPortable));
}

ImageData::~ImageData()
{
	CudaAssert(cudaFreeHost(m_rowMajorPixels));
}