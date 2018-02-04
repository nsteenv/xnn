// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Tests for jpeg data parser.
// Created: 12/12/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/testjpegdataparser.cuh"

TestJpegDataParser::TestJpegDataParser(string outputFolder)
{
	m_outputFolder = outputFolder;

	// Registering tests.
	m_jpegDataParserTests["resizeimagecu"] = &TestJpegDataParser::TestResizeImageCu;
	m_jpegDataParserTests["cropimage"] = &TestJpegDataParser::TestCropImage;
	m_jpegDataParserTests["resizeimagewithcropcu"] = &TestJpegDataParser::TestResizeImageWithCropCu;
}

bool TestJpegDataParser::HasTest(string testName)
{
	auto test = m_jpegDataParserTests.find(testName);
	return test != m_jpegDataParserTests.end();
}

void TestJpegDataParser::RunTest(string testName)
{
	auto test = m_jpegDataParserTests.find(testName);
	TestingAssert(test != m_jpegDataParserTests.end(), "Test not found!");

	((*this).*(test->second))();
}

void TestJpegDataParser::RunAllTests()
{
	for (auto test = m_jpegDataParserTests.begin(); test != m_jpegDataParserTests.end(); ++test)
	{
		((*this).*(test->second))();
		s_consoleHelper.SetConsoleForeground(ConsoleForeground::GREEN);
		cout << "Test " << test->first << " passed!" << endl << endl;
		s_consoleHelper.RevertConsoleForeground();
	}
}

//******************************************************************************************************
// Tests
//******************************************************************************************************

void TestJpegDataParser::TestResizeImageCu()
{
	if (m_outputFolder == "")
	{
		cout << "No output folder defined, TestResizeImageCu test was not run!" << endl;
		return;
	}

	vector<string> testImages;
	testImages.push_back("testImageH.JPEG");
	testImages.push_back("testImageV.JPEG");

	JpegDataParser dataParser;
	const uint imageSize = 224;
	ImageData* image;
	ImageData* resizedImage;
	
	for (string testImage : testImages)
	{
		image = dataParser.LoadImage(c_testResourcesFolder + testImage);
		string imageName = GetFileNameWithoutExtension("/" + testImage);

		// Test resizing to smaller.
		resizedImage = dataParser.ResizeImageCu(*image, imageSize, imageSize, ResizeMode::ResizeToSmaller);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToSmaller.jpg");
		delete resizedImage;
		// Test resizing to larger.
		resizedImage = dataParser.ResizeImageCu(*image, imageSize, imageSize, ResizeMode::ResizeToLarger);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToLarger.jpg");
		delete resizedImage;
		// Test resizing to fit.
		resizedImage = dataParser.ResizeImageCu(*image, imageSize, imageSize, ResizeMode::ResizeToFit);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToFit.jpg");
		delete resizedImage;

		delete image;
	}
}

void TestJpegDataParser::TestCropImage()
{
	if (m_outputFolder == "")
	{
		cout << "No output folder defined, TestCropImage test was not run!" << endl;
		return;
	}

	vector<string> testImages;
	testImages.push_back("testImageH.JPEG");
	testImages.push_back("testImageV.JPEG");

	JpegDataParser dataParser;
	const uint imageSize = 224;
	ImageData* image;
	ImageData* croppedImage;

	for (string testImage : testImages)
	{
		image = dataParser.LoadImage(c_testResourcesFolder + testImage);
		string imageName = GetFileNameWithoutExtension("/" + testImage);

		// Test crop of upper left corner, no flip.
		croppedImage = dataParser.CropImage(*image, 0, 0, imageSize, imageSize, false);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedUpperLeft.jpg");
		delete croppedImage;
		// Test crop of upper right corner, no flip.
		croppedImage = dataParser.CropImage(*image, image->GetWidth() - imageSize, 0, imageSize, imageSize, false);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedUpperRight.jpg");
		delete croppedImage;
		// Test crop center, no flip.
		croppedImage = dataParser.CropImage(*image, (image->GetWidth() - imageSize) /2, (image->GetHeight() - imageSize) / 2, imageSize, imageSize, false);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedCenter.jpg");
		delete croppedImage;
		// Test crop of lower left corner, no flip.
		croppedImage = dataParser.CropImage(*image, 0, image->GetHeight() - imageSize, imageSize, imageSize, false);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedLowerLeft.jpg");
		delete croppedImage;
		// Test crop of lower right corner, no flip.
		croppedImage = dataParser.CropImage(*image, image->GetWidth() - imageSize, image->GetHeight() - imageSize, imageSize, imageSize, false);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedLowerRight.jpg");
		delete croppedImage;

		// Test crop of upper left corner, with flip.
		croppedImage = dataParser.CropImage(*image, 0, 0, imageSize, imageSize, true);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedUpperLeftFlipped.jpg");
		delete croppedImage;
		// Test crop of upper right corner, with flip.
		croppedImage = dataParser.CropImage(*image, image->GetWidth() - imageSize, 0, imageSize, imageSize, true);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedUpperRightFlipped.jpg");
		delete croppedImage;
		// Test crop center, with flip.
		croppedImage = dataParser.CropImage(*image, (image->GetWidth() - imageSize) / 2, (image->GetHeight() - imageSize) / 2, imageSize, imageSize, true);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedCenterFlipped.jpg");
		delete croppedImage;
		// Test crop of lower left corner, with flip.
		croppedImage = dataParser.CropImage(*image, 0, image->GetHeight() - imageSize, imageSize, imageSize, true);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedLowerLeftFlipped.jpg");
		delete croppedImage;
		// Test crop of lower right corner, with flip.
		croppedImage = dataParser.CropImage(*image, image->GetWidth() - imageSize, image->GetHeight() - imageSize, imageSize, imageSize, true);
		dataParser.SaveImage(*croppedImage, m_outputFolder + "\\" + imageName + "-CroppedLowerRightFlipped.jpg");
		delete croppedImage;


		delete image;
	}
}

void TestJpegDataParser::TestResizeImageWithCropCu()
{
	if (m_outputFolder == "")
	{
		cout << "No output folder defined, TestResizeImageWithCropCu test was not run!" << endl;
		return;
	}

	vector<string> testImages;
	testImages.push_back("testImageH.JPEG");
	testImages.push_back("testImageV.JPEG");
	
	JpegDataParser dataParser;
	const uint imageSize = 224;
	const uint mediumImageSize = 2560;
	const uint largeImageSize = 3200;
	ImageData* image;
	ImageData* resizedImage;

	for (string testImage : testImages)
	{
		image = dataParser.LoadImage(c_testResourcesFolder + testImage);
		string imageName = GetFileNameWithoutExtension("/" + testImage);

		// Test resizing to smaller, cropping to left.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, imageSize, ResizeMode::ResizeToSmaller, CropMode::CropLeft);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToSmallerCroppedToLeft.jpg");
		delete resizedImage;
		// Test resizing to smaller, cropping to central.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, imageSize, ResizeMode::ResizeToSmaller, CropMode::CropCentral);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToSmallerCroppedToCentral.jpg");
		delete resizedImage;
		// Test resizing to smaller, cropping to right.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, imageSize, ResizeMode::ResizeToSmaller, CropMode::CropRight);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToSmallerCroppedToRight.jpg");
		delete resizedImage;

		// Test resizing to larger, cropping to left.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, imageSize, ResizeMode::ResizeToLarger, CropMode::CropLeft);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToLargerCroppedToLeft.jpg");
		delete resizedImage;
		// Test resizing to larger, cropping to central.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, imageSize, ResizeMode::ResizeToLarger, CropMode::CropCentral);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToLargerCroppedToCentral.jpg");
		delete resizedImage;
		// Test resizing to larger, cropping to right.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, imageSize, ResizeMode::ResizeToLarger, CropMode::CropRight);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToLargerCroppedToRight.jpg");
		delete resizedImage;

		// Test resizing to fit, cropping to left.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, 2 * imageSize, ResizeMode::ResizeToFit, CropMode::CropLeft);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToFitCroppedToLeft.jpg");
		delete resizedImage;
		// Test resizing to fit, cropping to central.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, 2 * imageSize, ResizeMode::ResizeToFit, CropMode::CropCentral);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToFitCroppedToCentral.jpg");
		delete resizedImage;
		// Test resizing to fit, cropping to right.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, imageSize, 2 * imageSize, ResizeMode::ResizeToFit, CropMode::CropRight);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-ResizedToFitCroppedToRight.jpg");
		delete resizedImage;

		// Test cropping with medium image size.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, mediumImageSize, mediumImageSize, ResizeMode::ResizeToSmaller, CropMode::CropCentral);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-CropMediumSize.jpg");
		delete resizedImage;

		// Test cropping with large image size.
		resizedImage = dataParser.ResizeImageWithCropCu(*image, mediumImageSize, largeImageSize, ResizeMode::ResizeToSmaller, CropMode::CropCentral);
		dataParser.SaveImage(*resizedImage, m_outputFolder + "\\" + imageName + "-CropLargeSize.jpg");
		delete resizedImage;

		delete image;
	}
}