// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Factory for creating data parsers.
// Created: 11/29/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/dataparserfactory.cuh"

bool DataParserFactory::CreateDataParser(string dataExtension, DataParser** outDataParser)
{
	if (dataExtension == "jpg" || dataExtension == "jpeg")
	{
		*outDataParser = new JpegDataParser();
		return true;
	}

	return false;
}

DataParserFactory::~DataParserFactory()
{
	for (auto it = m_dataParsers.begin(); it != m_dataParsers.end(); ++it)
	{
		delete it->second;
	}
}

DataParser* DataParserFactory::GetDataParser(string dataExtension)
{
	if (m_dataParsers.find(dataExtension) == m_dataParsers.end())
	{
		DataParser* newDataParser;
		ShipAssert(CreateDataParser(dataExtension, &newDataParser), "Can't create parser with that data extension.");
		m_dataParsers.insert(make_pair(dataExtension, newDataParser));
	}
	return m_dataParsers[dataExtension];
}