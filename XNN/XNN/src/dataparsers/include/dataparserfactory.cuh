// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Factory for creating data parsers.
// Created: 11/29/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <map>

#include "../image/include/jpegdataparser.cuh"

using namespace std;

class DataParserFactory
{
private:
	// Map of created data parsers, mapped on data extension.
	map<string, DataParser*> m_dataParsers;

	// Creates appropriate data parser depending on data extension.
	// (Data extension should be in lower case!)
	// Returns: true if creating data parser is successful, false otherwise.
	bool CreateDataParser(string dataExtension, DataParser** outDataParser);

public:
	// Destructor.
	~DataParserFactory();

	// Returns appropriate data parser, depending on data extension.
	DataParser* GetDataParser(string dataExtension);
};
