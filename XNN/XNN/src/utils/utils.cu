// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Utility functions.
// Created: 11/24/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/utils.cuh"

string ConvertToLowercase(string inputString)
{
	string outputString = inputString;
	if (inputString != "")
	{
		transform(outputString.begin(), outputString.end(), outputString.begin(), tolower);
	}

	return outputString;
}

string GetExtension(string fileName)
{
	size_t dotPosition = fileName.find_last_of('.');
	if (dotPosition != string::npos && dotPosition < fileName.size() - 1)
	{
		return ConvertToLowercase(fileName.substr(dotPosition + 1));
	}

	return "";
}

string GetFileName(string filePath)
{
	size_t slashPosition = filePath.find_last_of('\\');
	if (slashPosition == string::npos)
	{
		slashPosition = filePath.find_last_of('/');
	}
	if (slashPosition != string::npos && slashPosition < filePath.size() - 1)
	{
		return filePath.substr(slashPosition + 1);
	}

	return filePath;
}

string GetFileNameWithoutExtension(string filePath)
{
	string fileName = GetFileName(filePath);
	size_t dotPosition = fileName.find_last_of('.');
	if (dotPosition != string::npos)
	{
		return fileName.substr(0, dotPosition);
	}

	return fileName;
}

bool ParseArgument(int argc, char *argv[], string signature, string& out, bool convertToLowercase /*= false*/)
{
	for (int i = 0; i < argc - 1; ++i)
	{
		if (ConvertToLowercase(string(argv[i])) == signature)
		{			
			if (convertToLowercase)
			{
				out = ConvertToLowercase(string(argv[i + 1]));
			}
			else
			{
				out = string(argv[i + 1]);
			}
			return true;
		}
	}

	return false;
}

bool ParseArgument(int argc, char *argv[], string signature, uint& out)
{
	for (int i = 0; i < argc - 1; ++i)
	{
		if (ConvertToLowercase(string(argv[i])) == signature)
		{
			out = stoi(string(argv[i + 1]));
			return true;
		}
	}

	return false;
}

void ParseArgument(int argc, char *argv[], string signature, bool& out)
{
	for (int i = 0; i < argc; ++i)
	{
		if (ConvertToLowercase(string(argv[i])) == signature)
		{
			out = true;
			return;
		}
	}

	out = false;
}

size_t GetSizeOfAvailableGpuMemory()
{
	size_t free, total;
	CudaAssert(cudaMemGetInfo(&free, &total));

	return free;
}

string GetCurrentTimeStamp()
{
	SYSTEMTIME time;
	GetLocalTime(&time);

	string month = to_string(time.wMonth);
	if (month.length() == 1)
	{
		month = "0" + month;
	}

	string day = to_string(time.wDay);
	if (day.length() == 1)
	{
		day = "0" + day;
	}

	string hour = to_string(time.wHour);
	if (hour.length() == 1)
	{
		hour = "0" + hour;
	}

	string minute = to_string(time.wMinute);
	if (minute.length() == 1)
	{
		minute = "0" + minute;
	}

	return month + "/" + day + "/" + to_string(time.wYear) + " " + hour + ":" + minute;
}

void _Assert(bool condition, string message, const char* file, const char* function, int line, bool shouldExit)
{
	if (!condition)
	{
		{
			lock_guard<mutex> lock(s_consoleMutex);
			s_consoleHelper.SetConsoleForeground(ConsoleForeground::RED);
			cout << endl << "Fatal error encountered: " << message << endl << endl;
			cout << "Operation info: " << function << " (\"" << GetFileName(file) << "\" [line: " << line << "])" << endl << endl;
		}
		if (shouldExit)
		{
			exit(EXIT_FAILURE);
		}
	}
}

void _CudaAssert(cudaError_t cudaStatus, const char* file, const char* function, int line)
{
	if (cudaStatus != cudaSuccess)
	{
		{
			lock_guard<mutex> lock(s_consoleMutex);
			s_consoleHelper.SetConsoleForeground(ConsoleForeground::RED);
			cout << endl << "CUDA operation failed with status: " << cudaGetErrorName(cudaStatus) << " (" << cudaGetErrorString(cudaStatus) << ")" << endl << endl;
			cout << "Operation info: " << function << " (\"" << GetFileName(file) << "\" [line: " << line << "])" << endl << endl;
		}
		exit(EXIT_FAILURE);
	}
}

void _CudaNppAssert(NppStatus nppStatus, const char* file, const char* function, int line)
{
	if (nppStatus != NPP_SUCCESS)
	{
		{
			lock_guard<mutex> lock(s_consoleMutex);
			s_consoleHelper.SetConsoleForeground(ConsoleForeground::RED);
			cout << endl << "CUDA NPP operation failed with status: " << nppStatus << endl << endl;
			cout << "Operation info: " << function << " (\"" << GetFileName(file) << "\" [line: " << line << "])" << endl << endl;
		}
		exit(EXIT_FAILURE);
	}
}

void _CudaCublasAssert(cublasStatus_t cublasStatus, const char* file, const char* function, int line)
{
	if (cublasStatus != CUBLAS_STATUS_SUCCESS)
	{
		{
			lock_guard<mutex> lock(s_consoleMutex);
			s_consoleHelper.SetConsoleForeground(ConsoleForeground::RED);
			cout << endl << "CUDA CUBLAS operation failed with status: " << cublasStatus << endl << endl;
			cout << "Operation info: " << function << " (\"" << GetFileName(file) << "\" [line: " << line << "])" << endl << endl;
		}
		exit(EXIT_FAILURE);
	}
}

void EmitWarning(string message)
{
	lock_guard<mutex> lock(s_consoleMutex);
	s_consoleHelper.SetConsoleForeground(ConsoleForeground::YELLOW);
	cout << message << endl;
	s_consoleHelper.RevertConsoleForeground();
}