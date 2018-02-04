// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Utility functions.
// Created: 11/24/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <iostream>
#include <mutex>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <npp.h>

#include "consolehelper.cuh"

#define uchar unsigned char
#define uint unsigned int

// Kernel launcher macros.
#define LAUNCH_KERNEL(kernel, gridDimensions, blockDimensions) kernel<<<gridDimensions, blockDimensions>>>
#define LAUNCH_KERNEL_ASYNC(kernel, gridDimensions, blockDimensions, stream) kernel<<<gridDimensions, blockDimensions, 0, stream>>>

// Macro to check ship asserts.
// Asserts in both debug and ship if condition is not met.
#define ShipAssert(condition, message) _Assert(condition, message, __FILE__, __FUNCTION__, __LINE__, true)

// Macro to check debug asserts.
// Asserts in debug only if condition is not met.
#if !defined(NDEBUG)
	#define DebugAssert(condition, message) \
		do { _Assert(condition, message, __FILE__, __FUNCTION__, __LINE__, false); } while (0)
#else
	#define DebugAssert(condition, message) \
		do { (void)sizeof(condition); } while(0)
#endif

// Macro to check errors in CUDA operations.
// Asserts in both debug and ship if CUDA operation failed, can't allow this to happen.
#define CudaAssert(cudaStatus) _CudaAssert(cudaStatus, __FILE__, __FUNCTION__, __LINE__)

// Macro to check errors in CUDA NPP library operations.
// Asserts in both debug and ship if CUDA NPP library operation failed, can't allow this to happen.
#define CudaNppAssert(nppStatus) _CudaNppAssert(nppStatus, __FILE__, __FUNCTION__, __LINE__)

// Macro to check errors in CUDA CUBLAS library operations.
// Asserts in both debug and ship if CUDA CUBLAS library operation failed, can't allow this to happen.
#define CudaCublasAssert(cublasStatus) _CudaCublasAssert(cublasStatus, __FILE__, __FUNCTION__, __LINE__)

using namespace std;

// Console helper for manipulating console look and feel.
static ConsoleHelper s_consoleHelper;

// Mutex for console output.
static mutex s_consoleMutex;

// Returns lower case version of string.
string ConvertToLowercase(string inputString);

// Returns extension of file.
string GetExtension(string fileName);

// Returns file name from file path.
string GetFileName(string filePath);

// Returns file name from file path, without extension.
string GetFileNameWithoutExtension(string filePath);

// Parses argument from arguments list with specified signature to out variable.
bool ParseArgument(int argc, char *argv[], string signature, string& out, bool convertToLowercase = false);
bool ParseArgument(int argc, char *argv[], string signature, uint& out);

// Sets out as true if there is argument in arguments list with specified signature.
void ParseArgument(int argc, char *argv[], string signature, bool& out);

// Divides two numbers so that quotient times divisor is larger or equal than dividend.
inline uint DivideUp(uint dividend, uint divisor)
{
	return (dividend + divisor - 1) / divisor;
}

// Rounds up the number so it is divisible by base.
inline uint RoundUp(uint number, uint base)
{
	return DivideUp(number, base) * base;
}

// Gets size of available memory on current device, in bytes.
size_t GetSizeOfAvailableGpuMemory();

// Gets current time stamp.
string GetCurrentTimeStamp();

// Asserts in both debug and ship if condition is not met.
// Should never be called directly, use macro!
void _Assert(bool condition, string message, const char* file, const char* function, int line, bool shouldExit);

// Asserts in both debug and ship if CUDA operation failed, can't allow this to happen.
// Should never be called directly, use macro!
void _CudaAssert(cudaError_t cudaStatus, const char* file, const char* function, int line);

// Asserts in both debug and ship if CUDA NPP library operation failed, can't allow this to happen.
// Should never be called directly, use macro!
void _CudaNppAssert(NppStatus nppStatus, const char* file, const char* function, int line);

// Asserts in both debug and ship if CUDA CUBLAS library operation failed, can't allow this to happen.
// Should never be called directly, use macro!
void _CudaCublasAssert(cublasStatus_t cublasStatus, const char* file, const char* function, int line);

// Emits warning to console.
void EmitWarning(string message);