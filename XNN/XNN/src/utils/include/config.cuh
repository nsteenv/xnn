// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Configuration constants.
// Created: 12/19/2015.
// ----------------------------------------------------------------------------------------------------

#pragma once

namespace Config
{
	// Number of CPU cores (put double in case of Intel CPUs which support Hyper-Threading).
	const int NUM_CPU_CORES = 12;

	// Number of CPU cores for data load.
	const int NUM_DATA_LOAD_CPU_CORES = 8;

	// Number of GPUs in a system.
	const int NUM_GPUS = 2;

	// Maximum number of blocks per SM.
	const int MAX_NUM_BLOCKS = 32;

	// Maximum number of full blocks per SM.
	const int MAX_NUM_FULL_BLOCKS = 2;

	// Maximum number of threads per block.
	const int MAX_NUM_THREADS = 1024;

	// Size of a warp.
	const int WARP_SIZE = 32;
}