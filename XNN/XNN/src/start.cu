// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Program for training neural networks.
// Created: 11/24/2015.
// ----------------------------------------------------------------------------------------------------

#include "tests/include/testsdriver.cuh"
#include "tools/include/datamaker.cuh"
#include "tools/include/featurizer.cuh"
#include "tools/include/trainer.cuh"

/*
	Prints usage of this program.
*/
void PrintUsage()
{
	cout << endl;
	cout << "---------------------------------------------------------------------------" << endl;
	cout << "Usage:" << endl;
	cout << "---------------------------------------------------------------------------" << endl;
	cout << "    -makedatat    (prepares data for training, input data should be in folders train/validation/test and each folder should have labels.txt file)" << endl;
	cout << "     or" << endl;
	cout << "    -makedataf    (prepares data for featurization, input data should be listed in datalist.txt file)" << endl;
	cout << "     or" << endl;
	cout << "    -makedatae    (extends dataset by applying crops and flips, input data should be listed in datalist.txt file)" << endl;
	cout << "        " << DataMaker::c_inputFolderSignature << " \"...\"    (folder with original data)" << endl;
	cout << "        " << DataMaker::c_outputFolderSignature << " \"...\"    (folder in which to output prepared data)" << endl;
	cout << "        " << DataMaker::c_imageSizeSignature << " N    (output images will be size of NxN, default value: " << DataMaker::c_defaultImageSize << ")" << endl;
	cout << "        " << DataMaker::c_numImageChannelsSignature << " C    (input and output images will have C channels, default value: " << DataMaker::c_defaultNumOfImageChannels << ")" << endl;
	cout << endl;
	cout << "    -train    (runs training of network)" << endl;
	cout << "        " << Trainer::c_configurationSignature << " \"...\"    (network configuration file path)" << endl;
	cout << "        " << Trainer::c_dataFolderSignature << " \"...\"    (folder with data for training)" << endl;
	cout << "        " << Trainer::c_workFolderSignature << " \"...\"    (folder where trained models and checkpoints will be saved)" << endl;
	cout << "        " << Trainer::c_numEpochsSignature << " X    (number of epochs to train will be X)" << endl;
	cout << "        " << Trainer::c_batchSizeSignature << " N    (batch size will be N)" << endl;
	cout << endl;
	cout << "    -featurizer    (runs features extraction)" << endl;
	// TODO: finish
	cout << endl;
	cout << "    -runtests    (runs tests for solution, if no other argument specified then it runs all tests)" << endl;
	cout << "        " << TestsDriver::c_outputFolderSignature << " \"...\"    (folder for test output, if not specified some tests might not be run!)" << endl;
	cout << "        " << TestsDriver::c_componentToRunSignature << " ComponentName    (runs all tests in component with name equals to ComponentName)" << endl;
	cout << "        " << TestsDriver::c_testToRunSignature << " TestName    (runs only tests with name equal to TestName, and only if they are in component if component is specified)" << endl;
	cout << "---------------------------------------------------------------------------" << endl << endl << endl;
}

/*
	Main function.
*/
int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		PrintUsage();
	}
	else if (strcmp(argv[1], "-makedatat") == 0)
	{
		DataMaker dataMaker;
		if (dataMaker.ParseArguments(argc, argv))
		{
			dataMaker.MakeDataForTraining();
		}
		else
		{
			cout << endl << "Bad arguments format." << endl << endl;
			PrintUsage();
		}
	}
	else if (strcmp(argv[1], "-makedataf") == 0)
	{
		DataMaker dataMaker;
		if (dataMaker.ParseArguments(argc, argv))
		{
			dataMaker.MakeDataForFeaturization();
		}
		else
		{
			cout << endl << "Bad arguments format." << endl << endl;
			PrintUsage();
		}
	}
	else if (strcmp(argv[1], "-makedatae") == 0)
	{
		DataMaker dataMaker;
		if (dataMaker.ParseArguments(argc, argv))
		{
			dataMaker.MakeExtendedDataset();
		}
		else
		{
			cout << endl << "Bad arguments format." << endl << endl;
			PrintUsage();
		}
	}
	else if (strcmp(argv[1], "-train") == 0)
	{
		Trainer trainer;
		if (trainer.ParseArguments(argc, argv))
		{
			trainer.RunTraining();
		}
		else
		{
			cout << endl << "Bad arguments format." << endl << endl;
			PrintUsage();
		}
	}
	else if (strcmp(argv[1], "-featurize") == 0)
	{
		Featurizer featurizer;
		if (featurizer.ParseArguments(argc, argv))
		{
			featurizer.RunExtraction();
		}
		else
		{
			cout << endl << "Bad arguments format." << endl << endl;
			PrintUsage();
		}
	}
	else if (strcmp(argv[1], "-runtests") == 0)
	{
		TestsDriver testsDriver;
		if (testsDriver.ParseArguments(argc, argv))
		{
			testsDriver.RunTests();
		}
		else
		{
			cout << endl << "Bad arguments format." << endl << endl;
			PrintUsage();
		}
	}
	else
	{
		cout << endl << "Unknown command." << endl << endl;
		PrintUsage();
	}

	CudaAssert(cudaDeviceReset());

	return 0;
}