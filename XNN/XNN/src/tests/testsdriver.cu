// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Driver for all tests.
// Created: 12/12/2015.
// ----------------------------------------------------------------------------------------------------

#include "include/testsdriver.cuh"

const string TestsDriver::c_outputFolderSignature = "-outputfolder";
const string TestsDriver::c_componentToRunSignature = "-component";
const string TestsDriver::c_testToRunSignature = "-testname";

void TestsDriver::RegisterTesters()
{
	m_testers["jpegdataparser"] = new TestJpegDataParser(m_outputFolder);
	m_testers["convolutionallayer"] = new TestConvolutionalLayer(m_outputFolder);
	m_testers["maxpoollayer"] = new TestMaxPoolLayer(m_outputFolder);
	m_testers["responsenormalizationlayer"] = new TestResponseNormalizationLayer(m_outputFolder);
	m_testers["standardlayer"] = new TestStandardLayer(m_outputFolder);
	m_testers["dropoutlayer"] = new TestDropoutLayer(m_outputFolder);
	m_testers["softmaxlayer"] = new TestSoftMaxLayer(m_outputFolder);
	m_testers["outputlayer"] = new TestOutputLayer(m_outputFolder);
}

TestsDriver::~TestsDriver()
{
	for (auto tester = m_testers.begin(); tester != m_testers.end(); ++tester)
	{
		delete tester->second;
	}
}

bool TestsDriver::ParseArguments(int argc, char *argv[])
{
	m_outputFolder = "";
	m_componentToRun = "";
	m_testToRun = "";

	switch (argc)
	{
		case 2:
			return true;
		case 4:
		case 6:
		case 8:
		{
			bool argumentsParsed = ParseArgument(argc, argv, c_outputFolderSignature, m_outputFolder);
			argumentsParsed = ParseArgument(argc, argv, c_componentToRunSignature, m_componentToRun, true) || argumentsParsed;
			argumentsParsed = ParseArgument(argc, argv, c_testToRunSignature, m_testToRun, true) || argumentsParsed;
			return argumentsParsed;
		}
		default:
			return false;
	}
}

void TestsDriver::RunTests()
{
	RegisterTesters();
	
	cout << endl;

	bool noTestsRun = true;

	if (m_componentToRun == "" && m_testToRun == "")
	{
		for (auto tester = m_testers.begin(); tester != m_testers.end(); ++tester)
		{
			tester->second->RunAllTests();
			s_consoleHelper.SetConsoleForeground(ConsoleForeground::GREEN);
			cout << "Tests from component " << tester->first << " passed!" << endl << endl;
			s_consoleHelper.RevertConsoleForeground();
		}
		s_consoleHelper.SetConsoleForeground(ConsoleForeground::GREEN);
		cout << endl << "All tests passed!" << endl << endl;
		return;
	}
	else
	{
		for (auto tester = m_testers.begin(); tester != m_testers.end(); ++tester)
		{
			if (tester->first == m_componentToRun)
			{
				if (m_testToRun == "")
				{
					tester->second->RunAllTests();
					s_consoleHelper.SetConsoleForeground(ConsoleForeground::GREEN);
					cout << "Tests from component " << tester->first << " passed!" << endl << endl;
					s_consoleHelper.RevertConsoleForeground();
				}
				else
				{
					if (tester->second->HasTest(m_testToRun))
					{
						tester->second->RunTest(m_testToRun);
						s_consoleHelper.SetConsoleForeground(ConsoleForeground::GREEN);
						cout << "Test " << m_testToRun << " in component " << m_componentToRun << " passed!" << endl << endl;
						s_consoleHelper.RevertConsoleForeground();
					}
				}
				return;
			}
			else if (m_componentToRun == "" && tester->second->HasTest(m_testToRun))
			{
				tester->second->RunTest(m_testToRun);
				s_consoleHelper.SetConsoleForeground(ConsoleForeground::GREEN);
				cout << "Test " << m_testToRun << " in component " << tester->first << " passed!" << endl << endl;
				s_consoleHelper.RevertConsoleForeground();
				noTestsRun = false;
			}
		}
	}	

	if (noTestsRun)
	{
		s_consoleHelper.SetConsoleForeground(ConsoleForeground::RED);
		cout << endl << "No test has been run." << endl;
	}

	cout << endl;
}