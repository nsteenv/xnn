// ----------------------------------------------------------------------------------------------------
// Copyrighted by Marko Rakita.
// Author: Marko Rakita
// File contains: Console helper.
// Created: 03/06/2016.
// ----------------------------------------------------------------------------------------------------

#include "include/consolehelper.cuh"

ConsoleHelper::ConsoleHelper()
{
	m_consoleStdOutHandle = GetStdHandle(STD_OUTPUT_HANDLE);
	GetConsoleScreenBufferInfo(m_consoleStdOutHandle, &m_consoleScreenBufferInfo);
	m_lastSetForeground = m_lastSetBackground = m_consoleScreenBufferInfo.wAttributes;
}

ConsoleHelper::~ConsoleHelper()
{
	SetConsoleTextAttribute(m_consoleStdOutHandle, m_consoleScreenBufferInfo.wAttributes);
}

void ConsoleHelper::SetConsoleForeground(ConsoleForeground::Foreground foregroundColor)
{
	SetConsoleTextAttribute(m_consoleStdOutHandle, foregroundColor);
	m_lastSetForeground = foregroundColor;
}

void ConsoleHelper::RevertConsoleForeground()
{
	SetConsoleTextAttribute(m_consoleStdOutHandle, m_lastSetForeground);
}

void ConsoleHelper::SetConsoleBackground(ConsoleBackground::Background backgroundColor)
{
	SetConsoleTextAttribute(m_consoleStdOutHandle, backgroundColor);
	m_lastSetBackground = backgroundColor;
}

void ConsoleHelper::RevertConsoleBackground()
{
	SetConsoleTextAttribute(m_consoleStdOutHandle, m_lastSetBackground);
}