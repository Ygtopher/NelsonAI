@echo off
title Nelson AI - Chat
echo ========================================================
echo Starting Nelson AI Chat Interface...
echo ========================================================
cd /d "%~dp0"

echo Activating Python Environment...
call venv\Scripts\activate.bat

echo Launching Nelson...
python chat.py

pause
