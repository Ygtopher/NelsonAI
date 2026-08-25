@echo off
title Nelson AI - Training
echo ========================================================
echo Starting Nelson AI Trainer...
echo ========================================================
cd /d "%~dp0"

echo Activating Python Environment...
call venv\Scripts\activate.bat

echo Resuming Training (Batch Size 3)...
python training/trainer.py --batch-size 3

pause
