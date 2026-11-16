@echo off
cd /d "%~dp0\.."
if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe scripts\train.py --task both
) else (
  py -3.12 scripts\train.py --task both
)
pause
