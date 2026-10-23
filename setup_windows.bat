@echo off
setlocal
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error
echo.
echo Setup complete. Run run_windows.bat to start Vita AI.
pause
exit /b 0
:error
echo.
echo Setup failed. Confirm Python 3.11 or newer is installed and try again.
pause
exit /b 1

