@echo off
setlocal
cd /d "%~dp0"
set "WINDIR=C:\Windows"
set "MPLCONFIGDIR=%~dp0.matplotlib"
where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON_CMD=py -3.12"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo Python 3.12 was not found on PATH.
    echo Install Python from https://www.python.org/downloads/ and enable "Add Python to PATH".
    pause
    exit /b 1
  )
  set "PYTHON_CMD=python"
)
if not exist ".venv\Scripts\python.exe" %PYTHON_CMD% -m venv .venv
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
if not exist "models\plant_disease_mobilenetv2.keras" (
  ".venv\Scripts\python.exe" download_model.py
)
echo.
echo Setup complete. Double-click run_vita_ai.bat to start the app.
pause
endlocal
