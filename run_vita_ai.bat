@echo off
setlocal
cd /d "%~dp0"
set "WINDIR=C:\Windows"
set "MPLCONFIGDIR=%~dp0.matplotlib"
set "TF_CPP_MIN_LOG_LEVEL=2"
if not exist ".venv\Scripts\python.exe" (
  echo Vita AI is not set up yet.
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
endlocal
