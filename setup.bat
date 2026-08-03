@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher was not found. Install Python 3.11 or newer.
  pause
  exit /b 1
)
if not exist .venv (
  py -3.11 -m venv .venv
  if errorlevel 1 py -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -e .
if errorlevel 1 (
  echo Installation failed.
  pause
  exit /b 1
)
python -m btc_ema_trader init
endlocal
