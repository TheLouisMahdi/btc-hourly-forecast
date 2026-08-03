@echo off
setlocal
cd /d "%~dp0"
call setup.bat
if errorlevel 1 exit /b 1
call .venv\Scripts\activate.bat
python -m btc_ema_trader bootstrap --days 180 --provider auto
if errorlevel 1 (
  echo First-run bootstrap failed. Read docs\TROUBLESHOOTING.md
  pause
  exit /b 1
)
python -m btc_ema_trader dashboard
endlocal
