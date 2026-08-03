@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe call setup.bat
call .venv\Scripts\activate.bat
python -m btc_ema_trader status
pause
endlocal
