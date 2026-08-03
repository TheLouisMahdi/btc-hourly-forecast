@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe call setup.bat
if errorlevel 1 exit /b 1
call .venv\Scripts\activate.bat
python -m btc_ema_trader fetch --days 180 --provider auto
if errorlevel 1 goto :fail
python -m btc_ema_trader news --historical --days 180
python -m btc_ema_trader news
python -m btc_ema_trader train
if errorlevel 1 goto :fail
python -m btc_ema_trader reset-session
python -m btc_ema_trader dashboard
goto :eof
:fail
echo Retraining failed. Existing trained model was not deleted.
pause
exit /b 1
