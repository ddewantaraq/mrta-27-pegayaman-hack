@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
  echo Usage: assess.bat cand.csv [--skip-metrica] [--fetch] [--public-log] [--ref path]
  exit /b 1
)

call python starter\assess.py --cand %*
if errorlevel 1 (
  echo FAIL  assess
  exit /b 1
)
exit /b 0
