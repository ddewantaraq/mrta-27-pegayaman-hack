@echo off
setlocal
cd /d "%~dp0"

echo === predict ===
call python starter\predict_v1.py > "%TEMP%\mtah_predict_v1.log" 2>&1
set "PREDICT_ERR=%errorlevel%"
type "%TEMP%\mtah_predict_v1.log"
if not "%PREDICT_ERR%"=="0" (
  echo FAIL  predict
  exit /b 1
)

set "SUBMISSION="
for /f "tokens=1* delims= " %%A in ('findstr /b /c:"SUBMISSION " "%TEMP%\mtah_predict_v1.log"') do (
  set "SUBMISSION=%%B"
)
if not defined SUBMISSION (
  echo FAIL  predict did not report a submission path
  exit /b 1
)

echo === validate ===
call python evaluation\validate_submission.py "%SUBMISSION%"
if errorlevel 1 (
  echo FAIL  validate
  exit /b 1
)

echo === score ===
call python evaluation\run_evaluation.py
if errorlevel 1 (
  echo FAIL  score
  exit /b 1
)

echo OK  predicted, validated, and scored %SUBMISSION%
exit /b 0
