@echo off
REM End-to-end RETFound Plus-LAFT-XAI pipeline on real EyePACS data (Windows).
REM   run_all.bat          full run
REM   run_all.bat quick    small subset, for a fast smoke test
setlocal enabledelayedexpansion
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

set MODE=%1
if "%MODE%"=="" set MODE=full

REM Batch 16 is the throughput sweet spot on 8 GB accelerators. On a CUDA GPU
REM with >=12 GB, raise --batch-size to 32-64 and --samples-per-epoch to 24000.
set BATCH=16
if /i "%MODE%"=="quick" (
  set LIMIT=--limit 4000
  set EPOCHS=2
  set SPE=3000
  set CMP_TRAIN=2000
  set CMP_EPOCHS=1
) else (
  set LIMIT=
  set EPOCHS=4
  set SPE=11000
  set CMP_TRAIN=4000
  set CMP_EPOCHS=1
)

echo ==============================================================
echo  STEP 0  self-test: every block A1-A10 on synthetic tensors
echo ==============================================================
"%PY%" scripts\08_selftest.py || goto :failed

echo ==============================================================
echo  STEP 1  download real EyePACS 2015 from Kaggle
echo ==============================================================
"%PY%" scripts\01_download_data.py --dataset eyepacs-224 || goto :failed

echo ==============================================================
echo  STEP 2  A1 quality gate + A2 preprocessing + lesion priors
echo ==============================================================
"%PY%" -u scripts\02_preprocess.py --root data\raw\eyepacs-224 %LIMIT% || goto :failed

echo ==============================================================
echo  STEP 3  train (AL-LoRA + MoE + graph fusion + ordinal + XAI)
echo ==============================================================
"%PY%" -u scripts\03_train.py --epochs %EPOCHS% --samples-per-epoch %SPE% ^
    --max-val 2000 --batch-size %BATCH% || goto :failed

echo ==============================================================
echo  STEP 4  evaluate: 38 parameters across 7 metric groups
echo ==============================================================
set EXT=
if exist data\cache_external set EXT=--external-cache data\cache_external
"%PY%" -u scripts\04_evaluate.py %EXT% || goto :failed

echo ==============================================================
echo  STEP 5  single-image clinical report (A1..A10 end to end)
echo ==============================================================
for /f "delims=" %%F in ('dir /b /s data\raw\eyepacs-224\*.png 2^>nul') do (
  set IMG=%%F
  goto :gotimg
)
:gotimg
"%PY%" -u scripts\05_predict.py --image "!IMG!" || goto :failed

echo ==============================================================
echo  STEP 6  deployment: distillation, FP16/INT8, ONNX
echo ==============================================================
"%PY%" -u scripts\06_deploy.py --epochs 2 || goto :failed

echo ==============================================================
echo  STEP 7  model comparison + component ablations
echo ==============================================================
"%PY%" -u scripts\07_compare_models.py --epochs %CMP_EPOCHS% --max-train %CMP_TRAIN% ^
    --max-test 2000 || goto :failed

echo ==============================================================
echo  STEP 9  screening triage: val-fitted referral operating points
echo ==============================================================
"%PY%" -u scripts\09_screening.py %EXT% || goto :failed

echo.
echo All steps complete. Artefacts in outputs\
goto :eof

:failed
echo.
echo FAILED at the step above (exit code %errorlevel%).
exit /b %errorlevel%
