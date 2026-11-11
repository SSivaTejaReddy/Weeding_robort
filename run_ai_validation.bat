@echo off
setlocal

cd /d "%~dp0"

set "PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Virtual environment not found.
  echo Creating .venv with system Python...
  python -m venv .venv || goto :error
)

set "PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Could not find .venv Python at "%PYTHON%".
  goto :error
)

set "INTERACTIVE=0"
if "%~1"=="" (
  set "INTERACTIVE=1"
  goto :menu
)
set "ACTION=%~1"
goto :run

:menu
echo.
echo AI Validation Runner
echo ====================
echo 0. Exit
echo 1. Install dependencies
echo 2. Prepare selected-crop dataset
echo 3. Audit dataset
echo 4. Create dataset preview sheets
echo 5. Train onion selected-crop model
echo 6. Run detection on an image/video/camera
echo.
set /p ACTION="Choose 0-6: "

:run
if /i "%ACTION%"=="install" goto :install
if "%ACTION%"=="1" goto :install
if /i "%ACTION%"=="prepare" goto :prepare
if "%ACTION%"=="2" goto :prepare
if /i "%ACTION%"=="audit" goto :audit
if "%ACTION%"=="3" goto :audit
if /i "%ACTION%"=="preview" goto :preview
if "%ACTION%"=="4" goto :preview
if /i "%ACTION%"=="train" goto :train
if "%ACTION%"=="5" goto :train
if /i "%ACTION%"=="detect" goto :detect
if "%ACTION%"=="6" goto :detect
if /i "%ACTION%"=="exit" goto :quit
if "%ACTION%"=="0" goto :quit

echo Unknown option: %ACTION%
goto :error

:install
"%PYTHON%" -m pip install --timeout 1000 -r requirements-ai.txt || goto :error
goto :done

:prepare
"%PYTHON%" -m scripts.prepare_selected_crop_dataset --val-fraction 0.22 || goto :error
goto :done

:audit
"%PYTHON%" -m scripts.audit_yolo_dataset --classes 1 || goto :error
goto :done

:preview
"%PYTHON%" -m scripts.make_dataset_contact_sheet || goto :error
echo Preview saved to runs\dataset_review\all_contact_sheet.jpg
goto :done

:train
"%PYTHON%" -m scripts.train_yolo --data configs\onion_body_yolo26_dataset.yaml --task detect --epochs 50 --batch 8 --device cpu --name onion_body_yolo26_detect || goto :error
goto :done

:detect
set "DEFAULT_MODEL=weights\selected_crop\onion_body_yolo26_detect\best.pt"
if not exist "%DEFAULT_MODEL%" set "DEFAULT_MODEL=weights\selected_crop\onion\best.pt"
if not exist "%DEFAULT_MODEL%" set "DEFAULT_MODEL=runs\ai_validation\onion_selected_crop_clean\weights\best.pt"
if not exist "%DEFAULT_MODEL%" set "DEFAULT_MODEL=runs\segment\runs\ai_validation\onion_selected_crop_weak-2\weights\best.pt"
set /p SOURCE="Enter source path or camera index [0]: "
if "%SOURCE%"=="" set "SOURCE=0"
"%PYTHON%" -m scripts.run_detection --model "%DEFAULT_MODEL%" --source "%SOURCE%" || goto :error
goto :done

:done
echo.
echo Done.
pause
if "%INTERACTIVE%"=="1" goto :menu
exit /b 0

:error
echo.
echo Failed. Check the message above.
pause
if "%INTERACTIVE%"=="1" goto :menu
exit /b 1

:quit
echo.
echo Exiting.
exit /b 0
