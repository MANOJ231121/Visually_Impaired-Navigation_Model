@echo off
cd /d "%~dp0"
set KMP_DUPLICATE_LIB_OK=TRUE

if "%~1"=="" (
    echo Usage: run_demo_image.bat "path\to\person.jpg"
    echo.
    pause
    exit /b 1
)

echo Starting image demo with YOLO11s: %~1
echo Press any key in the preview window to continue.
echo.
python assistive\live_guide.py --weights models\YOLO11s\best.pt --names class_names.json --source "%~1" --motion-mode left --position-mode ahead
if errorlevel 1 echo.
if errorlevel 1 echo Demo failed. Run check_demo.bat and send the output.
pause
