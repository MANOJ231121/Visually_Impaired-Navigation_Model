@echo off
cd /d "%~dp0"
set KMP_DUPLICATE_LIB_OK=TRUE

echo Starting webcam demo with YOLO11s. Press q in the preview window to quit.
echo.
python assistive\live_guide.py --weights models\YOLO11s\best.pt --names class_names.json --source 0 --motion-mode left --position-mode ahead
if errorlevel 1 echo.
if errorlevel 1 echo Demo failed. Run check_demo.bat and send the output.
pause
