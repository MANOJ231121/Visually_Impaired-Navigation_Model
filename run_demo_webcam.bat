@echo off
cd /d "%~dp0"
set KMP_DUPLICATE_LIB_OK=TRUE

echo Starting webcam demo. Press q in the preview window to quit.
echo.
python assistive\live_guide.py --source 0 --motion-mode left --position-mode ahead
if errorlevel 1 echo.
if errorlevel 1 echo Demo failed. Run check_demo.bat and send the output.
pause
