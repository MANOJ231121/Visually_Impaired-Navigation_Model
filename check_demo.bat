@echo off
cd /d "%~dp0"
set KMP_DUPLICATE_LIB_OK=TRUE

echo Python:
python --version
echo.
python -c "import sys, ultralytics, cv2, pyttsx3; print('python:', sys.version); print('ultralytics:', ultralytics.__version__); print('opencv:', cv2.__version__); print('pyttsx3: ok')"
echo.
python assistive\live_guide.py --help
echo.
pause
