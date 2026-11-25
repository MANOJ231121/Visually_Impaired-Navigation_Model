@echo off
cd /d "%~dp0"
set KMP_DUPLICATE_LIB_OK=TRUE

echo Voice demo with DroidCam
echo -------------------------
echo Make sure the DroidCam Client on Windows shows a live phone preview.
echo Distance and motion are demo values; speech is a real Windows voice.
echo Press q in the video window to quit.
echo.
python assistive\live_guide.py --weights Assistive-Navigation\models\best.pt --names class_names.json --source droidcam --voice-backend pyttsx3 --motion-mode left --position-mode auto --distance 20 --conf 0.35
if errorlevel 1 echo.
if errorlevel 1 echo Demo failed. Run check_voice.bat and check config\cameras.json index.
pause
