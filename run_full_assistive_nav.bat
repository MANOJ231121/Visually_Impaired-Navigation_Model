@echo off
cd /d "%~dp0"
set KMP_DUPLICATE_LIB_OK=TRUE
echo Launching the full Assistive-Navigation app with DroidCam, YOLO11s and voice...
echo Presentation voice: object detected about 20 metres ahead.
echo.
call "Assistive-Navigation\RUN_droidcam.bat" --weights models\YOLO11s\best.pt --voice-backend pyttsx3 --distance 20 --position ahead %*
