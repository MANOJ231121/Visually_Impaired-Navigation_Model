@echo off
cd /d "%~dp0"
set KMP_DUPLICATE_LIB_OK=TRUE
echo Launching the full Assistive-Navigation app with DroidCam and voice...
echo.
call "Assistive-Navigation\RUN_droidcam.bat" %*
