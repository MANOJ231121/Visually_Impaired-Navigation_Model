@echo off
setlocal
set "KMP_DUPLICATE_LIB_OK=TRUE"
rem Double-clickable launcher for the DroidCam phone camera.
rem Voice is forced to audible pyttsx3 so it cannot silently use console text.
rem Extra arguments are passed through and can override these defaults.
call "%~dp0RUN_assistive_nav.bat" --camera droidcam --voice-backend pyttsx3 --volume 1.0 --rate 165 --cooldown 2 --min-gap 0.8 %*
