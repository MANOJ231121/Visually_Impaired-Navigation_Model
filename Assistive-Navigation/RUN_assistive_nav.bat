@echo off
setlocal
title Assistive Navigation prototype (research only)
set "HERE=%~dp0"
set "LOG=%HERE%checks\logs\last_run.log"
if not exist "%HERE%checks\logs" mkdir "%HERE%checks\logs" >nul 2>&1

echo ============================================
echo  Assistive Navigation prototype
echo  RESEARCH PROTOTYPE - not a mobility aid
echo ============================================
echo  A camera window should open in about 10 seconds.
echo  Press q inside that window to stop.
echo.
echo  Keys: q quit ^| v voice ^| +/- confidence
echo        s screenshot ^| m mute ^| t tracking
echo.
echo  Camera: default index 0 ^(laptop^)
echo          RUN_assistive_nav.bat --camera laptop
echo          RUN_assistive_nav.bat --camera droidcam
echo          RUN_assistive_nav.bat --list-cameras
echo.
echo  Or just double-click: RUN_laptop.bat or RUN_droidcam.bat
echo.
echo  Full log: %LOG%
echo ============================================
echo.

where conda >nul 2>&1
if errorlevel 1 (
  echo ERROR: conda was not found.
  echo Open "Anaconda Prompt" instead and run:
  echo   conda run -n yolo python "%HERE%src\main.py"
  echo.
  pause
  exit /b 1
)

rem `conda` is conda.bat on Windows: without `call`, control never returns
rem here, so the summary below and the final `pause` would never run.
call conda run --no-capture-output -n yolo python -u "%HERE%src\main.py" %* > "%LOG%" 2>&1
set "CODE=%ERRORLEVEL%"

if not "%CODE%"=="0" (
  echo.
  echo ============================================
  echo  THE APP FAILED  ^(exit code %CODE%^)
  echo ============================================
  echo.
  type "%LOG%"
  echo.
  echo  Log saved here: %LOG%
  echo ============================================
) else (
  echo App closed normally.
)

echo.
pause
