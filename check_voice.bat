@echo off
cd /d "%~dp0"
set KMP_DUPLICATE_LIB_OK=TRUE

echo Windows voice check
echo -------------------
python -c "import pyttsx3; e=pyttsx3.init(); print('voices:'); [print(' -', getattr(v, 'name', v)) for v in e.getProperty('voices')]; e.setProperty('volume', 1.0); e.setProperty('rate', 165); e.say('Voice check. Object guidance is working.'); e.runAndWait(); print('voice test finished')"
if errorlevel 1 echo.
if errorlevel 1 echo Voice check failed. Check Windows audio output and pyttsx3 installation.
pause
