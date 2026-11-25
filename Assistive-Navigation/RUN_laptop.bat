@echo off
rem Double-clickable launcher for the laptop / built-in webcam.
rem Extra arguments are passed through, e.g. --conf 0.2
call "%~dp0RUN_assistive_nav.bat" --camera laptop --weights "%~dp0..\models\YOLO11s\best.pt" %*
