@echo off
cd /d "%~dp0"
set KMP_DUPLICATE_LIB_OK=TRUE

echo Phone camera demo
echo -----------------
echo Android example: http://192.168.1.15:8080/video
echo RTSP example:    rtsp://192.168.1.15:8080/h264_ulaw.sdp
echo.
set /P "URL=Paste your phone stream URL: "

echo.
echo Starting demo with YOLO11s. Press q in the preview window to quit.
python assistive\live_guide.py --weights models\YOLO11s\best.pt --names class_names.json --source "%URL%" --motion-mode left --position-mode ahead
if errorlevel 1 echo.
if errorlevel 1 echo Demo failed. Check that the phone and PC are on the same Wi-Fi.
pause
