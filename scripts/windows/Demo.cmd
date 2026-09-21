@echo off
setlocal
pushd "%~dp0"
"runtime\python.exe" -B -X utf8 "portable.py" demo
set "pinchpilot_exit=%errorlevel%"
if not "%pinchpilot_exit%"=="0" pause
popd
exit /b %pinchpilot_exit%
