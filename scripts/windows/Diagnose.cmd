@echo off
setlocal
pushd "%~dp0"
"runtime\python.exe" -B -X utf8 "portable.py" diagnose
set "pinchpilot_exit=%errorlevel%"
echo.
echo Reports are in reports\diagnostics. No camera or mouse control was used.
pause
popd
exit /b %pinchpilot_exit%
