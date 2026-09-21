@echo off
setlocal
pushd "%~dp0"
"runtime\python.exe" -B -X utf8 "portable.py" start
set "pinchpilot_exit=%errorlevel%"
if not "%pinchpilot_exit%"=="0" (
  echo.
  echo Startup failed. Run Diagnose.cmd or see the README for runtime troubleshooting.
  pause
)
popd
exit /b %pinchpilot_exit%
