@echo off
setlocal
pushd "%~dp0"
where uv >nul 2>&1
if errorlevel 1 (
  echo Install uv first: winget install --id=astral-sh.uv -e
  echo Then open a new terminal and run this file again.
  echo https://docs.astral.sh/uv/getting-started/installation/
  pause
  popd
  exit /b 1
)
uv sync --locked --no-dev
if errorlevel 1 goto failed
uv run --no-sync pinchpilot desktop %*
if errorlevel 1 goto failed
popd
exit /b 0
:failed
echo PinchPilot could not start. Keep this error message for troubleshooting.
pause
popd
exit /b 1
