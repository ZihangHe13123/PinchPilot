@echo off
call "%~dp0launch_windows.cmd" --interaction finger-dwell %*
exit /b %errorlevel%
