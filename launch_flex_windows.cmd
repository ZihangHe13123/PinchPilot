@echo off
call "%~dp0launch_windows.cmd" --interaction finger-flex %*
exit /b %errorlevel%
