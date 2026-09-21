@echo off
call "%~dp0launch_windows.cmd" --interaction tripod %*
exit /b %errorlevel%
