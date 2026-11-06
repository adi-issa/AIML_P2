@echo off
setlocal
title VibeCode Pathology Portal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_run.ps1"
if errorlevel 1 (
  echo.
  echo The portal could not start. Review the error above.
  pause
)
