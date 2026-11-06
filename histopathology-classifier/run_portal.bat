@echo off
setlocal
title Histopathology Cancer Classifier
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_run.ps1"
if errorlevel 1 (
  echo.
  echo The portal could not start. Review the error above.
  pause
)
