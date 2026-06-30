@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

set "PY=musubi-tuner\.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [ERROR] venv not found. Please run setup.bat first.
    pause
    exit /b 1
)

set "PYTHONUTF8=1"
"%PY%" -m app.main %*

set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% neq 0 (
    echo.
    echo Application exited with code %EXIT_CODE%
    pause
)
exit /b %EXIT_CODE%
