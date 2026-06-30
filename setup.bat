@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo ==========================================
echo  Musubi LoRA GUI - Setup
echo ==========================================
echo.

:: ================================================================
:: Python 検出 (3.12 -> 3.11 -> 3.10 の優先順)
:: py ランチャー優先、なければ PATH の python を使用
:: ================================================================
set "PYTHON_CMD="

:: py ランチャー経由で 3.12 を試す
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 --version >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3.12"
        goto :python_found
    )
    py -3.11 --version >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3.11"
        goto :python_found
    )
    py -3.10 --version >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3.10"
        goto :python_found
    )
)

:: py ランチャーがない場合は PATH の python を確認
where python >nul 2>nul
if not errorlevel 1 (
    python -c "import sys; v=sys.version_info; exit(0 if (v.major==3 and v.minor>=10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
        goto :python_found
    )
)

echo [ERROR] Python 3.10 or later not found.
echo         Install from https://www.python.org/ and enable "Add to PATH".
echo.
pause
exit /b 1

:python_found
echo [OK] Python command: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

:: ================================================================
:: git 確認
:: ================================================================
where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] git not found in PATH.
    echo         Install Git from https://git-scm.com/
    echo.
    pause
    exit /b 1
)
echo [OK] git found.
echo.

:: ================================================================
:: git clone (初回のみ)
:: ================================================================
set "MUSUBI_DIR=musubi-tuner"
if not exist "%MUSUBI_DIR%\.git" (
    echo [INFO] Cloning kohya-ss/musubi-tuner ...
    git clone https://github.com/kohya-ss/musubi-tuner.git "%MUSUBI_DIR%"
    if errorlevel 1 (
        echo [ERROR] git clone failed.
        pause
        exit /b 1
    )
    echo [OK] Cloned.
    echo.
) else (
    echo [OK] musubi-tuner already cloned.
)

:: ================================================================
:: venv 構築 (musubi-tuner\.venv)
:: ================================================================
set "VENV_DIR=%MUSUBI_DIR%\.venv"
set "PY=%VENV_DIR%\Scripts\python.exe"

if exist "%PY%" (
    echo [OK] venv already exists, skipping install.
    echo      Delete %VENV_DIR% to reinstall.
    echo.
    goto :verify
)

echo [INFO] Creating venv in %VENV_DIR% ...
%PYTHON_CMD% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [ERROR] venv creation failed.
    pause
    exit /b 1
)
echo [OK] venv created.
echo.

echo [INFO] Upgrading pip ...
"%PY%" -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 (
    echo [ERROR] pip upgrade failed.
    pause
    exit /b 1
)
echo [OK] pip upgraded.
echo.

:: ── PyTorch cu128 優先、失敗時 cu124 フォールバック ──────────
echo [INFO] Installing PyTorch cu128 ...
"%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [WARN] cu128 failed. Retrying with cu124 ...
    "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo [ERROR] PyTorch install failed.
        pause
        exit /b 1
    )
    echo [OK] PyTorch cu124 installed.
) else (
    echo [OK] PyTorch cu128 installed.
)
echo.

:: ── musubi-tuner 本体 ────────────────────────────────────────
echo [INFO] Installing musubi-tuner ...
"%PY%" -m pip install -e "%MUSUBI_DIR%[cu128]"
if errorlevel 1 (
    echo [WARN] Editable install failed. Trying requirements.txt ...
    if exist "%MUSUBI_DIR%\requirements.txt" (
        "%PY%" -m pip install -r "%MUSUBI_DIR%\requirements.txt"
        if errorlevel 1 (
            echo [ERROR] requirements.txt install failed.
            pause
            exit /b 1
        )
    )
)
echo.

:: ── GUI 追加依存 ─────────────────────────────────────────────
echo [INFO] Installing GUI extras (matplotlib, pillow) ...
"%PY%" -m pip install matplotlib pillow
if errorlevel 1 (
    echo [WARN] GUI extras failed - charts will not work.
)
echo.

:: ================================================================
:verify
echo ==========================================
echo  Verifying environment
echo ==========================================
"%PY%" -c "import torch; print('torch       :', torch.__version__, '| CUDA:', torch.cuda.is_available())"
"%PY%" -c "import accelerate; print('accelerate  :', accelerate.__version__)"
"%PY%" -c "import tkinter; print('tkinter     : OK')"
echo.
echo ==========================================
echo  Setup complete. Run start.bat to launch.
echo ==========================================
echo.
pause
