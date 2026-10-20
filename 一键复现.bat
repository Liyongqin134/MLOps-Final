@echo off
chcp 65001 >nul
title 一键复现 · 手写数字识别
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境 .venv，请先运行：
    echo        python -m venv .venv
    echo        .venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt
    pause
    exit /b 1
)

echo ============================================================
echo  一键复现：训练模型 + 打开可视化界面（MLflow + FastAPI）
echo ============================================================
echo.

".venv\Scripts\python.exe" reproduce.py %*

echo.
pause
