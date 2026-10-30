@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo 请先安装 requirements-individual.txt 中的依赖，然后使用 python main.py 启动。
    pause
    exit /b 1
)
echo 正在启动 NeuroGuide，请稍候...
".venv\Scripts\python.exe" "tools\launch_gui.py"
if errorlevel 1 pause
