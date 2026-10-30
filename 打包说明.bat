@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "NEUROGUIDE_RELEASE_NAME=NeuroGuide-V4.5"
set "TEMP=%~dp0build\manual_package\temp"
set "TMP=%TEMP%"
set "MPLCONFIGDIR=%~dp0build\manual_package\mpl_cache"
set "PYINSTALLER_CONFIG_DIR=%~dp0build\manual_package\pyinstaller_cache"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"
if not exist "%PYINSTALLER_CONFIG_DIR%" mkdir "%PYINSTALLER_CONFIG_DIR%"
if not exist "build\protected_V4_5\venv\Scripts\python.exe" (
    echo 请先使用目标 CUDA 环境的 Python 执行 tools\prepare_protected_env.py。
    echo 详细步骤见 docs\release_V4_5_protection.md。
    pause
    exit /b 1
)
echo 正在编译 NeuroGuide-V4.5 受保护模块并构建 GUI 与推理 Worker...
"build\protected_V4_5\venv\Scripts\python.exe" -u -X utf8 tools\build_protected.py --stage "build\manual_package\protected_V4_5" --distpath "build\manual_package\distribution"
if errorlevel 1 (
    echo 打包失败，请查看上方日志。
) else (
    echo 构建完成：build\manual_package\distribution\NeuroGuide-V4.5\NeuroGuide-V4.5.exe
    echo 请验证后再发布，保留整个目录及 NeuroGuideWorker.exe。
    echo 此脚本不会替换 dist 中的发布包或个体任务数据。
)
pause
