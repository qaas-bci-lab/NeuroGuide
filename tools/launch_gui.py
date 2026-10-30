"""Development launcher: hidden console worker, persistent logs, ready signal."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def launch(check=False):
    root = Path(__file__).resolve().parents[1]
    interpreter = root / ".venv/Scripts/python.exe"
    if not interpreter.is_file():
        raise FileNotFoundError("项目 .venv 中缺少 python.exe")
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    token = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    log_path = logs / f"startup_{token}.log"
    ready_path = logs / f"startup_{token}.ready"
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    environment["NEUROGUIDE_STARTUP_READY"] = str(ready_path)
    # Explorer does not inherit the activated Conda environment of a terminal.
    config = root / ".venv/pyvenv.cfg"
    for line in config.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "home":
            base = Path(value.strip())
            paths = [base / "Library/bin", base / "DLLs", base]
            environment["PATH"] = os.pathsep.join(str(p) for p in paths if p.is_dir()) + os.pathsep + environment.get("PATH", "")
    command = [str(interpreter), "-u", "-X", "faulthandler", str(root / "main.py")]
    if check:
        command.append("--startup-check")
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"NeuroGuide startup\nInterpreter: {interpreter}\nProject: {root}\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if ready_path.is_file():
            ready_path.unlink(missing_ok=True)
            if check:
                code = process.wait(timeout=10)
                if code:
                    raise RuntimeError(f"启动检查失败，日志：{log_path}")
            print(f"窗口初始化完成。启动日志：{log_path}")
            return 0
        code = process.poll()
        if code is not None:
            if code == 0:
                return 0
            message = f"NeuroGuide 启动失败。请查看日志：\n{log_path}"
            print(message)
            if os.name == "nt" and not check:
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, message, "NeuroGuide 启动失败", 0x10)
            return code or 1
        time.sleep(0.1)
    print(f"程序仍在初始化，详情记录在：{log_path}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(launch("--check" in sys.argv))
    except Exception as exc:
        print(f"无法启动 NeuroGuide：{exc}")
        raise SystemExit(1)
