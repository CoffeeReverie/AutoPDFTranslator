from __future__ import annotations

import atexit
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from autopdftranslator import APP_NAME, APP_VERSION


DESKTOP_VERSION = "0.1.0"
STREAMLIT_CHILD_ENV = "AUTOPDFTRANSLATOR_STREAMLIT_CHILD"


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_ready(url: str, timeout_sec: float = 45.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= int(response.status) < 500:
                    return True
        except Exception:
            time.sleep(0.4)
    return False


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=6)
    except subprocess.TimeoutExpired:
        process.kill()


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        return bundle_root
    return Path(__file__).resolve().parent


def _storage_root() -> Path:
    raw = os.getenv("AUTOPDFTRANSLATOR_STORAGE_DIR", "").strip()
    if raw:
        root = Path(raw).expanduser()
    elif sys.platform.startswith("win") and os.getenv("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / APP_NAME
    else:
        root = Path.home() / f".{APP_NAME.lower()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_streamlit_child() -> int:
    import streamlit.web.cli as stcli

    app_file = _app_root() / "app.py"
    if not app_file.exists():
        print(f"Cannot find Streamlit app file: {app_file}")
        return 2
    port = os.getenv("AUTOPDFTRANSLATOR_STREAMLIT_PORT", "8501")
    sys.argv = [
        "streamlit",
        "run",
        str(app_file),
        "--global.developmentMode=false",
        "--server.headless=true",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]
    return int(stcli.main() or 0)


def main() -> int:
    if os.getenv(STREAMLIT_CHILD_ENV) == "1":
        return _run_streamlit_child()

    try:
        import webview  # pywebview
    except Exception:
        print("pywebview is required for desktop mode. Install with: pip install pywebview")
        return 2

    app_root = _app_root()
    app_file = app_root / "app.py"
    if not app_file.exists():
        print(f"Cannot find Streamlit app file: {app_file}")
        return 2

    port = _pick_free_port()
    env = os.environ.copy()
    env["AUTOPDFTRANSLATOR_STORAGE_DIR"] = str(_storage_root())
    env["AUTOPDFTRANSLATOR_STREAMLIT_PORT"] = str(port)
    if getattr(sys, "frozen", False):
        env[STREAMLIT_CHILD_ENV] = "1"
        streamlit_cmd = [sys.executable]
    else:
        streamlit_cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_file),
            "--global.developmentMode=false",
            "--server.headless=true",
            f"--server.port={port}",
            "--browser.gatherUsageStats=false",
        ]
    process = subprocess.Popen(streamlit_cmd, cwd=str(app_root), env=env)
    atexit.register(_terminate_process, process)

    url = f"http://127.0.0.1:{port}"
    if not _wait_until_ready(url):
        _terminate_process(process)
        print("Failed to start embedded Streamlit server.")
        return 2

    title = f"{APP_NAME} Desktop v{DESKTOP_VERSION} (Core {APP_VERSION})"
    webview.create_window(title, url, width=1500, height=950, min_size=(1100, 700))
    webview.start()

    _terminate_process(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
