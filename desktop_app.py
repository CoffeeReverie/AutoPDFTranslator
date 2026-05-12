from __future__ import annotations

import atexit
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from autopdftranslator import APP_NAME, APP_VERSION


DESKTOP_VERSION = "0.1.0"


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


def main() -> int:
    try:
        import webview  # pywebview
    except Exception:
        print("pywebview is required for desktop mode. Install with: pip install pywebview")
        return 2

    app_root = Path(__file__).resolve().parent
    app_file = app_root / "app.py"
    if not app_file.exists():
        print(f"Cannot find Streamlit app file: {app_file}")
        return 2

    port = _pick_free_port()
    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_file),
        "--server.headless=true",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]
    process = subprocess.Popen(streamlit_cmd, cwd=str(app_root))
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
