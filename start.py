#!/usr/bin/env python3
# start.py
#
# ADRIS System Launcher (Updated)
#
# Starts (in this order):
# 1) camera_writer.sh
# 2) dashboard backend
# 3) inference app
#
# Critical fix:
# - Each process runs in its own process group
# - On shutdown we kill the entire group (prevents lingering gst-launch)

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "board_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

PORT = CONFIG["network"]["port"]
SHARED_FRAME_PATH = CONFIG["paths"]["shared_frame"]
SHARED_JSON_PATH = CONFIG["paths"]["shared_json"]

CAMERA_WRITER = PROJECT_ROOT / "camera_writer.sh"
DASHBOARD_SERVER = PROJECT_ROOT / "dashboard" / "dashboard_server.py"
INFERENCE_APP = PROJECT_ROOT / "main_app.py"


# ---------------------------------------------------------------------
# Process Helpers
# ---------------------------------------------------------------------

def start_process(cmd, name) -> subprocess.Popen:
    """
    Start a subprocess in its own process group so we can kill the group.
    """
    print(f"[START] {name}: {' '.join(map(str, cmd))}")
    return subprocess.Popen(
        list(map(str, cmd)),
        preexec_fn=os.setsid,  # Linux: new process group
        stdout=None,
        stderr=None
    )


def terminate_process_group(proc: Optional[subprocess.Popen], name: str) -> None:
    """
    Terminate the whole process group (TERM, then KILL).
    """
    if proc is None:
        return
    if proc.poll() is not None:
        return

    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None

    print(f"[STOP] {name}")

    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            pass
    else:
        try:
            proc.terminate()
        except Exception:
            pass

    # Wait then force kill
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        print(f"[KILL] {name}")
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass
        else:
            try:
                proc.kill()
            except Exception:
                pass


def wait_for_file(path: str, timeout_s: float, name: str) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        if os.path.exists(path):
            print(f"[OK] {name} ready: {path}")
            return True
        time.sleep(0.1)
    print(f"[WARN] {name} not ready after {timeout_s:.1f}s: {path}")
    return False


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("\n==============================")
    print(" ADRIS SYSTEM LAUNCHER")
    print("==============================\n")
    print(f"[INFO] Dashboard (LAN): http://<jetson-ip>:{PORT}")
    print(f"[INFO] Shared frame: {SHARED_FRAME_PATH}")
    print(f"[INFO] Shared json : {SHARED_JSON_PATH}\n")

    camera_proc: Optional[subprocess.Popen] = None
    dashboard_proc: Optional[subprocess.Popen] = None
    inference_proc: Optional[subprocess.Popen] = None

    try:
        # 1) Camera writer
        if not CAMERA_WRITER.exists():
            raise FileNotFoundError(f"Missing {CAMERA_WRITER}")

        try:
            CAMERA_WRITER.chmod(CAMERA_WRITER.stat().st_mode | 0o111)
        except Exception:
            pass

        camera_proc = start_process(["bash", str(CAMERA_WRITER)], "Camera Writer")
        wait_for_file(SHARED_FRAME_PATH, timeout_s=6.0, name="Camera frame")

        # 2) Dashboard server
        if not DASHBOARD_SERVER.exists():
            raise FileNotFoundError(f"Missing {DASHBOARD_SERVER}")

        dashboard_proc = start_process([sys.executable, str(DASHBOARD_SERVER)], "Dashboard Server")
        time.sleep(1.0)

        # 3) Inference app
        if not INFERENCE_APP.exists():
            raise FileNotFoundError(f"Missing {INFERENCE_APP}")

        inference_proc = start_process([sys.executable, str(INFERENCE_APP)], "Inference App")
        wait_for_file(SHARED_JSON_PATH, timeout_s=6.0, name="Inference JSON")

        print("\n[OK] ADRIS started successfully.")
        print("[INFO] Press Ctrl+C to stop.\n")

        # Monitor: if any dies, stop all
        while True:
            if camera_proc.poll() is not None:
                print("[ERROR] Camera Writer exited unexpectedly.")
                break
            if dashboard_proc.poll() is not None:
                print("[ERROR] Dashboard Server exited unexpectedly.")
                break
            if inference_proc.poll() is not None:
                print("[ERROR] Inference App exited unexpectedly.")
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received. Shutting down...")

    finally:
        # Stop in reverse order
        terminate_process_group(inference_proc, "Inference App")
        terminate_process_group(dashboard_proc, "Dashboard Server")
        terminate_process_group(camera_proc, "Camera Writer")
        print("[DONE] ADRIS stopped.\n")


if __name__ == "__main__":
    main()