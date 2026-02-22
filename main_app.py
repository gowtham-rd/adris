#!/usr/bin/env python3
# main_app.py

"""
ADRIS Inference Application (Skeleton Version)

This version:
- Loads configuration
- Simulates runtime loop
- Writes valid JSON payload to shared memory
- Does NOT run TensorRT yet
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime
import psutil

# ---------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "board_config.json"

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

TARGET_FPS = CONFIG["runtime"]["target_fps"]
FRAME_WIDTH = CONFIG["runtime"]["frame_width"]
FRAME_HEIGHT = CONFIG["runtime"]["frame_height"]

SHARED_JSON_PATH = CONFIG["paths"]["shared_json"]

# ---------------------------------------------------------------------
# Utility: Atomic JSON Write
# ---------------------------------------------------------------------

def write_json_atomic(data, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)

# ---------------------------------------------------------------------
# Main Loop (Simulation Only)
# ---------------------------------------------------------------------

def main():
    print("ADRIS Inference Skeleton Started")
    print(f"Target FPS (stream reference): {TARGET_FPS}")
    print(f"Frame Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print("Waiting for real TensorRT integration...\n")

    frame_interval = 1.0 / TARGET_FPS

    while True:
        start_time = time.time()

        # Simulated inference latency
        simulated_latency = 0.050  # 50 ms
        time.sleep(simulated_latency)

        # Simulated detection (empty by default)
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "detections": [],  # no detection in skeleton
            "latency_ms": simulated_latency * 1000,
            "fps": 1.0 / simulated_latency,
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent
            }
        }

        write_json_atomic(payload, SHARED_JSON_PATH)

        elapsed = time.time() - start_time
        sleep_time = max(0, frame_interval - elapsed)
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()