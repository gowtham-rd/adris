#!/usr/bin/env python3
# main_app.py

import json
import time
import os
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
import psutil

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "board_config.json"

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

ENGINE_PATH = PROJECT_ROOT / "model" / "best.engine"
FRAME_PATH = "/dev/shm/adris_latest.jpg"

TARGET_FPS = CONFIG["runtime"]["target_fps"]
CONF_THRESHOLD = 0.4

SHARED_JSON_PATH = CONFIG["paths"]["shared_json"]


# ------------------------------------------------------------
# Atomic JSON write
# ------------------------------------------------------------

def write_json_atomic(data, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)


# ------------------------------------------------------------
# TensorRT Setup
# ------------------------------------------------------------

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

with open(str(ENGINE_PATH), "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

input_index = engine.get_binding_index("images")
output_index = engine.get_binding_index("output0")

input_shape = engine.get_binding_shape(input_index)
output_shape = engine.get_binding_shape(output_index)

input_size = trt.volume(input_shape)
output_size = trt.volume(output_shape)

d_input = cuda.mem_alloc(input_size * np.float32().nbytes)
d_output = cuda.mem_alloc(output_size * np.float32().nbytes)

bindings = [int(d_input), int(d_output)]


# ------------------------------------------------------------
# Preprocess
# ------------------------------------------------------------

def preprocess_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((640, 640))
    img = np.array(image, dtype=np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # add batch dim
    return img


# ------------------------------------------------------------
# YOLO Decode
# ------------------------------------------------------------

def decode_output(output):
    detections = []

    output = output.reshape(25200, 6)

    for row in output:
        x, y, w, h, conf, cls = row
        if conf < CONF_THRESHOLD:
            continue

        detections.append({
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h),
            "confidence": float(conf),
            "class_id": int(cls)
        })

    return detections


# ------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------

def main():
    print("ADRIS TensorRT Inference Started")
    print(f"Engine: {ENGINE_PATH}")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}\n")

    frame_interval = 1.0 / TARGET_FPS

    while True:
        loop_start = time.time()

        if not os.path.exists(FRAME_PATH):
            time.sleep(0.1)
            continue

        try:
            img = preprocess_image(FRAME_PATH)

            cuda.memcpy_htod(d_input, img)
            start = time.time()
            context.execute_v2(bindings)
            cuda.memcpy_dtoh(np.zeros(output_size, dtype=np.float32), d_output)
            latency = (time.time() - start) * 1000

            output_host = np.empty(output_size, dtype=np.float32)
            cuda.memcpy_dtoh(output_host, d_output)

            detections = decode_output(output_host)

            payload = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "detections": detections,
                "latency_ms": latency,
                "fps": 1000.0 / latency if latency > 0 else 0,
                "system": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent
                }
            }

            write_json_atomic(payload, SHARED_JSON_PATH)

        except Exception as e:
            print("Inference error:", e)

        elapsed = time.time() - loop_start
        sleep_time = max(0, frame_interval - elapsed)
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()