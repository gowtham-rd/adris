#!/usr/bin/env python3
# main_app.py

import json
import os
import time
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

try:
    import psutil
except Exception:
    psutil = None


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "board_config.json"

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

TARGET_FPS = CONFIG["runtime"]["target_fps"]
SHARED_FRAME_PATH = CONFIG["paths"]["shared_frame"]
SHARED_JSON_PATH = CONFIG["paths"]["shared_json"]
ENGINE_PATH = CONFIG["mode"]["engine_path"]

CONF_THRESHOLD = CONFIG.get("detection", {}).get("conf_threshold", 0.3)


# ------------------------------------------------------------
# Atomic JSON write
# ------------------------------------------------------------

def write_json_atomic(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


# ------------------------------------------------------------
# TensorRT Engine Wrapper
# ------------------------------------------------------------

class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        self.input_index = 0
        self.output_index = 1

        self.input_shape = tuple(self.engine.get_binding_shape(self.input_index))
        self.output_shape = tuple(self.engine.get_binding_shape(self.output_index))

        self.d_input = cuda.mem_alloc(np.prod(self.input_shape) * 4)
        self.d_output = cuda.mem_alloc(np.prod(self.output_shape) * 4)

        self.output_host = np.empty(self.output_shape, dtype=np.float32)

        self.bindings = [None] * self.engine.num_bindings
        self.bindings[self.input_index] = int(self.d_input)
        self.bindings[self.output_index] = int(self.d_output)

        self.stream = cuda.Stream()

        print("Engine Loaded")
        print("Input shape:", self.input_shape)
        print("Output shape:", self.output_shape)

    def infer(self, inp):
        inp = np.ascontiguousarray(inp, dtype=np.float32)

        cuda.memcpy_htod_async(self.d_input, inp, self.stream)

        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        cuda.memcpy_dtoh_async(self.output_host, self.d_output, self.stream)
        self.stream.synchronize()

        return self.output_host


# ------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((640, 640))

    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    return np.ascontiguousarray(arr, dtype=np.float32)


# ------------------------------------------------------------
# Detection decoding (basic confidence filter only)
# ------------------------------------------------------------

def decode_output(output):
    detections = []

    output = output[0]  # (25200, 6)

    for row in output:
        conf = float(row[4])
        if conf < CONF_THRESHOLD:
            continue

        x, y, w, h = row[0:4]
        class_id = int(row[5])

        detections.append({
            "class_id": class_id,
            "confidence": conf,
            "bbox_xywh": [
                float(x),
                float(y),
                float(w),
                float(h)
            ]
        })

    return detections


# ------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------

def main():
    print("ADRIS Inference Started")

    trt_engine = TRTInference(ENGINE_PATH)

    frame_interval = 1.0 / TARGET_FPS

    while True:
        loop_start = time.time()

        timestamp = datetime.now().astimezone().isoformat()

        try:
            # 1. Load image
            inp = preprocess_image(SHARED_FRAME_PATH)

            # 2. Inference
            t0 = time.time()
            output = trt_engine.infer(inp)
            t1 = time.time()

            latency_ms = (t1 - t0) * 1000.0

            # 3. Decode
            detections = decode_output(output)

            # 4. System stats
            cpu = psutil.cpu_percent() if psutil else "N/A"
            mem = psutil.virtual_memory().percent if psutil else "N/A"

            payload = {
                "timestamp": timestamp,
                "detections": detections,
                "latency_ms": latency_ms,
                "fps": 1.0 / (time.time() - loop_start),
                "system": {
                    "cpu_percent": cpu,
                    "memory_percent": mem
                }
            }

        except Exception as e:
            payload = {
                "timestamp": timestamp,
                "detections": [],
                "latency_ms": "N/A",
                "fps": "N/A",
                "system": {},
                "error": str(e)
            }

        write_json_atomic(payload, SHARED_JSON_PATH)

        elapsed = time.time() - loop_start
        sleep_time = max(0, frame_interval - elapsed)
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()