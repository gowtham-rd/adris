#!/usr/bin/env python3
# main_app.py

import json
import os
import time
import signal
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

TARGET_FPS = float(CONFIG["runtime"]["target_fps"])
SHARED_FRAME_PATH = CONFIG["paths"]["shared_frame"]
SHARED_JSON_PATH = CONFIG["paths"]["shared_json"]
ENGINE_PATH = CONFIG["mode"]["engine_path"]

CONF_THRESHOLD = float(CONFIG.get("detection", {}).get("conf_threshold", 0.3))

MODEL_W = 640
MODEL_H = 640


# ------------------------------------------------------------
# Graceful Shutdown
# ------------------------------------------------------------

_STOP = False

def _stop_handler(signum, frame):
    global _STOP
    _STOP = True

signal.signal(signal.SIGINT, _stop_handler)
signal.signal(signal.SIGTERM, _stop_handler)


# ------------------------------------------------------------
# Atomic JSON Write
# ------------------------------------------------------------

def write_json_atomic(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


# ------------------------------------------------------------
# TensorRT Engine
# ------------------------------------------------------------

class TRTInference:

    def __init__(self, engine_path):

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine")

        self.context = self.engine.create_execution_context()

        self.input_index = 0
        self.output_index = 1

        self.input_shape = tuple(self.engine.get_binding_shape(self.input_index))
        self.output_shape = tuple(self.engine.get_binding_shape(self.output_index))

        self.input_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_index))
        self.output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_index))

        self.d_input = cuda.mem_alloc(
            int(np.prod(self.input_shape)) * np.dtype(self.input_dtype).itemsize
        )

        self.d_output = cuda.mem_alloc(
            int(np.prod(self.output_shape)) * np.dtype(self.output_dtype).itemsize
        )

        self.output_host = cuda.pagelocked_empty(
            self.output_shape,
            dtype=self.output_dtype
        )

        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.input_index] = int(self.d_input)
        self.bindings[self.output_index] = int(self.d_output)

        self.stream = cuda.Stream()

        print("Engine Loaded")
        print("Input shape:", self.input_shape)
        print("Output shape:", self.output_shape)


    def infer(self, inp):

        inp = np.ascontiguousarray(inp, dtype=self.input_dtype)

        if tuple(inp.shape) != self.input_shape:
            raise ValueError(f"Input shape mismatch: {inp.shape} vs {self.input_shape}")

        cuda.memcpy_htod_async(self.d_input, inp, self.stream)

        ok = self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        if not ok:
            raise RuntimeError("TensorRT execution failed")

        cuda.memcpy_dtoh_async(self.output_host, self.d_output, self.stream)
        self.stream.synchronize()

        return self.output_host


# ------------------------------------------------------------
# Image Preprocessing
# ------------------------------------------------------------

def preprocess_image(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Frame not found: {path}")

    img = Image.open(path).convert("RGB")
    img = img.resize((MODEL_W, MODEL_H))

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    return np.ascontiguousarray(arr, dtype=np.float32)


# ------------------------------------------------------------
# Detection Decode (Basic Confidence Filter)
# ------------------------------------------------------------

def decode_output(output):

    detections = []

    out = output[0]  # (25200, 6)

    for row in out:
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
    print("Engine:", ENGINE_PATH)
    print("Target FPS:", TARGET_FPS)

    trt_engine = TRTInference(ENGINE_PATH)

    frame_interval = 1.0 / TARGET_FPS

    while not _STOP:

        loop_start = time.time()
        timestamp = datetime.now().astimezone().isoformat()

        try:

            inp = preprocess_image(SHARED_FRAME_PATH)

            t0 = time.time()
            output = trt_engine.infer(inp)
            t1 = time.time()

            latency_ms = (t1 - t0) * 1000.0
            detections = decode_output(output)

            cpu = psutil.cpu_percent() if psutil else "N/A"
            mem = psutil.virtual_memory().percent if psutil else "N/A"

            payload = {
                "timestamp": timestamp,
                "detections": detections,
                "latency_ms": float(latency_ms),
                "fps": float(1.0 / max(time.time() - loop_start, 1e-9)),
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

    print("ADRIS stopped cleanly.")


if __name__ == "__main__":
    main()