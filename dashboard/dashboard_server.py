#!/usr/bin/env python3
# dashboard/dashboard_server.py
#
# ADRIS Dashboard Backend (Production Version)
#
# Key design fixes vs previous version:
# - NO state updates / CSV logging inside /video_feed generator
#   (prevents duplication when multiple clients connect)
# - A SINGLE background thread:
#   - reads /dev/shm/adris_latest.jpg and /dev/shm/adris_latest.json
#   - overlays bounding boxes server-side (Pillow)
#   - updates cached stats/logs/performance history
#   - appends CSV rows (detections-only)
# - /video_feed simply streams the latest cached annotated JPEG
# - All /api/* endpoints serve cached state
#
# Dependencies: flask, pillow

import io
import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, render_template
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------
# Load Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "board_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

HOST = CONFIG["network"]["host"]
PORT = CONFIG["network"]["port"]

TARGET_FPS = float(CONFIG["runtime"]["target_fps"])
FRAME_WIDTH = int(CONFIG["runtime"]["frame_width"])
FRAME_HEIGHT = int(CONFIG["runtime"]["frame_height"])

SHARED_FRAME_PATH = str(CONFIG["paths"]["shared_frame"])
SHARED_JSON_PATH = str(CONFIG["paths"]["shared_json"])
CSV_LOG_PATH = PROJECT_ROOT / CONFIG["paths"]["csv_log"]

# Background update frequency. Keep it aligned with target stream rate.
UPDATER_HZ = TARGET_FPS
UPDATER_SLEEP = 1.0 / max(1.0, UPDATER_HZ)

# Stale timeout (seconds). If JSON is older than this, treat as stale.
STALE_JSON_S = 2.0

# Buffer sizes
RECENT_LOGS_MAX = 200
PERF_MAX = 180
LAT_WINDOW_MAX = 60

# ---------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")


# ---------------------------------------------------------------------
# Cached State (shared between background thread + request handlers)
# ---------------------------------------------------------------------

state_lock = threading.Lock()

cached_annotated_jpeg: bytes = b""
cached_last_frame_ok: bool = False

recent_logs = deque(maxlen=RECENT_LOGS_MAX)           # list of dicts for /api/logs
performance_history = deque(maxlen=PERF_MAX)          # list of {cpu, memory} for chart
latency_window = deque(maxlen=LAT_WINDOW_MAX)         # numeric latency values

total_detections: int = 0
last_fps: float = 0.0

last_payload_timestamp_processed: Optional[str] = None  # ensures we don't double-log

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _ensure_csv_header() -> None:
    """
    Ensure logs/predictions_log.csv exists and has header.
    """
    CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CSV_LOG_PATH.exists() or CSV_LOG_PATH.stat().st_size == 0:
        header = "timestamp,class,confidence,x,y,width,height,latency_ms,fps,cpu_percent,memory_percent\n"
        CSV_LOG_PATH.write_text(header, encoding="utf-8")


def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_read_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def _parse_iso_datetime(ts: str) -> Optional[datetime]:
    """
    Parse ISO-8601 with timezone when possible.
    """
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _time_hhmmss(ts: str) -> str:
    """
    Convert timestamp to HH:MM:SS for UI logs; falls back safely.
    """
    dt = _parse_iso_datetime(ts) if ts else None
    if dt:
        return dt.strftime("%H:%M:%S")
    return datetime.now().strftime("%H:%M:%S")


def _load_font(size: int = 16) -> ImageFont.ImageFont:
    """
    Load a readable font if available; fallback to default.
    """
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT = _load_font(16)


def _generate_no_signal_frame() -> bytes:
    """
    Generate a "NO SIGNAL" JPEG in memory.
    """
    img = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (25, 25, 25))
    draw = ImageDraw.Draw(img)

    text = "NO SIGNAL"
    bbox = draw.textbbox((0, 0), text, font=FONT)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x = (FRAME_WIDTH - tw) // 2
    y = (FRAME_HEIGHT - th) // 2

    # red background block + white text
    pad = 8
    draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(180, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=FONT)

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=80)
    return out.getvalue()


def _overlay_detections(jpeg_bytes: bytes, payload: Dict[str, Any]) -> bytes:
    """
    Overlay person boxes + label on a JPEG frame. Uses Pillow only.
    bbox format: [x, y, w, h] in pixels on 640x640.
    """
    try:
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    except Exception:
        return jpeg_bytes

    draw = ImageDraw.Draw(img)

    detections = payload.get("detections", [])
    if not isinstance(detections, list):
        detections = []

    for det in detections:
        if not isinstance(det, dict):
            continue
        if det.get("class") != "person":
            continue

        bbox = det.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue

        try:
            x, y, w, h = [int(round(float(v))) for v in bbox]
        except Exception:
            continue

        conf = det.get("confidence", 0.0)
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0

        # rectangle
        x2, y2 = x + w, y + h
        draw.rectangle([x, y, x2, y2], outline="red", width=3)

        # label with background
        label = f"person {int(conf_f * 100)}%"
        tb = draw.textbbox((0, 0), label, font=FONT)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = 4

        # place above the box if possible; otherwise inside
        ly = y - (th + 2 * pad)
        if ly < 0:
            ly = y + 2
        lx = max(0, min(x, FRAME_WIDTH - (tw + 2 * pad)))

        draw.rectangle([lx, ly, lx + tw + 2 * pad, ly + th + 2 * pad], fill="red")
        draw.text((lx + pad, ly + pad), label, fill="white", font=FONT)

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85, optimize=True)
    return out.getvalue()


def _payload_is_stale(payload: Dict[str, Any]) -> bool:
    ts = payload.get("timestamp")
    if not isinstance(ts, str) or not ts:
        return True

    dt = _parse_iso_datetime(ts)
    if not dt:
        return False  # if we can't parse, don't mark stale aggressively

    age = (datetime.now(dt.tzinfo) - dt).total_seconds()
    return age > STALE_JSON_S


# ---------------------------------------------------------------------
# Background Updater (single source of truth)
# ---------------------------------------------------------------------

def _background_updater() -> None:
    """
    Single updater loop:
    - reads latest frame + payload
    - overlays detections
    - updates cached annotated frame
    - updates stats/logs/perf history
    - appends CSV rows (detections-only)
    """
    global cached_annotated_jpeg, cached_last_frame_ok
    global total_detections, last_fps
    global last_payload_timestamp_processed

    _ensure_csv_header()

    # Initialize cached frame so /video_feed always has something
    with state_lock:
        cached_annotated_jpeg = _generate_no_signal_frame()
        cached_last_frame_ok = False

    while True:
        frame_bytes = _safe_read_bytes(SHARED_FRAME_PATH)
        payload = _safe_read_json(SHARED_JSON_PATH)

        # If no frame, show NO SIGNAL. Keep API state unchanged unless payload is valid.
        if frame_bytes is None:
            annotated = _generate_no_signal_frame()
            with state_lock:
                cached_annotated_jpeg = annotated
                cached_last_frame_ok = False
            time.sleep(UPDATER_SLEEP)
            continue

        # If payload missing or stale, stream raw frame without overlay.
        if payload is None or not isinstance(payload, dict) or _payload_is_stale(payload):
            with state_lock:
                cached_annotated_jpeg = frame_bytes
                cached_last_frame_ok = True
            time.sleep(UPDATER_SLEEP)
            continue

        # Overlay detections
        annotated = _overlay_detections(frame_bytes, payload)

        # Update cached annotated frame
        with state_lock:
            cached_annotated_jpeg = annotated
            cached_last_frame_ok = True

        # Process payload ONCE per unique timestamp (prevents double-logging on slow updates)
        ts = payload.get("timestamp")
        if isinstance(ts, str) and ts and ts != last_payload_timestamp_processed:
            last_payload_timestamp_processed = ts

            # Update rolling stats
            fps_val = payload.get("fps", 0.0)
            lat_ms = payload.get("latency_ms", 0.0)

            try:
                last_fps = float(fps_val)
            except Exception:
                last_fps = 0.0

            try:
                lat_ms_f = float(lat_ms)
            except Exception:
                lat_ms_f = 0.0

            with state_lock:
                latency_window.append(lat_ms_f)

            # Performance history (optional)
            sysinfo = payload.get("system", {})
            cpu = None
            mem = None
            if isinstance(sysinfo, dict):
                cpu = sysinfo.get("cpu_percent", None)
                mem = sysinfo.get("memory_percent", None)

            try:
                cpu_f = float(cpu) if cpu is not None else 0.0
            except Exception:
                cpu_f = 0.0

            try:
                mem_f = float(mem) if mem is not None else 0.0
            except Exception:
                mem_f = 0.0

            with state_lock:
                performance_history.append({"cpu": cpu_f, "memory": mem_f})

            # Logs + CSV (detections-only)
            detections = payload.get("detections", [])
            if not isinstance(detections, list):
                detections = []

            time_str = _time_hhmmss(ts)

            # Collect CSV lines and apply state updates under lock once
            csv_lines: List[str] = []
            new_log_entries: List[Dict[str, Any]] = []
            new_detections_count = 0

            for det in detections:
                if not isinstance(det, dict):
                    continue
                if det.get("class") != "person":
                    continue

                bbox = det.get("bbox")
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue

                try:
                    x, y, w, h = [int(round(float(v))) for v in bbox]
                except Exception:
                    continue

                conf = det.get("confidence", 0.0)
                try:
                    conf_f = float(conf)
                except Exception:
                    conf_f = 0.0

                new_detections_count += 1

                new_log_entries.append({
                    "time": time_str,           # UI expects a displayable time string
                    "class": "person",
                    "confidence": conf_f,
                    "inference_time": lat_ms_f
                })

                csv_lines.append(
                    f"{ts},person,{conf_f},{x},{y},{w},{h},{lat_ms_f},{last_fps},{cpu_f},{mem_f}\n"
                )

            if new_detections_count > 0:
                with state_lock:
                    # Update totals and in-memory logs
                    nonlocal_total = new_detections_count
                    # (explicit variable to keep it clear)
                    total_detections_update = nonlocal_total
                    total_detections += total_detections_update

                    for entry in new_log_entries:
                        recent_logs.appendleft(entry)

                # Append CSV outside lock (I/O can block)
                try:
                    with open(CSV_LOG_PATH, "a", encoding="utf-8") as f:
                        f.writelines(csv_lines)
                except Exception:
                    # If CSV logging fails, keep system running
                    pass

        time.sleep(UPDATER_SLEEP)


# Start background updater thread once, at import time (safe for Flask run)
_updater_thread = threading.Thread(target=_background_updater, daemon=True)
_updater_thread.start()


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """
    MJPEG stream.
    This MUST NOT update stats/logs.
    It only streams the latest cached JPEG.
    """
    def generate():
        delay = 1.0 / max(1.0, TARGET_FPS)
        while True:
            with state_lock:
                frame = cached_annotated_jpeg or _generate_no_signal_frame()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

            time.sleep(delay)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def api_stats():
    with state_lock:
        avg_latency = (sum(latency_window) / len(latency_window)) if latency_window else 0.0
        return jsonify({
            "fps": round(float(last_fps), 2),
            "avg_inference_time": round(float(avg_latency), 2),
            "detections_count": int(total_detections)
        })


@app.route("/api/logs")
def api_logs():
    with state_lock:
        return jsonify({"logs": list(recent_logs)})


@app.route("/api/detection_stats")
def api_detection_stats():
    with state_lock:
        return jsonify({"class_counts": {"person": int(total_detections)}})


@app.route("/api/performance_history")
def api_performance_history():
    with state_lock:
        return jsonify(list(performance_history))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False, threaded=True)