#!/usr/bin/env bash
# camera_writer.sh
#
# ADRIS Camera Writer (Robust Version)
#
# Fixes vs earlier version:
# - GStreamer writes numbered frames: /dev/shm/adris_cam_%05d.jpg
# - A tiny loop publishes the newest frame atomically to /dev/shm/adris_latest.jpg
#   (prevents partial reads and guarantees continuous updates)
#
# Output contract (LOCKED):
# - /dev/shm/adris_latest.jpg is always a valid 640x640 JPEG and updates continuously

set -euo pipefail

FPS="15"

TMP_PATTERN="/dev/shm/adris_cam_%05d.jpg"
PUBLISH_PATH="/dev/shm/adris_latest.jpg"
PUBLISH_TMP="/dev/shm/adris_latest.jpg.tmp"

echo "[INFO] ADRIS camera writer starting..."
echo "[INFO] FPS: ${FPS}"
echo "[INFO] GStreamer pattern: ${TMP_PATTERN}"
echo "[INFO] Publish path: ${PUBLISH_PATH}"

command -v gst-launch-1.0 >/dev/null 2>&1 || {
  echo "[ERROR] gst-launch-1.0 not found."
  exit 1
}

if [ ! -d "/dev/shm" ]; then
  echo "[ERROR] /dev/shm not found."
  exit 1
fi

# Clean old frames
rm -f /dev/shm/adris_cam_*.jpg || true

# Start GStreamer in background
# Crop math:
# 1280x720 -> center 720x720:
# left=280 right=1000 top=0 bottom=720
gst-launch-1.0 -e \
  nvarguscamerasrc ! \
  video/x-raw(memory:NVMM),width=1280,height=720,framerate=${FPS}/1 ! \
  nvvidconv top=0 bottom=720 left=280 right=1000 ! \
  video/x-raw,width=640,height=640,format=I420 ! \
  jpegenc ! \
  multifilesink location="${TMP_PATTERN}" max-files=3 sync=false \
  >/dev/null 2>&1 &

GST_PID=$!

cleanup() {
  echo "[INFO] Stopping camera writer..."
  if kill -0 "${GST_PID}" >/dev/null 2>&1; then
    kill "${GST_PID}" >/dev/null 2>&1 || true
  fi
  rm -f /dev/shm/adris_cam_*.jpg >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# Publisher loop: always publish newest frame atomically
while true; do
  newest="$(ls -1t /dev/shm/adris_cam_*.jpg 2>/dev/null | head -n 1 || true)"
  if [ -n "${newest}" ] && [ -f "${newest}" ]; then
    # Atomic publish to fixed path
    cp -f "${newest}" "${PUBLISH_TMP}"
    mv -f "${PUBLISH_TMP}" "${PUBLISH_PATH}"
  fi
  sleep 0.02
done