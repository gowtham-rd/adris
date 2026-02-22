#!/usr/bin/env bash
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

rm -f /dev/shm/adris_cam_*.jpg >/dev/null 2>&1 || true
rm -f "${PUBLISH_PATH}" "${PUBLISH_TMP}" >/dev/null 2>&1 || true

gst-launch-1.0 -e \
  nvarguscamerasrc ! \
  "video/x-raw(memory:NVMM),width=1280,height=720,framerate=${FPS}/1" ! \
  nvvidconv top=0 bottom=720 left=280 right=1000 ! \
  "video/x-raw,width=640,height=640,format=I420" ! \
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

while true; do
  newest="$(ls -1t /dev/shm/adris_cam_*.jpg 2>/dev/null | head -n 1)"
  if [ -n "${newest}" ] && [ -f "${newest}" ]; then
    cp -f "${newest}" "${PUBLISH_TMP}"
    mv -f "${PUBLISH_TMP}" "${PUBLISH_PATH}"
  fi
  sleep 0.02
done