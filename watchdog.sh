#!/usr/bin/env bash
# watchdog.sh
#
# ADRIS Watchdog (Skeleton)
# - Runs start.py in a loop
# - Restarts it if it exits unexpectedly
# - Logs events to logs/watchdog.log
#
# Usage:
#   chmod +x watchdog.sh
#   ./watchdog.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${PROJECT_DIR}/logs/watchdog.log"

PYTHON_BIN="${PROJECT_DIR}/venv/bin/python3"
START_SCRIPT="${PROJECT_DIR}/start.py"

mkdir -p "${PROJECT_DIR}/logs"

echo "[$(date -Iseconds)] Watchdog started" | tee -a "${LOG_FILE}"

while true; do
  echo "[$(date -Iseconds)] Launching ADRIS via start.py" | tee -a "${LOG_FILE}"

  # Run ADRIS (blocking). If it exits, watchdog restarts.
  if [ -x "${PYTHON_BIN}" ]; then
    "${PYTHON_BIN}" "${START_SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"
  else
    # Fallback if venv not present
    python3 "${START_SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"
  fi

  echo "[$(date -Iseconds)] ADRIS stopped or crashed. Restarting in 3 seconds..." | tee -a "${LOG_FILE}"
  sleep 3
done