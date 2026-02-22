# ADRIS – Edge-Based Real-Time Person Detection System

ADRIS is an embedded real-time person detection system designed for deployment on NVIDIA Jetson Nano.  
The system performs GPU-accelerated inference using YOLOv5s optimized with TensorRT and provides a live LAN-accessible dashboard with bounding box overlay and performance monitoring.

---

## 🔹 System Overview

Camera (CSI IMX219)  
→ Frame Processing (Center Crop + Resize 640×640)  
→ TensorRT Inference (best.engine)  
→ JSON Payload Generation  
→ MJPEG Streaming + REST API  
→ Dashboard + CSV Logging  

---

## 🔹 Project Structure

adris/
├── config/
│   └── board_config.json
├── dashboard/
│   ├── dashboard_server.py
│   ├── static/
│   │   ├── main.js
│   │   └── style.css
│   └── templates/
│       └── index.html
├── logs/
│   ├── predictions_log.csv
│   └── watchdog.log
├── model/
│   ├── best.onnx
│   └── best.engine
├── main_app.py
├── start.py
├── watchdog.sh
├── requirements.txt
└── readme.md
└── folderstructure.md
└── camera_writer.sh

---

## 🔹 Installation (Jetson Nano)

1. Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate

	2.	Install dependencies:

pip install -r requirements.txt

TensorRT and CUDA libraries must be installed via JetPack.

⸻

🔹 Running the System

Start the full system:

python3 start.py

Dashboard will be accessible on LAN:

http://<jetson-ip>:5050


⸻

🔹 Shared Memory Files

Inference writes:
	•	/dev/shm/adris_latest.jpg
	•	/dev/shm/adris_latest.json

Dashboard reads these for streaming and statistics.

⸻

🔹 Logging

Detection events are appended to:

logs/predictions_log.csv

Each detected person generates one CSV row containing:
	•	timestamp
	•	confidence
	•	bounding box
	•	latency
	•	fps
	•	CPU usage
	•	memory usage

⸻

🔹 Design Principles
	•	No cloud dependency
	•	No OpenCV
	•	File-based inter-process communication
	•	LAN deployment ready
	•	Robust to camera disconnect
	•	15 FPS streaming target

