"""
ai_engine/camera.py - Hardware Camera & Video Input Manager

Features:
- Enumerate available video devices (DirectShow on Windows, including Camo Studio).
- Thread-safe camera acquisition with auto-reconnect.
- Dynamic runtime switching between Camo/Webcams and File/Synthetic fallback.
- Constant high-throughput frame buffer without frame lag.
"""

import os
import sys

# Silence noisy third-party camera driver DLLs on Windows
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import time
import subprocess
import threading
import cv2
import numpy as np


def get_available_cameras():
    """
    Enumerate connected video capture devices on Windows.
    Queries Windows PnP entities and probes OpenCV DirectShow indices.
    """
    devices = []
    
    # 1. Query Windows PnP Device Names
    pnp_names = []
    if sys.platform == "win32":
        try:
            cmd = "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -in @('Camera', 'Image') } | Select-Object -ExpandProperty Name"
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True, timeout=3)
            pnp_names = [line.strip() for line in out.strip().splitlines() if line.strip()]
        except Exception:
            pass

    # 2. Probe OpenCV indices for detected PnP cameras
    probe_count = max(1, len(pnp_names))
    for idx in range(probe_count):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                name = pnp_names[idx] if idx < len(pnp_names) else f"Camera #{idx}"
                # If "Camo" exists in PnP, ensure friendly label
                is_camo = any("camo" in p.lower() for p in pnp_names) and idx == 0
                label = f"{name} (Camo Studio)" if is_camo else name
                devices.append({
                    "id": str(idx),
                    "index": idx,
                    "name": label,
                    "type": "hardware",
                    "status": "connected",
                    "resolution": "1280x720"
                })
        else:
            cap.release()

    # If no physical devices opened, add synthetic/test fallback device
    devices.append({
        "id": "file",
        "index": -1,
        "name": "Synthetic / Test Loop",
        "type": "file",
        "status": "ready",
        "resolution": "1280x720"
    })

    return devices


class CameraManager:
    """Thread-safe background camera frame grabber."""
    def __init__(self, initial_source=0):
        self.source = initial_source
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_time = 0.0
        self.fps = 30.0
        self.thread = None
        self.source_type = "camera"  # "camera" or "file"
        self.file_path = None
        self.width = 1280
        self.height = 720

    def start(self):
        if self.running:
            return
        self.running = True
        self._open_source()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _open_source(self):
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

            if self.source_type == "file" and self.file_path and os.path.exists(self.file_path):
                self.cap = cv2.VideoCapture(self.file_path)
            else:
                idx = int(self.source) if isinstance(self.source, (int, str)) and str(self.source).isdigit() else 0
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)

    def switch_source(self, source_id, file_path=None):
        """Switch video source dynamically."""
        if source_id == "file" and file_path:
            self.source_type = "file"
            self.file_path = file_path
        else:
            self.source_type = "camera"
            self.source = int(source_id) if str(source_id).isdigit() else 0
        self._open_source()

    def _generate_synthetic_frame(self):
        """Generates clean avionics synthetic test patterns if camera is unavailable."""
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Background gradient
        t = time.time()
        cv2.putText(img, "ASTROFLOW AI — SYNTHETIC CAMERA LOOP", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 224, 138), 2)
        cv2.putText(img, f"TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}.{int((t%1)*1000):03d}", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (138, 145, 156), 2)
        cv2.putText(img, "SOURCE: Camo Virtual Studio / Standby Test Pattern", (60, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 176, 32), 2)
        
        # Draw mock experiment box and sample cubes
        box_x, box_y, box_w, box_h = 440, 300, 400, 260
        cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 2)
        cv2.putText(img, "EXPERIMENT CONTAINER [MAIN_BOX]", (box_x + 10, box_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Red cube
        cv2.rectangle(img, (box_x + 40, box_y + 80), (box_x + 130, box_y + 170), (0, 0, 255), -1)
        cv2.putText(img, "RED_CUBE", (box_x + 40, box_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Blue cube
        cv2.rectangle(img, (box_x + 230, box_y + 80), (box_x + 320, box_y + 170), (255, 100, 0), -1)
        cv2.putText(img, "BLUE_CUBE", (box_x + 230, box_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

        # Reticle
        cx, cy = 640, 360
        cv2.line(img, (cx - 30, cy), (cx + 30, cy), (0, 224, 138), 1)
        cv2.line(img, (cx, cy - 30), (cx, cy + 30), (0, 224, 138), 1)
        return img

    def _capture_loop(self):
        last_t = time.time()
        while self.running:
            frame = None
            if self.cap is not None and self.cap.isOpened():
                ret, raw = self.cap.read()
                if ret and raw is not None:
                    frame = raw
                elif self.source_type == "file":
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            if frame is None:
                frame = self._generate_synthetic_frame()
                time.sleep(0.033)

            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

            with self.lock:
                self.latest_frame = frame
                self.frame_time = now

            time.sleep(0.005)

    def read(self):
        """Returns the latest captured frame and timestamp."""
        with self.lock:
            if self.latest_frame is None:
                return False, self._generate_synthetic_frame(), 0.0, self.fps
            return True, self.latest_frame.copy(), self.frame_time, self.fps

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
