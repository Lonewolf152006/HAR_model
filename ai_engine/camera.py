"""
ai_engine/camera.py - Hardware Camera & Video Input Manager

Features:
- Enumerate ALL available video capture devices (DirectShow via pygrabber on Windows).
- Support USB webcams, Camo Studio, OBS Virtual Camera, NVIDIA Broadcast, etc.
- Dynamic runtime switching between any camera device index or Synthetic Test Loop.
- Standby warning overlay if a connected camera sends completely blank/black frames.
"""

import os
import sys

# Silence noisy third-party camera driver DLLs on Windows
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import time
import threading
import cv2
import numpy as np


def get_available_cameras():
    """
    Enumerate connected video capture devices on Windows.
    Uses DirectShow FilterGraph enumeration for friendly names.
    """
    devices = []
    
    if sys.platform == "win32":
        try:
            from pygrabber.dshow_graph import FilterGraph
            graph = FilterGraph()
            dev_names = graph.get_input_devices()
            for idx, name in enumerate(dev_names):
                devices.append({
                    "id": str(idx),
                    "index": idx,
                    "name": name,
                    "type": "hardware",
                    "status": "connected",
                    "resolution": "1280x720" if "camo" in name.lower() else "640x480"
                })
        except Exception:
            pass

    # Fallback if pygrabber didn't populate
    if not devices:
        for idx in range(3):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    devices.append({
                        "id": str(idx),
                        "index": idx,
                        "name": f"Video Device #{idx}",
                        "type": "hardware",
                        "status": "connected",
                        "resolution": "640x480"
                    })
            else:
                cap.release()

    # Always include Synthetic / Test Loop fallback
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
    """Thread-safe background camera frame grabber supporting all video devices."""
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
        self.active_device_name = "Camera #0"
        self.is_blank_frame = False

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
                self.active_device_name = "Synthetic Test Video"
            elif self.source_type == "file":
                self.active_device_name = "Synthetic Test Pattern"
                self.cap = None
            else:
                idx = int(self.source) if isinstance(self.source, (int, str)) and str(self.source).isdigit() else 0
                backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
                self.cap = cv2.VideoCapture(idx, backend)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)

                # Identify name
                try:
                    cams = get_available_cameras()
                    matched = next((c for c in cams if c["id"] == str(idx)), None)
                    if matched:
                        self.active_device_name = matched["name"]
                    else:
                        self.active_device_name = f"Camera #{idx}"
                except Exception:
                    self.active_device_name = f"Camera #{idx}"

    def switch_source(self, source_id, file_path=None):
        """Switch video source dynamically."""
        print(f"[CAMERA] Switching video source to: {source_id}")
        if source_id == "file":
            self.source_type = "file"
            self.file_path = file_path
            self.source = "file"
        else:
            self.source_type = "camera"
            self.source = int(source_id) if str(source_id).isdigit() else 0
        self._open_source()

    def _generate_synthetic_frame(self):
        """Generates clean avionics synthetic test patterns if camera is unavailable or in test mode."""
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        t = time.time()
        cv2.putText(img, "ASTROFLOW AI — SYNTHETIC VIDEO FEED", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 224, 138), 2)
        cv2.putText(img, f"TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}.{int((t%1)*1000):03d}", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (138, 145, 156), 2)
        cv2.putText(img, f"ACTIVE SOURCE: {self.active_device_name} (STANDBY TEST PATTERN)", (60, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 176, 32), 2)
        
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
                    # Check if the device is outputting a totally black screen (e.g. Camo app idle / phone locked)
                    mean_val = float(raw.mean())
                    if mean_val < 0.5:
                        self.is_blank_frame = True
                        # Overlay helpful instruction so user isn't looking at an empty black void
                        h, w, _ = raw.shape
                        cv2.rectangle(raw, (w//2 - 360, h//2 - 60), (w//2 + 360, h//2 + 60), (20, 24, 30), -1)
                        cv2.rectangle(raw, (w//2 - 360, h//2 - 60), (w//2 + 360, h//2 + 60), (255, 176, 32), 2)
                        cv2.putText(raw, f"[{self.active_device_name.upper()}] NO VIDEO SIGNAL", (w//2 - 330, h//2 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 176, 32), 2)
                        cv2.putText(raw, "Check that Camo app on your phone is open & streaming,", (w//2 - 330, h//2 + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 233, 237), 1)
                        cv2.putText(raw, "or switch to OBS Virtual Camera / Test Mode in Settings.", (w//2 - 330, h//2 + 38),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 224, 138), 1)
                    else:
                        self.is_blank_frame = False
                    frame = raw
                elif self.source_type == "file" and self.cap:
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            if frame is None:
                self.is_blank_frame = False
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
        """Returns the latest captured frame, timestamp, and device metadata."""
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
