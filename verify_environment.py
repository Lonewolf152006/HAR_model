"""
Environment & Dependency Verification Script
Run this script on a new machine to verify that all dependencies,
model weights, and hardware (camera/GPU) are ready for TAR inference.

Usage:
    python verify_environment.py
"""

import sys
import os

def check_mark(status):
    return "[\033[92mPASS\033[0m]" if status else "[\033[91mFAIL\033[0m]"

def run_verification():
    print("=" * 60)
    print("   TAR PIPELINE: SYSTEM & ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    all_ok = True
    
    # 1. Python version
    py_ver = sys.version_info
    py_ok = (3, 8) <= (py_ver.major, py_ver.minor) <= (3, 11)
    print(f"{check_mark(py_ok)} Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if not py_ok:
        print("       ⚠️  Recommended: Python 3.10 (3.9 - 3.11). Python 3.12+ may have MediaPipe issues.")
        if py_ver.minor > 11:
            all_ok = False

    # 2. Package Imports
    packages = [
        ("torch", "PyTorch (Deep Learning)"),
        ("cv2", "OpenCV (Computer Vision)"),
        ("mediapipe", "MediaPipe (Pose & Hand Tracking)"),
        ("ultralytics", "Ultralytics YOLO (Object Detection)"),
        ("numpy", "NumPy (Array Computations)"),
        ("sklearn", "Scikit-Learn (Metrics)"),
    ]
    
    for pkg_name, desc in packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"{check_mark(True)} {desc} ({pkg_name} {ver})")
        except ImportError as e:
            print(f"{check_mark(False)} {desc} ({pkg_name}) - NOT INSTALLED")
            all_ok = False

    # 3. Hardware acceleration (CUDA / CPU)
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU only"
        print(f"[{'INFO'}] PyTorch Device: {device_name} (CUDA available: {cuda_avail})")
    except Exception:
        pass

    # 4. Check Model Weight Files
    required_files = [
        ("best_tar_model.pth", "Trained Action Classifier (TAR Model)"),
        ("yolo_boxes.pt", "Custom Box/Object Detector Weights"),
        ("yolov8n.pt", "Base YOLOv8n Weights (Fallback)"),
    ]
    print("-" * 60)
    for fname, desc in required_files:
        exists = os.path.exists(fname)
        size_mb = (os.path.getsize(fname) / (1024 * 1024)) if exists else 0
        status_str = f"found ({size_mb:.1f} MB)" if exists else "MISSING!"
        print(f"{check_mark(exists)} {desc}: {fname} -> {status_str}")
        if not exists and fname != "yolov8n.pt":
            all_ok = False

    # 5. Check Camera (Optional check)
    print("-" * 60)
    try:
        import cv2
        # Use DirectShow backend on Windows to avoid long timeouts
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(0, backend)
        cam_ok = cap.isOpened()
        if cam_ok:
            ret, frame = cap.read()
            cap.release()
            print(f"{check_mark(ret)} Camera Index 0: Accessible")
        else:
            print(f"[{'WARN'}] Camera Index 0: Not accessible (Normal if running headless or video-only)")
    except Exception as e:
        print(f"[{'WARN'}] Camera check skipped: {e}")

    print("=" * 60)
    if all_ok:
        print("  ALL CORE CHECKS PASSED! You are ready to run:")
        print("    - Real-time webcam: python realtime.py")
        print("    - Video test:       python test_video_tar.py")
    else:
        print("  SOME CHECKS FAILED. Please review the errors above.")
        print("  Run: pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    run_verification()
