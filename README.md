# Threat Action Recognition (TAR) Pipeline

Real-time action recognition and state classification pipeline combining **MediaPipe** (pose & hand landmarks) and **YOLOv8** (object detection) with a **BiLSTM + Attention** neural network.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.10** installed (Python 3.9–3.11 supported).
- Built-in or USB webcam (for live inference).

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone <YOUR-REPO-URL>
cd <REPO-FOLDER>

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (Command Prompt / PowerShell):
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Environment
Run the automated environment check to ensure your camera, PyTorch, and model weights are ready:
```bash
python verify_environment.py
```

---

## 🖥️ Running the Project

### A. Live Webcam Inference
To run live camera tracking with real-time HUD and decision stabilization:
```bash
python realtime.py
```
- Press `q` to exit.

### B. Offline Video Evaluation
To evaluate the model on pre-recorded video and generate a markdown/JSON report:
```bash
python test_video_tar.py --video videos/1.mp4
```

---

## 📂 Key Model Weights & Architecture
- **`best_tar_model.pth`**: BiLSTM + Self-Attention network trained on 7 action classes:
  - `idle`, `open_box`, `pick_red`, `place_red_out`, `pick_blue`, `place_blue_in`, `close_box`
- **`yolo_boxes.pt`**: Custom YOLOv8 weights for box and object bounding box detection.
