"""
ai_engine/server.py - FastAPI Real-Time Edge Video & Telemetry Server

Runs on port 8080.
Bridges Python computer vision inference with the Next.js mission console.
"""

import os
import sys
import time
import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Ensure root directory is on python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine.pipeline import AstroFlowPipeline
from contextlib import asynccontextmanager
from ai_engine.camera import get_available_cameras

# Global Pipeline Singleton
pipeline = None
flight_logs = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("[SERVER] Initializing AstroFlowPipeline...")
    pipeline = AstroFlowPipeline()
    pipeline.start()
    print("[SERVER] AstroFlow AI FastAPI Edge Hub running on http://0.0.0.0:8080")
    yield
    if pipeline:
        pipeline.stop()

app = FastAPI(title="AstroFlow AI Edge Inference Server", version="1.0.0", lifespan=lifespan)

# Enable CORS for Next.js console
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 1. Live Video Feed (MJPEG Stream)
# ---------------------------------------------------------------------------
async def generate_mjpeg_frames():
    """Async generator for multipart/x-mixed-replace MJPEG video stream."""
    while True:
        if pipeline:
            frame_bytes = pipeline.get_latest_jpeg()
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
        await asyncio.sleep(0.033)  # ~30 FPS


@app.get("/video_feed")
async def video_feed():
    """Returns continuous multipart/x-mixed-replace annotated video stream."""
    return StreamingResponse(
        generate_mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ---------------------------------------------------------------------------
# 2. WebSocket Telemetry Stream (10Hz)
# ---------------------------------------------------------------------------
@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    last_frame_id = -1

    # Attach speech listener to forward speech events to browser Web Speech
    def on_voice(text):
        try:
            asyncio.create_task(
                websocket.send_text(json.dumps({"msg_type": "VOICE_PROMPT", "text": text}))
            )
        except Exception:
            pass

    pipeline.voice.on_speech_event_callback = on_voice

    try:
        while True:
            telemetry = pipeline.get_latest_telemetry()
            if telemetry and telemetry["frame_id"] != last_frame_id:
                last_frame_id = telemetry["frame_id"]

                # Record log events for state transitions / alerts
                if telemetry.get("is_transition") or telemetry.get("active_alert"):
                    level = "ALERT" if telemetry.get("active_alert") else "ACCEPTED"
                    log_entry = {
                        "id": f"log-{len(flight_logs)+1}",
                        "timestamp": telemetry["timestamp"],
                        "level": level,
                        "state": telemetry["current_state"],
                        "message": f"STATE -> {telemetry['current_state'].upper()} (p={telemetry['confidence']:.2f})" if level == "ACCEPTED" else f"VIOLATION: {telemetry['active_alert']['reason']}",
                        "confidence": telemetry["confidence"]
                    }
                    flight_logs.append(log_entry)
                    if len(flight_logs) > 1000:
                        flight_logs.pop(0)

                await websocket.send_text(json.dumps({
                    "msg_type": "TELEMETRY",
                    "data": telemetry
                }))
            await asyncio.sleep(0.08)  # ~10Hz
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Disconnected: {e}")
    finally:
        pipeline.voice.on_speech_event_callback = None


# ---------------------------------------------------------------------------
# 3. REST Endpoints: State, Health & Hardware
# ---------------------------------------------------------------------------
@app.get("/health")
def get_health():
    return {
        "status": "nominal",
        "node": "ISS-COLUMBUS-HAR",
        "models": {
            "tar_bilstm": "loaded",
            "yolov8_boxes": "loaded" if pipeline.yolo else "missing"
        },
        "camera": {
            "status": "streaming" if pipeline.camera.running else "offline",
            "fps": round(pipeline.camera.fps, 1)
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/state")
def get_current_state():
    telemetry = pipeline.get_latest_telemetry()
    if not telemetry:
        return {
            "ok": True,
            "data": {
                "currentState": "idle",
                "expectedNext": "open_box",
                "gates": {},
                "containment": {"red_box": "INSIDE", "blue_box": "INSIDE", "main_box": "CLOSED"},
                "capturedAt": datetime.now().isoformat()
            }
        }
    return {
        "ok": True,
        "data": {
            "currentState": telemetry["current_state"],
            "expectedNext": telemetry["expected_next"],
            "gates": telemetry["gates"],
            "containment": telemetry["containment"],
            "capturedAt": telemetry["timestamp"]
        },
        "error": None,
        "meta": {"requestId": f"req_{int(time.time()*1000)}", "timestamp": datetime.now().isoformat()}
    }


class GateUpdateRequest(BaseModel):
    confidence: Optional[float] = None
    stability: Optional[int] = None
    cooldown: Optional[float] = None
    motion: Optional[float] = None


@app.get("/api/v1/settings/gates")
def get_gates():
    return {
        "confidence": pipeline.stabilizer.confidence_threshold,
        "stability": pipeline.stabilizer.stability_window,
        "cooldown": pipeline.stabilizer.cooldown_sec,
        "motion": pipeline.stabilizer.motion_floor
    }


@app.put("/api/v1/settings/gates")
def update_gates(req: GateUpdateRequest):
    pipeline.update_gate_thresholds(
        confidence=req.confidence,
        stability=req.stability,
        cooldown=req.cooldown,
        motion=req.motion
    )
    return {"ok": True, "settings": get_gates()}


@app.get("/api/v1/camera/devices")
def list_cameras():
    devices = get_available_cameras()
    return {"ok": True, "devices": devices, "active_source": pipeline.camera.source}


class CameraSelectRequest(BaseModel):
    device_id: str


@app.put("/api/v1/camera/select")
def select_camera(req: CameraSelectRequest):
    pipeline.switch_camera(req.device_id)
    return {"ok": True, "selected": req.device_id}


class SpeechRequest(BaseModel):
    text: str


@app.post("/api/v1/voice/speak")
def trigger_speech(req: SpeechRequest):
    pipeline.voice.speak(req.text, priority=0)
    return {"ok": True, "spoken": req.text}


@app.get("/api/v1/logs/export")
def export_flight_logs(format: str = Query("csv", pattern="^(csv|jsonl)$")):
    if format == "jsonl":
        content = "\n".join([json.dumps(log) for log in flight_logs])
        media_type = "application/x-ndjson"
        filename = f"astroflow_flight_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    else:
        # CSV format
        lines = ["id,timestamp,level,state,confidence,message"]
        for l in flight_logs:
            lines.append(f"{l.get('id','')},{l.get('timestamp','')},{l.get('level','')},{l.get('state','')},{l.get('confidence',0.0):.2f},\"{l.get('message','')}\"")
        content = "\n".join(lines)
        media_type = "text/csv"
        filename = f"astroflow_flight_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.post("/api/v1/session/reset")
def reset_session():
    pipeline.stabilizer.fsm.reset()
    pipeline.stabilizer.current_state = "idle"
    pipeline.stabilizer.state_index = 0
    pipeline.stabilizer.cycles_completed = 0
    return {"ok": True, "state": "idle"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_engine.server:app", host="0.0.0.0", port=8080, reload=False)
