"""
ai_engine/tts.py - Offline Voice Copilot (Personal Assistant TTS)

Features:
- 100% offline text-to-speech using local system speech engine (pyttsx3).
- Non-blocking priority audio queue so inference loop never stalls.
- Debounced voice assistance explaining exact missed steps and protocol errors.
- Emits voice events for browser-side speech fallback.
"""

import sys
import time
import queue
import threading


# Friendly human descriptions for SOP steps
STEP_DESCRIPTIONS = {
    "idle": "Standby baseline posture",
    "open_box": "Opening the main experiment container lid",
    "pick_red": "Picking the red sample cube from the container",
    "place_red_out": "Depositing the red sample cube onto the exterior bracket",
    "pick_blue": "Retrieving the blue sample cube from the mount",
    "place_blue_in": "Placing the blue sample cube inside the container",
    "close_box": "Closing and latching the experiment container lid",
}

STEP_NUMBERS = {
    "idle": 0,
    "open_box": 1,
    "pick_red": 2,
    "place_red_out": 3,
    "pick_blue": 4,
    "place_blue_in": 5,
    "close_box": 6,
}


class VoiceCopilot:
    """Thread-safe background speech synthesizer for astronaut guidance."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.speech_queue = queue.Queue(maxsize=10)
        self.running = False
        self.thread = None
        self.last_spoken_text = ""
        self.last_spoken_time = 0.0
        self.lock = threading.Lock()
        self.on_speech_event_callback = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _worker_loop(self):
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)
            voices = engine.getProperty("voices")
            # Select clear female/natural voice if available
            for v in voices:
                if "zira" in v.name.lower() or "david" in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break
        except Exception as e:
            print(f"[TTS] pyttsx3 init warning: {e}. Browser Web Speech will handle audio.")

        while self.running:
            try:
                priority, text = self.speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.enabled:
                self.speech_queue.task_done()
                continue

            # Notify attached callbacks (e.g. WebSocket streamer for browser speech)
            if self.on_speech_event_callback:
                try:
                    self.on_speech_event_callback(text)
                except Exception:
                    pass

            if engine:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"[TTS] Speech synthesis error: {e}")

            self.speech_queue.task_done()
            time.sleep(0.1)

    def speak(self, text, priority=1, min_interval=3.5):
        """Enqueue speech prompt with anti-spam debouncing."""
        now = time.time()
        with self.lock:
            # Suppress identical phrases within debounce window
            if text == self.last_spoken_text and (now - self.last_spoken_time) < min_interval:
                return
            if (now - self.last_spoken_time) < 1.5 and priority > 0:
                # Give breathing room between different low-priority speech
                return
            self.last_spoken_text = text
            self.last_spoken_time = now

        try:
            self.speech_queue.put_nowait((priority, text))
        except queue.Full:
            pass

    def announce_step_confirmed(self, state):
        step_num = STEP_NUMBERS.get(state, 0)
        desc = STEP_DESCRIPTIONS.get(state, state)
        if state != "idle":
            msg = f"Step {step_num} verified: {desc}."
            self.speak(msg, priority=2, min_interval=4.0)

    def announce_step_missed(self, expected_state, attempted_state):
        exp_num = STEP_NUMBERS.get(expected_state, 0)
        exp_desc = STEP_DESCRIPTIONS.get(expected_state, expected_state)
        msg = f"Astronaut, hold on. You missed step {exp_num}. Please complete {exp_desc} before proceeding."
        self.speak(msg, priority=0, min_interval=4.0)

    def announce_causal_violation(self, reason):
        msg = f"Procedure alert: {reason}. Check experiment protocol."
        self.speak(msg, priority=0, min_interval=3.5)

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
