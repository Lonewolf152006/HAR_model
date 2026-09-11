"""
tests/test_backend.py - Automated Verification Suite for AstroFlow AI Edge Hub
"""

import os
import sys
import unittest
import torch
import numpy as np

# Ensure root directory is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine.model_def import TARModel, FEATURE_DIM, NUM_CLASSES
from ai_engine.pipeline import PhysicalCausalLogic, DecisionStabilizer
from ai_engine.camera import get_available_cameras


class TestAstroFlowAI(unittest.TestCase):
    def test_01_tar_model_forward(self):
        """Test TARModel forward pass with dummy 332-D temporal sequence."""
        model = TARModel(input_size=FEATURE_DIM, num_classes=NUM_CLASSES)
        weight_path = "ai_engine/models/best_tar_model.pth"
        if os.path.exists(weight_path):
            sd = torch.load(weight_path, map_location="cpu", weights_only=True)
            model.load_state_dict(sd)
        model.eval()

        dummy_seq = torch.zeros(1, 48, FEATURE_DIM, dtype=torch.float32)
        with torch.no_grad():
            out = model(dummy_seq)
        self.assertEqual(out.shape, (1, NUM_CLASSES))

    def test_02_causal_logic_nominal_sequence(self):
        """Verify strict sequential SOP transitions succeed."""
        fsm = PhysicalCausalLogic()
        nominal_steps = [
            "idle",
            "open_box",
            "pick_red",
            "place_red_out",
            "pick_blue",
            "place_blue_in",
            "close_box",
        ]
        for step in nominal_steps:
            can, reason = fsm.can_transition(step)
            self.assertTrue(can, f"Expected step {step} to be permitted, got rejection: {reason}")
            fsm.apply(step)

    def test_03_causal_logic_step_skip_rejection(self):
        """Verify out-of-order action is blocked with specific reason."""
        fsm = PhysicalCausalLogic()
        # Skipping open_box and trying to pick_red immediately
        can, reason = fsm.can_transition("pick_red")
        self.assertFalse(can)
        self.assertIn("closed", reason.lower())

        # Open box first
        fsm.apply("open_box")
        # Skipping red cube and trying to pick blue cube immediately
        can, reason = fsm.can_transition("pick_blue")
        self.assertFalse(can)
        self.assertIn("red cube must be placed", reason.lower())

    def test_04_decision_stabilizer_5_gates(self):
        """Verify 5 gates evaluation structure."""
        stab = DecisionStabilizer(confidence=0.52, stability_window=3, cooldown=0.1, motion_floor=0.10)
        raw_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
        raw_probs[0] = 0.85  # idle
        dec = stab.update(raw_probs, motion_energy=0.20)
        
        self.assertIn("gates", dec)
        self.assertEqual(len(dec["gates"]), 5)
        self.assertIn("confidence", dec["gates"])
        self.assertIn("stability", dec["gates"])
        self.assertIn("cooldown", dec["gates"])
        self.assertIn("motion", dec["gates"])
        self.assertIn("causal_logic", dec["gates"])
        self.assertTrue(dec["gates"]["confidence"]["passed"])

    def test_05_camera_enumeration(self):
        """Verify camera enumeration finds devices or synthetic fallback."""
        cameras = get_available_cameras()
        self.assertTrue(len(cameras) >= 1)
        # Verify device schema
        cam = cameras[0]
        self.assertIn("id", cam)
        self.assertIn("name", cam)
        self.assertIn("status", cam)


if __name__ == "__main__":
    unittest.main()
