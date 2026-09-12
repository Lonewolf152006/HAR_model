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
from ai_engine.pipeline import AstroFlowPipeline, PhysicalCausalLogic
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
        mock_box = {"rect": (100, 100, 200, 200)}

        # open_box
        can, _ = fsm.can_transition("open_box", main_box=mock_box, dist_main=0.1)
        self.assertTrue(can)
        fsm.apply("open_box")

        # pick_red
        can, _ = fsm.can_transition("pick_red", main_box=mock_box, red_box=mock_box, dist_red=0.1)
        self.assertTrue(can)
        fsm.apply("pick_red")

        # place_red_out
        can, _ = fsm.can_transition("place_red_out")
        self.assertTrue(can)
        fsm.apply("place_red_out")

        # pick_blue
        can, _ = fsm.can_transition("pick_blue", main_box=mock_box, blue_box=mock_box, dist_blue=0.1)
        self.assertTrue(can)
        fsm.apply("pick_blue")

        # place_blue_in
        can, _ = fsm.can_transition("place_blue_in")
        self.assertTrue(can)
        fsm.apply("place_blue_in")

        # close_box
        can, _ = fsm.can_transition("close_box", main_box=mock_box, dist_main=0.1)
        self.assertTrue(can)
        fsm.apply("close_box")

    def test_03_causal_logic_step_skip_rejection(self):
        """Verify out-of-order action is blocked with specific reason."""
        fsm = PhysicalCausalLogic()
        # Skipping open_box and trying to pick_red immediately
        can, reason = fsm.can_transition("pick_red")
        self.assertFalse(can)
        self.assertIn("open", reason.lower())

        # Open box first
        fsm.apply("open_box")
        # Skipping red cube and trying to pick blue cube immediately
        can, reason = fsm.can_transition("pick_blue")
        self.assertFalse(can)
        self.assertIn("red", reason.lower())

    def test_04_pipeline_5_gates(self):
        """Verify 5 gates evaluation structure in end-to-end pipeline."""
        pipe = AstroFlowPipeline()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        telem = pipe.process_frame(frame)

        self.assertIn("gates", telem)
        self.assertEqual(len(telem["gates"]), 5)
        self.assertIn("confidence", telem["gates"])
        self.assertIn("stability", telem["gates"])
        self.assertIn("cooldown", telem["gates"])
        self.assertIn("motion", telem["gates"])
        self.assertIn("causal_logic", telem["gates"])

    def test_05_camera_enumeration(self):
        """Verify camera enumeration finds devices or synthetic fallback."""
        cameras = get_available_cameras()
        self.assertTrue(len(cameras) >= 1)
        cam = cameras[0]
        self.assertIn("id", cam)
        self.assertIn("name", cam)
        self.assertIn("status", cam)


if __name__ == "__main__":
    unittest.main()
