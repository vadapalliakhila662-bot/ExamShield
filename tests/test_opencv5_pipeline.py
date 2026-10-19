"""
Automated Tests for OpenCV 5 Pipeline Primitives
"""

import pytest
import cv2
import numpy as np

from examshield.perception.quality_diagnostics import evaluate_frame_quality
from examshield.perception.adaptive_tools import apply_clahe_enhancement, extract_peripheral_roi
from examshield.perception.opencv5_engine import OpenCV5PerceptionEngine

def test_opencv5_version():
    """Verify that the pinned OpenCV version is 5.x."""
    assert cv2.__version__.startswith("5."), f"Expected OpenCV 5, got {cv2.__version__}"

def test_opencv5_required_modules():
    """Verify all required OpenCV 5 perception functions are available."""
    assert hasattr(cv2, "solvePnP"), "cv2.solvePnP missing"
    assert hasattr(cv2, "createCLAHE"), "cv2.createCLAHE missing"
    assert hasattr(cv2, "Laplacian"), "cv2.Laplacian missing"
    assert hasattr(cv2, "mean"), "cv2.mean missing"
    assert hasattr(cv2, "cvtColor"), "cv2.cvtColor missing"
    assert hasattr(cv2, "RQDecomp3x3"), "cv2.RQDecomp3x3 missing"
    assert hasattr(cv2, "FaceDetectorYN_create"), "cv2.FaceDetectorYN_create missing"

def test_quality_diagnostics_normal_frame():
    """Test diagnostics on normal illumination frame."""
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    diag = evaluate_frame_quality(frame, min_luminance=45.0, blur_threshold=100.0)
    assert not diag.is_low_light
    assert diag.mean_luminance >= 45.0

def test_quality_diagnostics_low_light_frame():
    """Test diagnostics on dim illumination frame."""
    frame = np.full((480, 640, 3), 20, dtype=np.uint8)
    diag = evaluate_frame_quality(frame, min_luminance=45.0, blur_threshold=100.0)
    assert diag.is_low_light
    assert diag.is_degraded

def test_clahe_enhancement_increases_luminance():
    """Test CLAHE dynamic tool equalization."""
    frame = np.full((480, 640, 3), 25, dtype=np.uint8)
    # Add gradient
    frame[:, :, 0] = np.linspace(10, 40, 640, dtype=np.uint8)
    enhanced = apply_clahe_enhancement(frame)
    assert enhanced is not None
    assert enhanced.shape == frame.shape
    # CLAHE redistributes and improves local contrast
    assert enhanced.mean() >= frame.mean()

def test_solve_pnp_euler_angles():
    """Test 3D head pose estimation using OpenCV 5 solvePnP."""
    engine = OpenCV5PerceptionEngine()
    image_points = np.array([
        (320.0, 240.0), # nose
        (320.0, 350.0), # chin
        (240.0, 180.0), # left eye
        (400.0, 180.0), # right eye
        (260.0, 300.0), # left mouth
        (380.0, 300.0)  # right mouth
    ], dtype=np.float64)

    angles = engine.estimate_head_pose(image_points, (480, 640))
    assert "yaw" in angles
    assert "pitch" in angles
    assert "roll" in angles
    assert isinstance(angles["yaw"], float)
