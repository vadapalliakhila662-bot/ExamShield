"""
ExamShield Perception Module (OpenCV 5 Core Vision Engine)
"""
from examshield.perception.quality_diagnostics import evaluate_frame_quality
from examshield.perception.adaptive_tools import apply_clahe_enhancement, extract_peripheral_roi
from examshield.perception.opencv5_engine import OpenCV5PerceptionEngine

__all__ = [
    "evaluate_frame_quality",
    "apply_clahe_enhancement",
    "extract_peripheral_roi",
    "OpenCV5PerceptionEngine",
]
