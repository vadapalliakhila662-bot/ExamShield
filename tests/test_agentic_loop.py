"""
Automated Tests for Agentic Decision Loop and Trace Verification
"""

import pytest
import cv2
import numpy as np

from examshield.perception.opencv5_engine import OpenCV5PerceptionEngine
from examshield.agent.state_machine import AgenticVisionLoop, AccommodationProfile

def test_agentic_trace_captures_clahe_dispatch():
    """Verify that low lighting triggers adaptive CLAHE and records decision change."""
    engine = OpenCV5PerceptionEngine()
    loop = AgenticVisionLoop(session_id="test-trace-clahe", engine=engine)

    # Synthetic dim frame
    dim_frame = np.full((480, 640, 3), 20, dtype=np.uint8)
    cv2.circle(dim_frame, (320, 240), 90, (30, 30, 30), -1)

    decision = loop.evaluate_frame_agentically(dim_frame, duration_anomaly_active=5.0)

    # Check trace
    trace = loop.trace_recorder.steps
    assert len(trace) >= 2, "Agentic trace must record multi-step tool calls"
    
    # Assert at least one step was a dynamic tool dispatch
    tool_steps = [s for s in trace if s.stage == "TOOL_DISPATCH"]
    assert len(tool_steps) > 0
    assert any("CLAHE" in s.tool_dispatched for s in tool_steps if s.tool_dispatched)

def test_agentic_blur_guard_routes_to_human_review():
    """Verify that severe blur flags the frame as UNRELIABLE rather than generating false penalty."""
    engine = OpenCV5PerceptionEngine()
    loop = AgenticVisionLoop(session_id="test-trace-blur", engine=engine)

    # Uniform flat frame yields 0 Laplacian variance (maximum blur)
    flat_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    decision = loop.evaluate_frame_agentically(flat_frame, duration_anomaly_active=10.0)

    assert decision.status == "QUALITY_DEGRADED_UNRELIABLE"
    assert decision.requires_human_reviewer is True
    assert "ROUTE_TO_HUMAN_REVIEW_CONTEXT" in decision.routing_action

def test_transient_lookaway_suppression():
    """Verify that looking away under 120s with no secondary device is suppressed."""
    engine = OpenCV5PerceptionEngine()
    loop = AgenticVisionLoop(session_id="test-transient", engine=engine)

    # Frame with head contour offset to side and sufficient texture for sharpness
    np.random.seed(42)
    frame = np.random.randint(110, 140, (480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (180, 240), 90, (220, 220, 220), -1)

    decision = loop.evaluate_frame_agentically(frame, duration_anomaly_active=15.0)

    assert decision.status == "TRANSIENT_SUPPRESSED"
    assert decision.requires_human_reviewer is False
    assert decision.confidence <= 0.30

def test_accommodation_profile_suppresses_gaze_flag():
    """Verify that ADHD / Strabismus accommodation adjusts the perception baseline."""
    engine = OpenCV5PerceptionEngine()
    accommodation = AccommodationProfile(
        student_id="student-adhd-01",
        has_adhd=True,
        has_strabismus=True
    )
    loop = AgenticVisionLoop(session_id="test-accommodation", engine=engine, accommodation_profile=accommodation)

    np.random.seed(42)
    frame = np.random.randint(110, 140, (480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 240), 90, (220, 220, 220), -1)

    decision = loop.evaluate_frame_agentically(frame, duration_anomaly_active=5.0)
    assert decision.status in ("NORMAL", "TRANSIENT_SUPPRESSED")
