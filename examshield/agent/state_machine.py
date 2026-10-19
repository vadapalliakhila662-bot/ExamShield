"""
ExamShield Agentic Vision State Machine
Implements dynamic multi-step perception -> tool dispatch -> action routing.
Guarantees agentic criteria: OpenCV 5 outputs alter downstream tool calls and decisions.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from examshield.perception.opencv5_engine import OpenCV5PerceptionEngine, PerceptionTelemetry
from examshield.perception.adaptive_tools import apply_clahe_enhancement, extract_peripheral_roi
from examshield.agent.trace_logger import AgentTraceRecorder

class AccommodationProfile(BaseModel):
    student_id: str
    has_motor_tic: bool = False
    has_adhd: bool = False
    has_strabismus: bool = False
    notes: Optional[str] = None

class AgentDecisionResult(BaseModel):
    status: str # NORMAL, TRANSIENT_SUPPRESSED, QUALITY_DEGRADED_UNRELIABLE, SUSTAINED_ANOMALY, HUMAN_REVIEW_REQUIRED
    confidence: float
    routing_action: str
    reasoning: str
    tools_executed: List[str]
    flags_raised: List[str]
    requires_human_reviewer: bool = False

class AgenticVisionLoop:
    """
    Agentic loop orchestrating OpenCV 5 perception and dynamic tool execution.
    """
    def __init__(
        self,
        session_id: str,
        engine: OpenCV5PerceptionEngine,
        accommodation_profile: Optional[AccommodationProfile] = None
    ):
        self.session_id = session_id
        self.engine = engine
        self.accommodation = accommodation_profile or AccommodationProfile(student_id="default")
        self.trace_recorder = AgentTraceRecorder(session_id=session_id)
        self.anomaly_duration_seconds = 0.0

    def evaluate_frame_agentically(
        self,
        frame: np.ndarray,
        duration_anomaly_active: float = 0.0
    ) -> AgentDecisionResult:
        """
        Executes multi-step agentic perception loop:
          Step 1: Ingest frame & extract quality diagnostics using OpenCV 5.
          Step 2: If low-light, dispatch adaptive CLAHE enhancement tool and re-examine.
          Step 3: If quality is too blurred/degraded, mark flag as UNRELIABLE and route to human review.
          Step 4: If gaze deviation or head rotation detected:
                  - Dynamic tool dispatch: run peripheral ROI extraction & secondary object scan.
                  - Apply accommodation adjustments (e.g. strabismus/ADHD relaxes gaze threshold).
                  - Apply temporal gating: if duration < 120s (transient) and no contraband object, suppress alert.
          Step 5: Record trace showing how OpenCV output altered subsequent actions.
        """
        tools_executed = []
        flags = []
        self.anomaly_duration_seconds = duration_anomaly_active

        # Step 1: Initial Perception without CLAHE
        initial_telemetry = self.engine.process_frame(frame, enable_clahe_on_low_light=False)
        self.trace_recorder.record_step(
            stage="PERCEPTION",
            opencv_telemetry=initial_telemetry.model_dump(),
            action_taken="Initial OpenCV 5 frame extraction",
            rationale="Captured baseline luminance, blur score, and face/pose tracking."
        )

        active_frame = frame
        active_telemetry = initial_telemetry

        # Step 2: Agentic Tool Call - Dynamic Contrast Equalization
        if initial_telemetry.quality.is_low_light:
            tools_executed.append("cv2.createCLAHE (Dynamic Contrast Equalizer)")
            active_frame = apply_clahe_enhancement(frame)
            active_telemetry = self.engine.process_frame(active_frame, enable_clahe_on_low_light=False)
            active_telemetry.clahe_enhanced = True
            
            self.trace_recorder.record_step(
                stage="TOOL_DISPATCH",
                opencv_telemetry=active_telemetry.model_dump(),
                action_taken="Dispatched CLAHE adaptive enhancement",
                tool_dispatched="cv2.createCLAHE",
                rationale=f"Mean luminance ({initial_telemetry.quality.mean_luminance}) was below threshold (45.0). Enhanced image to prevent false face-loss alerts.",
                decision_changed=True,
                prior_verdict="PROBABLE_LIGHTING_ERROR",
                updated_verdict="RE-ANALYZED_WITH_ENHANCEMENT"
            )

        # Step 3: Optical Degradation & Blur Guard
        if active_telemetry.quality.is_blurred:
            # When image is severely blurred, we refuse to generate automated high-confidence infractions
            self.trace_recorder.record_step(
                stage="DECISION",
                opencv_telemetry=active_telemetry.model_dump(),
                action_taken="Marked as UNRELIABLE due to optical blur",
                rationale=f"Laplacian variance ({active_telemetry.quality.blur_score}) indicates motion blur or lens focus loss.",
                decision_changed=True,
                prior_verdict="ALERT_FLAG",
                updated_verdict="QUALITY_DEGRADED_UNRELIABLE"
            )
            return AgentDecisionResult(
                status="QUALITY_DEGRADED_UNRELIABLE",
                confidence=0.35,
                routing_action="ROUTE_TO_HUMAN_REVIEW_CONTEXT",
                reasoning="Optical blur prevented deterministic landmark extraction; suppressed automated alert.",
                tools_executed=tools_executed,
                flags_raised=["DEGRADED_OPTICAL_BLUR"],
                requires_human_reviewer=True
            )

        # Step 4: Gaze Deviation / Head Rotation Check
        has_pose_deviation = (
            abs(active_telemetry.head_pose["yaw"]) > 25.0 or
            abs(active_telemetry.head_pose["pitch"]) > 20.0
        )
        has_gaze_deviation = abs(active_telemetry.gaze_vector["horizontal"]) > 0.40

        # Adjust for accommodation profile
        if self.accommodation.has_strabismus or self.accommodation.has_adhd:
            # Strabismus/ADHD creates normal horizontal gaze shifts; suppress purely gaze-based flag
            has_gaze_deviation = False

        if has_pose_deviation or has_gaze_deviation or not active_telemetry.face_detected:
            # Dynamic Tool Dispatch: Extract peripheral ROI and scan for secondary contraband
            tools_executed.append("extract_peripheral_roi + cv2.Canny/Contour Scan")
            peripheral_roi = extract_peripheral_roi(active_frame)
            secondary_objects = self.engine.detect_secondary_objects(peripheral_roi)

            self.trace_recorder.record_step(
                stage="TOOL_DISPATCH",
                opencv_telemetry={"secondary_objects": secondary_objects},
                action_taken="Dispatched peripheral device scanner",
                tool_dispatched="extract_peripheral_roi",
                rationale="Gaze/pose deviation detected; verifying whether external contraband or secondary device is visible."
            )

            # Apply temporal rule (transient <= 120s vs sustained >= 600s)
            if not secondary_objects and duration_anomaly_active < 120.0:
                # Transient look-away (thinking, natural shift, pen dropped): suppress alert!
                self.trace_recorder.record_step(
                    stage="DECISION",
                    opencv_telemetry=active_telemetry.model_dump(),
                    action_taken="Suppressed alert (Transient Natural Interruption)",
                    rationale=f"Duration ({duration_anomaly_active:.1f}s) is under the 120s transient threshold with no secondary devices detected.",
                    decision_changed=True,
                    prior_verdict="MALPRACTICE_FLAG",
                    updated_verdict="TRANSIENT_SUPPRESSED"
                )
                return AgentDecisionResult(
                    status="TRANSIENT_SUPPRESSED",
                    confidence=0.20,
                    routing_action="LOG_EVENT_LOCALLY",
                    reasoning="Natural transient gaze shift under 120 seconds. Automated proctor alert suppressed.",
                    tools_executed=tools_executed,
                    flags_raised=[],
                    requires_human_reviewer=False
                )
            elif secondary_objects:
                flags.append("SECONDARY_DEVICE_DETECTED")
                self.trace_recorder.record_step(
                    stage="DECISION",
                    opencv_telemetry={"detected": secondary_objects},
                    action_taken="Flagged sustained infraction with secondary device",
                    rationale=f"Confirmed unauthorized item in periphery: {secondary_objects}",
                    decision_changed=True,
                    prior_verdict="TRANSIENT",
                    updated_verdict="SUSTAINED_ANOMALY"
                )
                return AgentDecisionResult(
                    status="SUSTAINED_ANOMALY",
                    confidence=0.92,
                    routing_action="ROUTE_TO_HUMAN_REVIEWER_DASHBOARD",
                    reasoning=f"Secondary device ({','.join(secondary_objects)}) confirmed concurrently with look-away.",
                    tools_executed=tools_executed,
                    flags_raised=flags,
                    requires_human_reviewer=True
                )
            else:
                flags.append("SUSTAINED_LOOKAWAY")
                return AgentDecisionResult(
                    status="SUSTAINED_ANOMALY",
                    confidence=0.78,
                    routing_action="ROUTE_TO_HUMAN_REVIEWER_DASHBOARD",
                    reasoning=f"Persistent lookaway sustained for {duration_anomaly_active:.1f} seconds (exceeds 120s limit).",
                    tools_executed=tools_executed,
                    flags_raised=flags,
                    requires_human_reviewer=True
                )

        # Standard compliant frame
        return AgentDecisionResult(
            status="NORMAL",
            confidence=0.98,
            routing_action="PASS",
            reasoning="Participant centered; posture, gaze vector, and illumination within nominal bounds.",
            tools_executed=tools_executed,
            flags_raised=[],
            requires_human_reviewer=False
        )
