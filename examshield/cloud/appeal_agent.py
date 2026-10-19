"""
ExamShield Bedrock Appeal & Consistency Agent
Evaluates student claims against physical OpenCV 5 telemetry using Claude 3.5 Sonnet.
Enforces that automated flags NEVER produce automated penalties without human review.
"""

import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from examshield.config import settings

logger = logging.getLogger("ExamShield-AppealAgent")

class AppealEvaluationResult(BaseModel):
    consistency_verdict: str # CONSISTENT, INCONSISTENT, AMBIGUOUS
    consistency_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    recommended_action: str # DISMISS_FLAG, HUMAN_REVIEW_MANDATORY, RE_EXAMINE_EVIDENCE
    model_used: str

class BedrockAppealAgent:
    def __init__(self, aws_manager=None):
        self.aws_manager = aws_manager
        self.model_id = settings.BEDROCK_MODEL_ID

    def evaluate_appeal(
        self,
        student_claim: str,
        opencv_telemetry_history: list,
        environmental_context: Dict[str, Any]
    ) -> AppealEvaluationResult:
        """
        Submits physical OpenCV 5 metrics and student explanation to Bedrock Claude 3.5 Sonnet.
        """
        prompt = f"""
You are an impartial Fairness and Evidence Consistency Auditor for academic assessments.
Your role is to cross-examine student-provided explanations against physical sensor telemetry from OpenCV 5.
Student context claims are treated as procedural context, not proof.
You must NOT produce a punitive sanction; you only assess consistency and recommend human review.

PHYSICAL OPENCV 5 TELEMETRY:
{json.dumps(opencv_telemetry_history, indent=2)}

ENVIRONMENTAL CONDITIONS:
{json.dumps(environmental_context, indent=2)}

STUDENT CLAIM / EXPLANATION:
"{student_claim}"

Evaluate whether the head pose angles (yaw/pitch), gaze direction, illumination, and detected objects
are physically consistent with the student's claimed narrative (e.g. dropped pen, lighting flicker, family member entered).

Respond ONLY with a valid JSON object matching this schema:
{{
  "consistency_verdict": "CONSISTENT" | "INCONSISTENT" | "AMBIGUOUS",
  "consistency_score": <float between 0.0 and 1.0>,
  "reasoning": "<detailed physical alignment explanation>",
  "recommended_action": "DISMISS_FLAG" | "HUMAN_REVIEW_MANDATORY"
}}
"""
        # Try live AWS Bedrock invocation if connected
        if self.aws_manager and self.aws_manager.is_connected and self.aws_manager.bedrock:
            try:
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                })
                response = self.aws_manager.bedrock.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=body
                )
                response_body = json.loads(response["body"].read().decode("utf-8"))
                text = response_body["content"][0]["text"].strip()
                parsed = json.loads(text)
                return AppealEvaluationResult(
                    consistency_verdict=parsed["consistency_verdict"],
                    consistency_score=float(parsed["consistency_score"]),
                    reasoning=parsed["reasoning"],
                    recommended_action=parsed["recommended_action"],
                    model_used=self.model_id
                )
            except Exception as e:
                logger.warning(f"Live Bedrock call failed ({e}). Executing local deterministic consistency evaluation.")

        # Resilient heuristic audit fallback
        return self._local_consistency_eval(student_claim, opencv_telemetry_history)

    def _local_consistency_eval(self, claim: str, history: list) -> AppealEvaluationResult:
        claim_lower = claim.lower()
        # Analyze pitch and yaw in telemetry
        pitches = []
        yaws = []
        secondary_devices = []

        for item in history:
            hp = item.get("head_pose", {})
            if "pitch" in hp:
                pitches.append(hp["pitch"])
            if "yaw" in hp:
                yaws.append(hp["yaw"])
            devs = item.get("secondary_devices_detected", [])
            secondary_devices.extend(devs)

        mean_pitch = sum(pitches) / len(pitches) if pitches else 0.0
        mean_yaw = sum(yaws) / len(yaws) if yaws else 0.0

        if "drop" in claim_lower or "floor" in claim_lower or "pen" in claim_lower:
            # Dropped pen requires downward pitch (< -10 degrees)
            if mean_pitch < -10.0 and not secondary_devices:
                return AppealEvaluationResult(
                    consistency_verdict="CONSISTENT",
                    consistency_score=0.88,
                    reasoning=f"Downward pitch deviation ({mean_pitch:.1f} deg) aligns with physical motion of retrieving an item from floor. Zero secondary devices observed.",
                    recommended_action="DISMISS_FLAG",
                    model_used="ExamShield-LocalConsistencyEngine"
                )
            else:
                return AppealEvaluationResult(
                    consistency_verdict="INCONSISTENT",
                    consistency_score=0.25,
                    reasoning=f"Student claimed a dropped object, but telemetry shows lateral yaw ({mean_yaw:.1f} deg) without downward pitch deviation ({mean_pitch:.1f} deg).",
                    recommended_action="HUMAN_REVIEW_MANDATORY",
                    model_used="ExamShield-LocalConsistencyEngine"
                )

        return AppealEvaluationResult(
            consistency_verdict="AMBIGUOUS",
            consistency_score=0.50,
            reasoning="Telemetry requires manual human proctor review to cross-validate non-standard claim.",
            recommended_action="HUMAN_REVIEW_MANDATORY",
            model_used="ExamShield-LocalConsistencyEngine"
        )
