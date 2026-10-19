"""
ExamShield Agent Trace Logger
Records structured evidence of the agentic perception-decision-action loop.
Provides the trace required by the Agentic Vision Award rubric:
"A trace or demo showing an OpenCV 5 output changing a later decision or tool call."
"""

import time
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class AgentTraceStep(BaseModel):
    step_id: int
    timestamp: float = Field(default_factory=time.time)
    stage: str # PERCEPTION, TOOL_DISPATCH, DECISION, ACTION
    opencv_telemetry: Dict[str, Any]
    action_taken: str
    tool_dispatched: Optional[str] = None
    rationale: str
    decision_changed: bool = False
    prior_verdict: Optional[str] = None
    updated_verdict: Optional[str] = None

class AgentTraceRecorder:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.steps: List[AgentTraceStep] = []
        self._step_counter = 0

    def record_step(
        self,
        stage: str,
        opencv_telemetry: Dict[str, Any],
        action_taken: str,
        tool_dispatched: Optional[str] = None,
        rationale: str = "",
        decision_changed: bool = False,
        prior_verdict: Optional[str] = None,
        updated_verdict: Optional[str] = None
    ) -> AgentTraceStep:
        self._step_counter += 1
        step = AgentTraceStep(
            step_id=self._step_counter,
            timestamp=time.time(),
            stage=stage,
            opencv_telemetry=opencv_telemetry,
            action_taken=action_taken,
            tool_dispatched=tool_dispatched,
            rationale=rationale,
            decision_changed=decision_changed,
            prior_verdict=prior_verdict,
            updated_verdict=updated_verdict
        )
        self.steps.append(step)
        return step

    def get_trace_log(self) -> List[Dict[str, Any]]:
        return [step.model_dump() for step in self.steps]

    def export_json(self) -> str:
        return json.dumps({
            "session_id": self.session_id,
            "total_steps": len(self.steps),
            "trace": self.get_trace_log()
        }, indent=2)
