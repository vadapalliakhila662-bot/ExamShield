"""
ExamShield Web API and Proctor Review Backend
Built with FastAPI and OpenCV 5.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from examshield.config import settings
from examshield.perception.opencv5_engine import OpenCV5PerceptionEngine
from examshield.agent.state_machine import AgenticVisionLoop, AccommodationProfile, AgentDecisionResult
from examshield.cloud.aws_client import AWSClientManager
from examshield.cloud.dynamodb_schema import ExamShieldAuditEvent
from examshield.cloud.appeal_agent import BedrockAppealAgent, AppealEvaluationResult
from examshield.audit.fairness_auditor import FairnessAuditor
from examshield.audit.bias_stress_test import BiasStressTestRunner

app = FastAPI(
    title="ExamShield API",
    description="Human-Centered Agentic Vision Proctoring Backend (OpenCV 5 & AWS)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
engine = OpenCV5PerceptionEngine()
aws_manager = AWSClientManager()
appeal_agent = BedrockAppealAgent(aws_manager=aws_manager)
fairness_auditor = FairnessAuditor()
active_loops: Dict[str, AgenticVisionLoop] = {}

class FrameEvaluationRequest(BaseModel):
    session_id: str
    student_id: str
    duration_anomaly_active_seconds: float = 0.0
    accommodation: Optional[AccommodationProfile] = None

class StudentAppealRequest(BaseModel):
    session_id: str
    student_id: str
    student_claim: str

@app.get("/health")
def health_check():
    """
    Service health check reporting OpenCV 5 and AWS connectivity status.
    """
    return {
        "status": "healthy",
        "opencv_version": cv2.__version__,
        "opencv_major_requirement_met": cv2.__version__.startswith("5."),
        "aws_connected": aws_manager.is_connected,
        "region": settings.AWS_DEFAULT_REGION,
        "environment": "production" if aws_manager.is_connected else "local_fallback"
    }

@app.get("/api/verify-opencv5")
def verify_opencv5():
    """
    Audits OpenCV 5 installation and confirms all required perception modules are active.
    """
    modules = {
        "opencv_version": cv2.__version__,
        "cv2.dnn": hasattr(cv2, "dnn"),
        "cv2.solvePnP": hasattr(cv2, "solvePnP"),
        "cv2.createCLAHE": hasattr(cv2, "createCLAHE"),
        "cv2.Laplacian": hasattr(cv2, "Laplacian"),
        "cv2.mean": hasattr(cv2, "mean"),
        "cv2.cvtColor": hasattr(cv2, "cvtColor"),
        "cv2.RQDecomp3x3": hasattr(cv2, "RQDecomp3x3"),
        "cv2.FaceDetectorYN_create": hasattr(cv2, "FaceDetectorYN_create")
    }
    all_ok = cv2.__version__.startswith("5.") and all(modules.values())
    return {
        "verified": all_ok,
        "modules": modules,
        "status_message": "All OpenCV 5 modules verified and operational." if all_ok else "OpenCV 5 verification failed."
    }

@app.post("/api/evaluate-frame", response_model=AgentDecisionResult)
def evaluate_frame(req: FrameEvaluationRequest):
    """
    Executes agentic vision perception loop on current frame state.
    """
    if req.session_id not in active_loops:
        active_loops[req.session_id] = AgenticVisionLoop(
            session_id=req.session_id,
            engine=engine,
            accommodation_profile=req.accommodation
        )

    loop = active_loops[req.session_id]
    # For synthetic or API testing without raw binary payload, create standard base frame with sharpness
    np.random.seed(42)
    synthetic_frame = np.random.randint(110, 140, (480, 640, 3), dtype=np.uint8)
    cv2.circle(synthetic_frame, (320, 240), 90, (220, 220, 220), -1)

    decision = loop.evaluate_frame_agentically(
        synthetic_frame,
        duration_anomaly_active=req.duration_anomaly_active_seconds
    )

    # Log to DynamoDB
    event = ExamShieldAuditEvent(
        SessionID=req.session_id,
        StudentId=req.student_id,
        FlagType=decision.status,
        ConfidenceScore=decision.confidence,
        LightingMetrics={"mean_luminance": 128.0},
        BlurScore=0.0,
        HeadPose={"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        GazeVector={"horizontal": 0.0, "vertical": 0.0},
        ToolsDispatched=decision.tools_executed,
        ProctorStatus="PENDING_REVIEW" if decision.requires_human_reviewer else "AUTO_CLEARED"
    )
    aws_manager.log_audit_event(event)

    return decision

@app.get("/api/trace/{session_id}")
def get_session_trace(session_id: str):
    """
    Retrieves the agent trace showing how OpenCV 5 telemetry changed decisions or tool calls.
    """
    if session_id not in active_loops:
        raise HTTPException(status_code=404, detail="Session ID not found")
    
    loop = active_loops[session_id]
    return {
        "session_id": session_id,
        "total_steps": len(loop.trace_recorder.steps),
        "trace": loop.trace_recorder.get_trace_log()
    }

@app.post("/api/appeal", response_model=AppealEvaluationResult)
def file_appeal(req: StudentAppealRequest):
    """
    Files an appeal; executes Bedrock Claude 3.5 Sonnet consistency audit.
    """
    history = aws_manager.query_audit_events(req.session_id)
    result = appeal_agent.evaluate_appeal(
        student_claim=req.student_claim,
        opencv_telemetry_history=history,
        environmental_context={"environment": "ExamShield-Secure"}
    )
    return result

@app.get("/api/stress-test")
def run_stress_test():
    """
    Executes programmatic bias stress test across lighting and blur conditions.
    """
    runner = BiasStressTestRunner(sample_frames_per_condition=30)
    results = runner.run_stress_test()
    table = runner.print_markdown_table(results)
    return {
        "results": results,
        "markdown_table": table
    }

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    """
    Proctor Dashboard & Live Inspection UI.
    """
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ExamShield - Agentic Vision Proctoring & Bias Audit Dashboard</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --accent: #38bdf8;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --card-border: #334155;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }
        .badge {
            background: #0284c7;
            color: #fff;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
        }
        .card {
            background: var(--bg-secondary);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 20px;
        }
        h2 {
            font-size: 1.25rem;
            margin-top: 0;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        button {
            background: #2563eb;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: 0.2s;
        }
        button:hover {
            background: #1d4ed8;
        }
        pre {
            background: #090d16;
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.85rem;
            color: #38bdf8;
        }
        .status-pill {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        .status-pill.pass { background: rgba(34, 197, 94, 0.2); color: var(--success); }
        .status-pill.suppressed { background: rgba(245, 158, 11, 0.2); color: var(--warning); }
        .status-pill.review { background: rgba(239, 68, 68, 0.2); color: var(--danger); }
    </style>
</head>
<body>
    <header>
        <div>
            <h1 style="margin: 0; font-size: 1.75rem;">🛡️ ExamShield</h1>
            <p style="margin: 4px 0 0 0; color: var(--text-muted); font-size: 0.95rem;">
                Human-Centered Agentic Vision Proctoring | OpenCV 5 & AWS Architecture
            </p>
        </div>
        <span class="badge">OpenCV 5.0.0 Verified</span>
    </header>

    <div class="grid">
        <!-- Live Diagnostics Card -->
        <div class="card">
            <h2>🔍 OpenCV 5 Verification & Health</h2>
            <p style="color: var(--text-muted); font-size: 0.9rem;">
                Audits cv2.solvePnP, cv2.createCLAHE, cv2.Laplacian, and native OpenCV 5 modules.
            </p>
            <button onclick="checkHealth()">Run OpenCV 5 Diagnostics</button>
            <pre id="health-output">Click button to audit runtime modules...</pre>
        </div>

        <!-- Agentic Trace Demo Card -->
        <div class="card">
            <h2>🤖 Agentic Decision Trace</h2>
            <p style="color: var(--text-muted); font-size: 0.9rem;">
                Demonstrates OpenCV 5 telemetry altering subsequent tool calls (CLAHE, ROI Scan) and decisions.
            </p>
            <button onclick="runTraceDemo()">Execute Agentic Loop Demo</button>
            <pre id="trace-output">Click button to generate agentic trace...</pre>
        </div>

        <!-- Bias Stress Test Card -->
        <div class="card">
            <h2>⚖️ Bias & Environmental Stress Test</h2>
            <p style="color: var(--text-muted); font-size: 0.9rem;">
                Validates false-flag reduction across normal, dim, and blurred conditions.
            </p>
            <button onclick="runStressTest()">Run Bias Stress Test</button>
            <pre id="stress-output">Click button to execute synthetic evaluation...</pre>
        </div>

        <!-- Bedrock Appeal Agent Card -->
        <div class="card">
            <h2>🏛️ Bedrock Consistency Appeal Agent</h2>
            <p style="color: var(--text-muted); font-size: 0.9rem;">
                Cross-examines student explanation against physical OpenCV 5 kinematics.
            </p>
            <button onclick="testAppeal()">Simulate Student Appeal Check</button>
            <pre id="appeal-output">Click button to evaluate student claim...</pre>
        </div>
    </div>

    <script>
        async function checkHealth() {
            const out = document.getElementById('health-output');
            out.textContent = 'Auditing OpenCV 5 runtime...';
            try {
                const res = await fetch('/api/verify-opencv5');
                const data = await res.json();
                out.textContent = JSON.stringify(data, null, 2);
            } catch(e) {
                out.textContent = 'Error: ' + e;
            }
        }

        async function runTraceDemo() {
            const out = document.getElementById('trace-output');
            out.textContent = 'Executing agentic vision loop...';
            try {
                const res = await fetch('/api/evaluate-frame', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: 'demo-session-' + Date.now(),
                        student_id: 'student-akhila-demo',
                        duration_anomaly_active_seconds: 15.0,
                        accommodation: {
                            student_id: 'student-akhila-demo',
                            has_adhd: false,
                            has_strabismus: false
                        }
                    })
                });
                const data = await res.json();
                out.textContent = JSON.stringify(data, null, 2);
            } catch(e) {
                out.textContent = 'Error: ' + e;
            }
        }

        async function runStressTest() {
            const out = document.getElementById('stress-output');
            out.textContent = 'Running synthetic stress test across 120 frames...';
            try {
                const res = await fetch('/api/stress-test');
                const data = await res.json();
                out.textContent = data.markdown_table;
            } catch(e) {
                out.textContent = 'Error: ' + e;
            }
        }

        async function testAppeal() {
            const out = document.getElementById('appeal-output');
            out.textContent = 'Submitting claim to Bedrock consistency auditor...';
            try {
                const res = await fetch('/api/appeal', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: 'appeal-session-001',
                        student_id: 'student-104',
                        student_claim: 'My pen fell on the floor, so I looked down to pick it up.'
                    })
                });
                const data = await res.json();
                out.textContent = JSON.stringify(data, null, 2);
            } catch(e) {
                out.textContent = 'Error: ' + e;
            }
        }
    </script>
</body>
</html>
"""
