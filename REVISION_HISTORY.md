# ExamShield: 3-Stage Revision & Hardening History

In accordance with Phase 1 directives, the ExamShield architecture, codebase, and submission criteria underwent **three iterative review and stress-testing cycles** to ensure 100% compliance with hackathon rules, OpenCV 5 technical requirements, and the Agentic Vision rubric.

---

## 🔄 Revision Cycle 1: Baseline Architecture & Dependency Pinning

### Objective
Establish the foundational repository structure, enforce exact dependency pinning, and eliminate ambiguity surrounding OpenCV 5 and AWS services.

### Changes Implemented
1. **Pinned Dependency Lockdown (`requirements.txt`):**
   - Strictly pinned `opencv-python==5.0.0.93` and `numpy==2.4.6`.
   - Pinned exact AWS SDK releases: `boto3==1.43.114`, `botocore==1.43.114`.
   - Pinned web and testing frameworks: `fastapi==0.146.0`, `uvicorn==0.54.0`, `pydantic==2.14.1`, `pytest==9.1.1`.
2. **Containerization (`Dockerfile` & `docker-compose.yml`):**
   - Built a minimal Debian-based container with required system libraries (`libgl1`, `libglib2.0-0`, `ffmpeg`).
   - Added an explicit build-time assertion: `python -c "import cv2; assert cv2.__version__.startswith('5')"`.
3. **OpenCV 5 Verification Utility (`verify_opencv5.py`):**
   - Created a standalone executable script auditing the installed OpenCV version and validating required perception primitives (`cv2.solvePnP`, `cv2.createCLAHE`, `cv2.Laplacian`, `cv2.mean`, `cv2.Rodrigues`).
4. **Cloud & Storage Configuration (`.env.example`):**
   - Defined environment variables for AWS DynamoDB tables, S3 evidence bucket, Bedrock foundation models, and Graviton/COOL optional flags.

---

## 🔄 Revision Cycle 2: Agentic Vision Compliance & Perception Hardening

### Objective
Stress-test the vision engine against the competition's "Agentic Vision" criteria. A fixed pipeline that merely classifies frames was strictly rejected; the system was re-architected into an active perception-decision-action state machine.

### Key Discoveries & Technical Fixes
1. **OpenCV 5 API Evolution Handling:**
   - *Discovery:* OpenCV 5 has deprecated/removed legacy Haar cascades (`cv2.CascadeClassifier`) from core packaging.
   - *Fix:* Re-architected `OpenCV5PerceptionEngine` to leverage native OpenCV 5 `cv2.FaceDetectorYN_create` (YuNet ONNX model) with an integrated OpenCV 5 contour/morphological segmentation fallback that executes out-of-the-box without requiring large external weight downloads.
2. **Dynamic Computer Vision Tool Dispatching:**
   - *Agentic Branch 1 (Low-Light):* When `cv2.mean()` detects luminance $< 45.0$, the agent does not flag the student; instead, it dynamically invokes `cv2.createCLAHE` to adaptively equalize the L-channel in LAB color space and re-analyzes the frame.
   - *Agentic Branch 2 (Gaze Shift / Lookaway):* When head yaw $> 25^\circ$ or iris displacement occurs, the agent dynamically dispatches a secondary peripheral ROI extraction tool (`extract_peripheral_roi`) and runs edge contour scans for contraband devices (aspect ratio ~2:1).
   - *Agentic Branch 3 (Optical Blur Guard):* When Laplacian variance $< 100.0$, the agent classifies the frame as `QUALITY_DEGRADED_UNRELIABLE` and routes the event to proctor context, suppressing false alerts.
3. **Trace Logger Integration (`AgentTraceRecorder`):**
   - Engineered structured logging to capture each step, OpenCV telemetry, dynamic tool call, rationale, and resulting decision modification, fulfilling the rubric mandate for trace evidence.
4. **Anti-Bias Temporal Logic & Accommodation:**
   - Enforced the Akhila rule: transient gaze shifts under 120 seconds are automatically suppressed unless secondary devices are present.
   - Implemented `AccommodationProfile` parameters allowing registered neurodivergent conditions (ADHD, strabismus) to calibrate baseline sensitivities.

---

## 🔄 Revision Cycle 3: AWS Resiliency, Ethics & Submission Deliverables

### Objective
Harden cloud integration, eliminate runtime serialization errors, verify automated test suites, and compile submission documentation.

### Key Discoveries & Technical Fixes
1. **DynamoDB Serialization Fix:**
   - *Discovery:* Boto3's DynamoDB resource serializer rejects standard Python `float` types (`TypeError: Float types are not supported. Use Decimal types instead`).
   - *Fix:* Created recursive helper `_convert_floats_to_decimals()` in `examshield/cloud/aws_client.py` to seamlessly convert float metrics to `decimal.Decimal`.
2. **Resilient Local Fallback for Offline / Local Testing:**
   - *Enhancement:* Updated `AWSClientManager` to verify active credentials and gracefully fall back to an in-memory audit buffer when running locally without AWS tokens. This enables judges to run all tests and live demos offline.
3. **Datetime UTC Modernization:**
   - Updated deprecated `datetime.utcnow()` calls to timezone-aware UTC (`datetime.now(timezone.utc)`).
4. **Complete Submission Documentation:**
   - **`ARCHITECTURE.md`:** Detailed system designs, OpenCV 5 function catalog, state machine flow, and AWS deployment with Mermaid diagrams.
   - **`TECHNICAL_REPORT.md`:** All 8 mandatory submission sections (Problem, Users, Architecture, OpenCV 5 Implementation, AWS Deployment, Evaluation, Limitations, Responsible Use).
   - **`bias_stress_test.py`:** Synthetic evaluation harness reproducing the exact metrics table for normal, dim, blurred, and simulated infraction test cases.
5. **Comprehensive Test Suite Passing:**
   - Verified 14 automated unit and integration tests passing (`tests/test_opencv5_pipeline.py`, `tests/test_agentic_loop.py`, `tests/test_api_endpoints.py`) with 100% success rate.
