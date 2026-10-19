# 🛡️ ExamShield: Human-Centered Agentic Vision Proctoring

> **Competition Submission**: OpenCV 5 AI Hackathon  
> **Primary Track**: Agentic Vision with OpenCV 5  
> **Secondary / Bonus Track**: Cloud-Optimized Vision with COOL (AWS Graviton)  
> **OpenCV Version**: `5.0.0.93` (strictly pinned)  
> **Cloud Provider**: Amazon Web Services (AWS)

ExamShield is a human-centered, procedural justice proctoring architecture that replaces brittle, discriminatory black-box classifiers with an **active agentic vision loop** built on **OpenCV 5** and **AWS Cloud Infrastructure**.

---

## 🌟 Key Differentiators

- **Strictly Agentic Vision (Not a Static Pipeline):** Perception findings directly alter subsequent tool invocations (dynamic `cv2.createCLAHE` dispatch, peripheral ROI extraction) and modify operational thresholds based on environmental conditions.
- **Akhila Temporal Rule:** Suppresses alerts for transient natural lookaways ($\le 120$ seconds) unless secondary contraband objects are verified.
- **Zero Automated Penalties:** The software cannot cancel exams or issue academic sanctions. All flags are routed to a human proctor with full contextual evidence.
- **Fairness & Demographic Bias Auditing:** Continuously computes Disparate Impact Ratios (DIR) across lighting and camera quality buckets to prevent discrimination.
- **Bedrock Consistency Appeal Agent:** Uses Claude 3.5 Sonnet to evaluate student post-exam explanations against physical OpenCV 5 kinematics.
- **Privacy by Design:** Zero continuous cloud streaming; 3-second incident clips stored in Amazon S3 auto-purge after 30 days.

---

## 📋 Pinned Dependencies & Environment

The core perception engine strictly runs on **OpenCV 5.0.0.93**:

```txt
opencv-python==5.0.0.93
numpy==2.4.6
boto3==1.43.114
botocore==1.43.114
fastapi==0.146.0
uvicorn==0.54.0
pydantic==2.14.1
pydantic-core==2.50.1
starlette==1.9.0
scipy==1.18.0
matplotlib==3.10.9
pytest==9.1.1
requests==2.34.2
httpx==0.28.1
```

---

## 🚀 Quickstart & Verification

### 1. Verify OpenCV 5 Installation
Run our standalone verification script to test OpenCV 5 and its perception primitives:

```bash
python verify_opencv5.py
```

Expected output:
```
======================================================================
  EXAMSHIELD: OPENCV 5 CORE VERIFICATION SUITE
======================================================================
[*] Detected OpenCV Version: 5.0.0
[PASS] OpenCV 5 pinned version requirement met: 5.0.0
[*] Auditing OpenCV 5 Modules and Functions:
  [PASS] cv2.dnn (Deep Neural Network Engine)
  [PASS] cv2.solvePnP (Head Pose 3D Estimation)
  [PASS] cv2.createCLAHE (Adaptive Contrast Equalizer)
  [PASS] cv2.Laplacian (Blur & Focus Estimation)
  [PASS] cv2.mean (Luminance Diagnostic)
  [PASS] cv2.cvtColor (Color Space Transforms)
  [PASS] cv2.Rodrigues (Euler Rotation Matrix Vectorization)
  [PASS] cv2.equalizeHist (Global Histogram Equalization)
...
======================================================================
  RESULT: OPENCV 5 CORE PIPELINE FULLY FUNCTIONAL AND VERIFIED
======================================================================
```

### 2. Run Automated Test Suite
Execute the full pytest suite (perception, state machine, and API):

```bash
python -m pytest tests/
```

### 3. Run Bias & Environmental Stress Test
Run the programmatic stress test across normal, dim, blurred, and simulated infraction conditions:

```bash
python examshield/audit/bias_stress_test.py
```

### 4. Start the Application & Proctor Dashboard
Launch the FastAPI service locally:

```bash
uvicorn examshield.web.api:app --reload --port 8000
```
Open your browser to:
- **Proctor Dashboard & Diagnostics UI**: `http://localhost:8000/`
- **Interactive Swagger Documentation**: `http://localhost:8000/docs`
- **Health & OpenCV 5 Status**: `http://localhost:8000/health`

---

## 🐳 Docker Build & AWS Deployment

### Build the Docker Container
```bash
docker build -t examshield:latest .
```
*(The Dockerfile automatically runs `python -c "import cv2; assert cv2.__version__.startswith('5')"` at build time).*

### Run with Docker Compose
```bash
docker-compose up --build -d
```

---

## 🏛️ OpenCV 5 Function Catalog

| Function / Module | Purpose |
| :--- | :--- |
| `cv2.mean()` | Real-time luminance calculation to assess room lighting |
| `cv2.Laplacian()` | Second-order spatial derivative variance to detect lens blur / focus degradation |
| `cv2.createCLAHE()` | Dynamic adaptive contrast enhancement tool for dim frames |
| `cv2.solvePnP()` | 3D Perspective-n-Point pose calculation for pitch, yaw, and roll |
| `cv2.RQDecomp3x3()` | Decomposes 3D rotation matrix into Euler rotation angles |
| `cv2.minMaxLoc()` | Locates pupil minimum intensity coordinates for gaze direction |
| `cv2.absdiff()` | Inter-frame pixel subtraction to measure mouth/lip movement |
| `cv2.findContours()` | Dispatched secondary tool scanning peripheral ROI for contraband devices |

---

## 🔒 Responsible Use & Data Ethics

1. **Simulated & Consented Data Only**: No student video from real exams was collected or utilized.
2. **Zero Automated Punishment**: No student is penalized without human sign-off.
3. **Data Auto-Purge**: All S3 evidence is hard-deleted automatically after 30 days.
4. **48-Hour Appeal Buffer**: Students have procedural rights to explain anomalies before proctors make determinations.
