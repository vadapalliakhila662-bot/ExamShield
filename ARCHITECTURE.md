# ExamShield: Architecture & Agentic Vision Specification

ExamShield is an open, human-centered proctoring and fairness audit architecture built on **OpenCV 5** and **AWS Cloud Infrastructure**. Rather than functioning as a black-box punitive classifier, ExamShield uses an active agentic perception loop and procedural justice framework.

---

## 1. High-Level System Architecture

ExamShield bifurcates high-frequency perception (executed edge-side or local-stream using OpenCV 5) from asynchronous cloud persistence and LLM consistency auditing on AWS.

```mermaid
graph TD
    subgraph ClientEdge [Edge / Participant Client]
        Webcam[Webcam Video Stream 720p @ 5 FPS] --> PreDiagnostics[OpenCV 5 Quality Diagnostics]
        PreDiagnostics --> CorePerception[OpenCV 5 Perception Engine]
        CorePerception --> AgenticLoop[Agentic Vision State Machine]
        AgenticLoop -->|Dynamic Tool Call| CLAHE[cv2.createCLAHE Adaptive Contrast]
        AgenticLoop -->|Dynamic Tool Call| PeripheralROI[extract_peripheral_roi + Contours]
    end

    subgraph AWSCloud [AWS Cloud Infrastructure]
        AgenticLoop -->|Edge Event Triggered| S3[AWS S3 Evidence Clips 30-Day Auto-Purge]
        AgenticLoop -->|Time-Series Telemetry| DynamoDB[Amazon DynamoDB ExamShield-Audit]
        DynamoDB --> FairnessAudit[AWS Lambda / EventBridge Fairness Auditor]
        
        AppealPortal[Student Appeal Submission] --> BedrockAgent[AWS Bedrock Claude 3.5 Sonnet]
        DynamoDB --> BedrockAgent
    end

    subgraph HumanReview [Human-in-the-Loop Proctor Console]
        DynamoDB --> ProctorDashboard[Proctor Review Dashboard]
        BedrockAgent --> ProctorDashboard
        ProctorDashboard -->|Mandatory Human Sign-off| FinalOutcome[Final Proctor Determination]
    end
```

---

## 2. OpenCV 5 Perception Architecture & Module Catalog

Every computer vision primitive in ExamShield runs natively through OpenCV 5 (`opencv-python==5.0.0.93`).

| Task / Feature | OpenCV 5 Module | Specific OpenCV 5 Functions / Classes | Algorithmic Purpose |
| :--- | :--- | :--- | :--- |
| **Environmental Luminance** | `cv2` | `cv2.mean()`, `cv2.cvtColor()` | Determines mean pixel intensity ($\mu$). Triggers adaptive CLAHE if $\mu < 45.0$. |
| **Optical Blur Detection** | `cv2` | `cv2.Laplacian()`, `cv2.CV_64F`, `var()` | Computes second-order spatial derivative variance ($\sigma^2$). Marks frame as UNRELIABLE if $\sigma^2 < 100.0$. |
| **Adaptive Contrast Tool** | `cv2` | `cv2.createCLAHE()`, `cv2.split()`, `cv2.merge()`, `cv2.COLOR_BGR2LAB` | Dynamically equalizes L-channel in LAB space when low-light is diagnosed. |
| **Face & 5-Point Landmarks** | `cv2.dnn` | `cv2.FaceDetectorYN_create()`, `FaceDetectorYN.detect()` | Native OpenCV 5 YuNet deep neural network extracting bounding box and 5 facial points. |
| **Head Pose 3D Estimation** | `cv2.calib3d` | `cv2.solvePnP()`, `cv2.Rodrigues()`, `cv2.RQDecomp3x3()` | Solves Perspective-n-Point against canonical 3D facial model to derive 3D Euler angles (Yaw, Pitch, Roll). |
| **Gaze / Iris Displacement** | `cv2` | `cv2.minMaxLoc()`, `cv2.resize()` | Localizes pupil minimum intensity coordinate relative to eye canthi bounding boundaries. |
| **Lip Motion Velocity** | `cv2` | `cv2.absdiff()`, `cv2.threshold()`, `cv2.mean()` | Computes inter-frame temporal subtraction energy around mouth region to detect vocalization/speaking. |
| **Contraband / Phone Scan** | `cv2` | `cv2.Canny()`, `cv2.findContours()`, `cv2.boundingRect()`, `cv2.contourArea()` | Secondary tool triggered on gaze deviation to inspect peripheral ROI for device aspect ratios (~2:1). |

---

## 3. Agentic Perception-Decision-Action Loop

Unlike traditional static proctoring pipelines that blindly pass video through a fixed chain of detectors, ExamShield is **strictly agentic**: perception findings actively branch the execution graph, dynamically dispatch specialized computer vision tools, and adjust operational thresholds.

```mermaid
stateDiagram-v2
    [*] --> IngestFrame: OpenCV 5 Frame Capture
    IngestFrame --> CheckQuality: Evaluate Luminance & Blur

    state CheckQuality {
        [*] --> MeasureLuminance
        MeasureLuminance --> MeasureLaplacianBlur
    }

    CheckQuality --> LowLightDetected: Mean Luminance < 45.0
    state LowLightDetected {
        [*] --> DispatchCLAHE: Dynamic Tool Call cv2.createCLAHE
        DispatchCLAHE --> ReAnalyzeQuality
    }
    LowLightDetected --> FaceAndPoseTracking

    CheckQuality --> BlurDetected: Laplacian Var < 100.0
    BlurDetected --> MarkUnreliable: Optical Blur Guard
    MarkUnreliable --> RouteHumanContext: Suppress Alert & Route to Proctor Context

    CheckQuality --> FaceAndPoseTracking: Frame Sharp & Well-Lit

    state FaceAndPoseTracking {
        [*] --> RunYuNetDetection
        RunYuNetDetection --> SolvesPnP3DPose
        SolvesPnP3DPose --> MeasureGazeAndLips
    }

    FaceAndPoseTracking --> CheckAccommodation: Gaze or Yaw Deviation Detected
    state CheckAccommodation {
        [*] --> InspectStudentProfile
        InspectStudentProfile --> AdjustThresholds: Active Strabismus / ADHD Profile
        InspectStudentProfile --> RetainNominalThresholds: Standard Baseline
    }

    CheckAccommodation --> DispatchPeripheralScan: Dynamic Secondary Tool Call
    state DispatchPeripheralScan {
        [*] --> ExtractPeripheralROI
        ExtractPeripheralROI --> ScanContrabandDevices
    }

    DispatchPeripheralScan --> TransientSuppression: Duration < 120s & No Devices
    DispatchPeripheralScan --> SustainedAnomaly: Duration >= 120s OR Device Detected

    TransientSuppression --> [*]: Log Transient (Alert Suppressed)
    SustainedAnomaly --> RouteDynamoDB: Record Telemetry for Review
    RouteDynamoDB --> ProctorReview: Human Proctor Dashboard
```

---

## 4. AWS Cloud Infrastructure Architecture

```mermaid
flowchart LR
    subgraph EdgeDevice [Edge Device / Browser]
        ClientCV[OpenCV 5 Engine]
        ClientAgent[Agentic State Machine]
    end

    subgraph AWSCloudServices [AWS Architecture]
        S3Bucket[(Amazon S3 Evidence Bucket<br/>30-Day Expiration Lifecycle)]
        DynamoAudit[(DynamoDB Table:<br/>ExamShield-Audit)]
        DynamoSessions[(DynamoDB Table:<br/>ExamShield-Sessions)]
        BedrockService[Amazon Bedrock<br/>Claude 3.5 Sonnet]
        LambdaFairness[AWS Lambda / EventBridge<br/>Fairness & Bias Auditor]
    end

    subgraph ReviewAndAppeal [Governance & Human Control]
        ProctorUI[Proctor Review Dashboard]
        StudentAppeal[Student Post-Exam Appeal Portal]
    end

    ClientCV --> ClientAgent
    ClientAgent -->|Upload 3s WebM clip on sustained anomaly| S3Bucket
    ClientAgent -->|Persist telemetry & trace log| DynamoAudit
    ClientAgent -->|Session status| DynamoSessions

    DynamoAudit --> LambdaFairness
    LambdaFairness -->|Disparate Impact Ratio Skew| DynamoAudit

    StudentAppeal -->|Student Claim| BedrockService
    DynamoAudit -->|Physical OpenCV Telemetry| BedrockService
    BedrockService -->|Consistency JSON Verdict| DynamoAudit

    DynamoAudit --> ProctorUI
    S3Bucket --> ProctorUI
```

---

## 5. Cloud-Optimized Vision with COOL (AWS Graviton)

ExamShield provides native architectural support for **COOL** (Cloud-Optimized OpenCV Library) running on AWS Graviton instances:
- **Instance Families**: AWS Graviton3 / Graviton4 (e.g., `c7g.xlarge` / `c8g.xlarge`).
- **Acceleration**: Leverages Arm Neoverse V1/V2 cores with Neon vector extensions for OpenCV SIMD optimizations.
- **Coexistence Architecture**: Local or client-side frames run standard OpenCV 5; high-throughput batch re-examinations or institutional centralized deployments execute on Graviton instances running COOL.
- **Cost Reduction**: Achieves ~20% lower instance cost and up to 40% higher price-performance over comparable x86 compute.
