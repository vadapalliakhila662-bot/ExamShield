"""
ExamShield Core Perception Engine (OpenCV 5 Native Implementation)
Leverages OpenCV 5 modern vision APIs:
  - Face detection & 5-point landmark extraction via cv2.FaceDetectorYN (YuNet) / cv2 fallback
  - Head pose estimation (Yaw, Pitch, Roll) via cv2.solvePnP & cv2.RQDecomp3x3
  - Gaze and eye-direction estimation via iris minimum intensity mapping (cv2.minMaxLoc)
  - Lip-motion measurement via temporal frame differences (cv2.absdiff, cv2.threshold)
  - Secondary phone/contraband object scanning via OpenCV 5 contour morphology and aspect ratios
  - Environmental diagnostics (mean luminance & Laplacian blur variance)
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from pydantic import BaseModel, Field

from examshield.perception.quality_diagnostics import evaluate_frame_quality, QualityDiagnosticResult
from examshield.perception.adaptive_tools import apply_clahe_enhancement, extract_peripheral_roi

class PerceptionTelemetry(BaseModel):
    timestamp_offset_ms: float = 0.0
    quality: QualityDiagnosticResult
    face_detected: bool = False
    head_pose: Dict[str, float] = Field(default_factory=lambda: {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
    gaze_vector: Dict[str, float] = Field(default_factory=lambda: {"horizontal": 0.0, "vertical": 0.0})
    lip_motion_energy: float = 0.0
    secondary_devices_detected: List[str] = Field(default_factory=list)
    clahe_enhanced: bool = False
    raw_anomaly_detected: bool = False
    anomaly_reasons: List[str] = Field(default_factory=list)

class OpenCV5PerceptionEngine:
    """
    Standard OpenCV 5 Perception Engine.
    All computer vision algorithms execute through native OpenCV 5 modules.
    """
    def __init__(
        self,
        min_luminance: float = 45.0,
        blur_threshold: float = 100.0,
        yunet_model_path: Optional[str] = None
    ):
        self.min_luminance = min_luminance
        self.blur_threshold = blur_threshold
        self.prev_mouth_roi: Optional[np.ndarray] = None
        
        # Canonical 3D facial model for cv2.solvePnP (in millimeters)
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye outer corner
            (225.0, 170.0, -135.0),      # Right eye outer corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

        # Initialize OpenCV 5 YuNet Face Detector if model weights exist
        self.yunet_detector = None
        if yunet_model_path and os.path.exists(yunet_model_path):
            try:
                self.yunet_detector = cv2.FaceDetectorYN_create(
                    model=yunet_model_path,
                    config="",
                    input_size=(640, 480),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=5000
                )
            except Exception:
                self.yunet_detector = None

    def detect_face_and_landmarks(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
        """
        Detects primary face bounding box and key landmark points using OpenCV 5.
        Uses cv2.FaceDetectorYN when available, or OpenCV 5 contour/morphological segmentation.
        """
        h, w = frame.shape[:2]
        
        # 1. Native OpenCV 5 YuNet DNN Detection
        if self.yunet_detector is not None:
            self.yunet_detector.setInputSize((w, h))
            _, faces = self.yunet_detector.detect(frame)
            if faces is not None and len(faces) > 0:
                box = faces[0][:4].astype(int)
                x, y, bw, bh = box
                landmarks = faces[0][4:14].reshape((5, 2))
                # 5 landmarks: right_eye, left_eye, nose_tip, right_mouth, left_mouth
                image_points = np.array([
                    landmarks[2], # nose
                    (x + bw * 0.5, y + bh * 0.95), # chin approx
                    landmarks[0], # right eye
                    landmarks[1], # left eye
                    landmarks[3], # right mouth
                    landmarks[4]  # left mouth
                ], dtype=np.float64)
                return True, (x, y, bw, bh), image_points

        # 2. Resilient OpenCV 5 Intensity & Contour Fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Threshold for subject foreground
        _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_box = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 4000:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                # Check aspect ratio typical for human head
                aspect = float(bh) / bw if bw > 0 else 0
                if 0.8 <= aspect <= 1.8 and area > max_area:
                    max_area = area
                    best_box = (bx, by, bw, bh)

        if best_box is not None:
            bx, by, bw, bh = best_box
            image_points = np.array([
                (bx + bw * 0.5, by + bh * 0.5),   # Nose tip
                (bx + bw * 0.5, by + bh * 0.9),   # Chin
                (bx + bw * 0.3, by + bh * 0.35),  # Left eye
                (bx + bw * 0.7, by + bh * 0.35),  # Right eye
                (bx + bw * 0.35, by + bh * 0.75), # Left mouth
                (bx + bw * 0.65, by + bh * 0.75)  # Right mouth
            ], dtype=np.float64)
            return True, best_box, image_points

        return False, None, None

    def estimate_head_pose(
        self,
        image_points: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> Dict[str, float]:
        """
        Calculates 3D Euler angles using OpenCV 5 cv2.solvePnP and cv2.RQDecomp3x3.
        """
        h, w = frame_shape[:2]
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.model_points_3d,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        rmat, _ = cv2.Rodrigues(rvec)
        # OpenCV 5 RQ Decomposition for Euler angles (degrees)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = angles[0], angles[1], angles[2]
        return {
            "yaw": round(float(yaw), 2),
            "pitch": round(float(pitch), 2),
            "roll": round(float(roll), 2)
        }

    def measure_lip_motion(self, mouth_roi: np.ndarray) -> float:
        """
        Measures inter-frame lip motion velocity using cv2.absdiff and cv2.threshold.
        """
        if mouth_roi is None or mouth_roi.size == 0:
            return 0.0

        gray_mouth = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY) if len(mouth_roi.shape) == 3 else mouth_roi
        gray_mouth = cv2.resize(gray_mouth, (64, 48))

        if self.prev_mouth_roi is None:
            self.prev_mouth_roi = gray_mouth
            return 0.0

        diff = cv2.absdiff(self.prev_mouth_roi, gray_mouth)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        energy = float(cv2.mean(thresh)[0])
        self.prev_mouth_roi = gray_mouth
        return round(energy, 2)

    def estimate_gaze_displacement(self, eye_roi: np.ndarray) -> Dict[str, float]:
        """
        Estimates normalized iris offset within the eye bounding region.
        """
        if eye_roi is None or eye_roi.size == 0:
            return {"horizontal": 0.0, "vertical": 0.0}

        gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY) if len(eye_roi.shape) == 3 else eye_roi
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_eye)
        h, w = gray_eye.shape[:2]
        pupil_x, pupil_y = min_loc

        norm_x = (pupil_x - (w / 2.0)) / (w / 2.0) if w > 0 else 0.0
        norm_y = (pupil_y - (h / 2.0)) / (h / 2.0) if h > 0 else 0.0
        return {
            "horizontal": round(float(norm_x), 2),
            "vertical": round(float(norm_y), 2)
        }

    def detect_secondary_objects(self, frame: np.ndarray) -> List[str]:
        """
        Scans for contraband devices (cell phone, notes, extra screens) in the periphery.
        Uses OpenCV 5 Canny edge detection and rectangular contour geometry matching.
        """
        if frame is None or frame.size == 0:
            return []

        detected_devices = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter contours matching a typical phone/tablet form-factor
            if 3000 < area < 60000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(h) / w if w > 0 else 0.0
                if 1.5 <= aspect_ratio <= 2.5 or 0.4 <= aspect_ratio <= 0.65:
                    detected_devices.append("cell_phone")
                    break

        return detected_devices

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_offset_ms: float = 0.0,
        enable_clahe_on_low_light: bool = True
    ) -> PerceptionTelemetry:
        """
        Executes complete OpenCV 5 perception pipeline on a single video frame.
        """
        # Step 1: Initial Quality Diagnostics
        quality = evaluate_frame_quality(frame, self.min_luminance, self.blur_threshold)
        active_frame = frame
        clahe_enhanced = False

        # Step 2: Adaptive CLAHE Enhancement (if low light detected and permitted)
        if quality.is_low_light and enable_clahe_on_low_light:
            active_frame = apply_clahe_enhancement(frame)
            clahe_enhanced = True
            quality = evaluate_frame_quality(active_frame, self.min_luminance, self.blur_threshold)

        # Step 3: Face & Landmark Detection via OpenCV 5
        face_detected, face_box, image_points = self.detect_face_and_landmarks(active_frame)

        head_pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        gaze_vector = {"horizontal": 0.0, "vertical": 0.0}
        lip_motion = 0.0
        anomaly_reasons = []

        if face_detected and image_points is not None:
            bx, by, bw, bh = face_box
            head_pose = self.estimate_head_pose(image_points, active_frame.shape)

            # Eye gaze extraction
            eye_y_start = int(by + bh * 0.2)
            eye_y_end = int(by + bh * 0.5)
            eye_roi = active_frame[eye_y_start:eye_y_end, bx:bx+bw]
            gaze_vector = self.estimate_gaze_displacement(eye_roi)

            # Lip motion extraction
            mouth_y_start = int(by + bh * 0.65)
            mouth_y_end = int(by + bh)
            mouth_roi = active_frame[mouth_y_start:mouth_y_end, bx:bx+bw]
            lip_motion = self.measure_lip_motion(mouth_roi)

            # Anomaly Checks
            if abs(head_pose["yaw"]) > 25.0:
                anomaly_reasons.append(f"Yaw deviation ({head_pose['yaw']} deg)")
            if abs(head_pose["pitch"]) > 20.0:
                anomaly_reasons.append(f"Pitch deviation ({head_pose['pitch']} deg)")
            if abs(gaze_vector["horizontal"]) > 0.4:
                anomaly_reasons.append(f"Gaze deviation ({gaze_vector['horizontal']})")
        else:
            anomaly_reasons.append("No face detected in frame")

        # Step 4: Scan for secondary devices
        secondary_devices = self.detect_secondary_objects(active_frame)
        if secondary_devices:
            anomaly_reasons.append(f"Secondary device detected: {','.join(secondary_devices)}")

        raw_anomaly = len(anomaly_reasons) > 0

        return PerceptionTelemetry(
            timestamp_offset_ms=timestamp_offset_ms,
            quality=quality,
            face_detected=face_detected,
            head_pose=head_pose,
            gaze_vector=gaze_vector,
            lip_motion_energy=lip_motion,
            secondary_devices_detected=secondary_devices,
            clahe_enhanced=clahe_enhanced,
            raw_anomaly_detected=raw_anomaly,
            anomaly_reasons=anomaly_reasons
        )
