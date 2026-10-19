"""
ExamShield Quality Diagnostics using OpenCV 5
Calculates frame luminance and Laplacian blur variance to assess video fidelity.
"""

from typing import Dict, Any
import cv2
import numpy as np
from pydantic import BaseModel

class QualityDiagnosticResult(BaseModel):
    mean_luminance: float
    blur_score: float
    is_low_light: bool
    is_blurred: bool
    is_degraded: bool
    diagnostic_message: str

def evaluate_frame_quality(
    frame: np.ndarray,
    min_luminance: float = 45.0,
    blur_threshold: float = 100.0
) -> QualityDiagnosticResult:
    """
    Evaluates environmental quality of an incoming frame using OpenCV 5 primitives.
    
    OpenCV 5 functions used:
      - cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY): Grayscale intensity transform
      - cv2.mean(gray): Mean pixel intensity calculation
      - cv2.Laplacian(gray, cv2.CV_64F): Second-order spatial derivative for sharpness
    """
    if frame is None or frame.size == 0:
        return QualityDiagnosticResult(
            mean_luminance=0.0,
            blur_score=0.0,
            is_low_light=True,
            is_blurred=True,
            is_degraded=True,
            diagnostic_message="Empty or invalid frame received."
        )

    # 1. Grayscale conversion via OpenCV 5
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Mean Luminance via cv2.mean
    mean_luminance = float(cv2.mean(gray)[0])

    # 3. Blur Detection via Laplacian spatial derivative
    laplacian_matrix = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian_matrix.var())

    is_low_light = mean_luminance < min_luminance
    is_blurred = blur_score < blur_threshold
    is_degraded = is_low_light or is_blurred

    messages = []
    if is_low_light:
        messages.append(f"Low lighting detected ({mean_luminance:.1f} < {min_luminance})")
    if is_blurred:
        messages.append(f"Optical blur detected ({blur_score:.1f} < {blur_threshold})")
    if not messages:
        messages.append("Optimal frame quality")

    return QualityDiagnosticResult(
        mean_luminance=round(mean_luminance, 2),
        blur_score=round(blur_score, 2),
        is_low_light=is_low_light,
        is_blurred=is_blurred,
        is_degraded=is_degraded,
        diagnostic_message="; ".join(messages)
    )
