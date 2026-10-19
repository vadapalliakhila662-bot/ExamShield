"""
ExamShield Adaptive Vision Tools using OpenCV 5
Tools dispatched dynamically by the Agentic loop when environmental degradation is identified.
"""

from typing import Tuple
import cv2
import numpy as np

def apply_clahe_enhancement(
    frame: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) using OpenCV 5.
    Converts BGR to LAB color space, equalizes the L (luminance) channel, and converts back.
    
    OpenCV 5 functions used:
      - cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
      - cv2.createCLAHE(clipLimit=..., tileGridSize=...)
      - cv2.split(lab) / cv2.merge(...)
      - cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    """
    if frame is None or frame.size == 0:
        return frame

    # Convert to LAB color space to isolate luminance from chromaticity
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Initialize and apply OpenCV 5 CLAHE algorithm
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    equalized_l = clahe.apply(l_channel)

    # Re-merge channels and transform back to BGR
    merged_lab = cv2.merge((equalized_l, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr

def extract_peripheral_roi(
    frame: np.ndarray,
    margin_ratio: float = 0.25
) -> np.ndarray:
    """
    Extracts the peripheral boundaries of the video frame where contraband devices
    (e.g., cell phones, notes) are commonly positioned outside the primary face region.
    """
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    # Crop outer periphery: bottom and side boundaries
    y_start = int(h * (1.0 - margin_ratio))
    return frame[y_start:h, :]
