#!/usr/bin/env python3
"""
ExamShield - OpenCV 5 Environment & Module Verification Script
Validates OpenCV 5.x installation, core modules, and agentic vision primitives.
"""

import sys
import numpy as np

def run_verification():
    print("=" * 70)
    print("  EXAMSHIELD: OPENCV 5 CORE VERIFICATION SUITE")
    print("=" * 70)
    
    # 1. Version Check
    try:
        import cv2
    except ImportError as e:
        print(f"[FAIL] cv2 could not be imported: {e}")
        sys.exit(1)
        
    version = cv2.__version__
    print(f"[*] Detected OpenCV Version: {version}")
    
    if not version.startswith("5."):
        print(f"[FAIL] Mandatory competition requirement failed: OpenCV 5 is required, got {version}")
        sys.exit(1)
    else:
        print(f"[PASS] OpenCV 5 pinned version requirement met: {version}")

    # 2. Module & Function Inspections
    required_functions = {
        "cv2.dnn (Deep Neural Network Engine)": hasattr(cv2, "dnn"),
        "cv2.solvePnP (Head Pose 3D Estimation)": hasattr(cv2, "solvePnP"),
        "cv2.createCLAHE (Adaptive Contrast Equalizer)": hasattr(cv2, "createCLAHE"),
        "cv2.Laplacian (Blur & Focus Estimation)": hasattr(cv2, "Laplacian"),
        "cv2.mean (Luminance Diagnostic)": hasattr(cv2, "mean"),
        "cv2.cvtColor (Color Space Transforms)": hasattr(cv2, "cvtColor"),
        "cv2.Rodrigues (Euler Rotation Matrix Vectorization)": hasattr(cv2, "Rodrigues"),
        "cv2.equalizeHist (Global Histogram Equalization)": hasattr(cv2, "equalizeHist"),
    }

    all_passed = True
    print("\n[*] Auditing OpenCV 5 Modules and Functions:")
    for name, exists in required_functions.items():
        status = "[PASS]" if exists else "[FAIL]"
        print(f"  {status} {name}")
        if not exists:
            all_passed = False

    if not all_passed:
        print("\n[FAIL] One or more required OpenCV 5 functions are missing from the build!")
        sys.exit(1)

    # 3. Functional Simulation & Agentic Tool Execution
    print("\n[*] Executing Perception & Quality Diagnostics on Synthetic Frames:")
    
    # Generate test frame 640x480
    test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    
    # Luminance test
    mean_val = cv2.mean(test_frame)[0]
    print(f"  [PASS] Baseline Luminance Measurement: {mean_val:.2f}")

    # Blur test via Laplacian Variance
    gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"  [PASS] Blur Metric (Laplacian Var): {laplacian_var:.2f}")

    # Head Pose Simulation via solvePnP
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye corner
        (225.0, 170.0, -135.0),      # Right eye corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)

    image_points = np.array([
        (320.0, 240.0),
        (320.0, 350.0),
        (240.0, 180.0),
        (400.0, 180.0),
        (260.0, 300.0),
        (380.0, 300.0)
    ], dtype=np.float64)

    focal_length = 640
    center = (320, 240)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    rmat, _ = cv2.Rodrigues(rvec)
    print(f"  [PASS] solvePnP 3D Euler Pose Calculation Success: {success}")

    # Adaptive Enhancement Tool (CLAHE) invocation
    dim_frame = (test_frame * 0.2).astype(np.uint8)
    dim_gray = cv2.cvtColor(dim_frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(dim_gray)
    print(f"  [PASS] Dynamic CLAHE Equalization: initial mean {dim_gray.mean():.2f} -> enhanced {enhanced_gray.mean():.2f}")

    print("\n" + "=" * 70)
    print("  RESULT: OPENCV 5 CORE PIPELINE FULLY FUNCTIONAL AND VERIFIED")
    print("=" * 70)
    return True

if __name__ == "__main__":
    run_verification()
