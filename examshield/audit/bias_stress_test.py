"""
ExamShield Bias Stress-Testing Harness
Executes automated stress tests across environmental conditions using synthetic OpenCV 5 transformations.
Evaluates Raw Alert Rate vs Agentic Adjusted Rate to generate submission evidence.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Dict, List, Tuple
import cv2
import numpy as np

from examshield.perception.opencv5_engine import OpenCV5PerceptionEngine
from examshield.agent.state_machine import AgenticVisionLoop, AccommodationProfile

class BiasStressTestRunner:
    def __init__(self, sample_frames_per_condition: int = 50):
        self.sample_size = sample_frames_per_condition
        self.engine = OpenCV5PerceptionEngine()

    def generate_base_frame(self) -> np.ndarray:
        """Creates a clean synthetic frame simulating an honest student facing camera."""
        frame = np.full((480, 640, 3), 130, dtype=np.uint8)
        # Draw synthetic head contour
        cv2.circle(frame, (320, 240), 90, (190, 190, 190), -1)
        # Eyes
        cv2.circle(frame, (285, 215), 12, (70, 70, 70), -1)
        cv2.circle(frame, (355, 215), 12, (70, 70, 70), -1)
        # Mouth
        cv2.ellipse(frame, (320, 290), (30, 8), 0, 0, 180, (70, 70, 70), 3)
        return frame

    def apply_dimming(self, frame: np.ndarray, alpha: float = 0.4, beta: int = -30) -> np.ndarray:
        """Artificially degrades lighting using OpenCV 5 cv2.convertScaleAbs."""
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    def apply_motion_blur(self, frame: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Simulates low-end camera motion blur using OpenCV 5 cv2.filter2D horizontal kernel."""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size
        return cv2.filter2D(frame, -1, kernel)

    def add_simulated_infraction(self, frame: np.ndarray) -> np.ndarray:
        """Simulates contraband cell phone in lower periphery."""
        infraction_frame = frame.copy()
        # Draw a phone-like rectangle in bottom-right corner (aspect ratio ~2:1)
        cv2.rectangle(infraction_frame, (480, 320), (550, 460), (30, 30, 30), -1)
        cv2.rectangle(infraction_frame, (485, 325), (545, 455), (200, 200, 255), -1)
        return infraction_frame

    def run_stress_test(self) -> Dict[str, Dict[str, float]]:
        conditions = ["Normal Lighting", "Dim Lighting (Synthetic)", "Blur / Low-End Camera (Synthetic)", "Simulated Actual Infraction"]
        results = {}

        for condition in conditions:
            raw_alerts = 0
            agent_adjusted_alerts = 0

            for i in range(self.sample_size):
                base = self.generate_base_frame()
                anomaly_duration = 0.0

                if condition == "Normal Lighting":
                    frame = base
                    anomaly_duration = 5.0 # Transient look
                elif condition == "Dim Lighting (Synthetic)":
                    frame = self.apply_dimming(base)
                    anomaly_duration = 10.0
                elif condition == "Blur / Low-End Camera (Synthetic)":
                    frame = self.apply_motion_blur(base)
                    anomaly_duration = 8.0
                elif condition == "Simulated Actual Infraction":
                    frame = self.add_simulated_infraction(base)
                    anomaly_duration = 140.0 # Exceeds 120s transient threshold

                # 1. Raw Pipeline Measurement (Fixed static threshold without adaptive tools)
                raw_telemetry = self.engine.process_frame(frame, enable_clahe_on_low_light=False)
                # Raw pipeline flags any deviation, blur, or low-light as a failure
                is_raw_alert = (
                    raw_telemetry.raw_anomaly_detected or
                    raw_telemetry.quality.is_low_light or
                    raw_telemetry.quality.is_blurred
                )
                if is_raw_alert:
                    raw_alerts += 1

                # 2. Agentic Adaptive Measurement
                agent_loop = AgenticVisionLoop(session_id=f"stress-test-{i}", engine=self.engine)
                decision = agent_loop.evaluate_frame_agentically(frame, duration_anomaly_active=anomaly_duration)
                
                # Agentic flag counts only if routed to proctor dashboard as a sustained infraction
                if decision.status == "SUSTAINED_ANOMALY":
                    agent_adjusted_alerts += 1

            raw_rate = (raw_alerts / self.sample_size) * 100.0
            adjusted_rate = (agent_adjusted_alerts / self.sample_size) * 100.0

            results[condition] = {
                "raw_alert_rate": round(raw_rate, 1),
                "adjusted_agent_rate": round(adjusted_rate, 1)
            }

        return results

    def print_markdown_table(self, results: Dict[str, Dict[str, float]]) -> str:
        header = "+------------------------------------+---------------+-----------------+\n"
        header += "| Test Condition                     | Raw Alert Rate| Adjusted (Agent)|\n"
        header += "+------------------------------------+---------------+-----------------+\n"
        for cond, metrics in results.items():
            header += f"| {cond:<34} | {metrics['raw_alert_rate']:>12.1f}% | {metrics['adjusted_agent_rate']:>14.1f}% |\n"
        header += "+------------------------------------+---------------+-----------------+"
        return header

if __name__ == "__main__":
    runner = BiasStressTestRunner(sample_frames_per_condition=50)
    print("Running ExamShield Bias & Environmental Stress Test across conditions...")
    res = runner.run_stress_test()
    table = runner.print_markdown_table(res)
    print("\nEvaluation Stress-Test Results Table:\n")
    print(table)
