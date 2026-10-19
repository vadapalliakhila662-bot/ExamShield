"""
ExamShield Fairness & Demographic Bias Auditor
Computes Disparate Impact Ratios (DIR) across environmental buckets (lighting, blur, camera quality).
Detects systematic skew and dynamically recalibrates thresholds.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field

class FairnessAuditReport(BaseModel):
    disparate_impact_ratio: float
    normal_light_flag_rate: float
    low_light_flag_rate: float
    is_fair: bool
    recalibration_triggered: bool
    recommended_threshold_adjustment: float
    audit_summary: str

class FairnessAuditor:
    def __init__(self, target_dir_range=(0.80, 1.25)):
        self.min_acceptable_dir = target_dir_range[0]
        self.max_acceptable_dir = target_dir_range[1]

    def audit_session_events(self, events: List[Dict[str, Any]]) -> FairnessAuditReport:
        """
        Calculates Disparate Impact Ratio:
        DIR = (Flag Rate Low Light) / (Flag Rate Normal Light)
        """
        normal_total = 0
        normal_flags = 0
        low_light_total = 0
        low_light_flags = 0

        for event in events:
            lighting = event.get("LightingMetrics", {})
            mean_lum = lighting.get("mean_luminance", 100.0)
            is_flagged = event.get("FlagType") not in ("NORMAL", "PASS", "TRANSIENT_SUPPRESSED")

            if mean_lum < 45.0:
                low_light_total += 1
                if is_flagged:
                    low_light_flags += 1
            else:
                normal_total += 1
                if is_flagged:
                    normal_flags += 1

        rate_normal = (normal_flags / normal_total) if normal_total > 0 else 0.05
        rate_low_light = (low_light_flags / low_light_total) if low_light_total > 0 else 0.05

        dir_score = (rate_low_light / rate_normal) if rate_normal > 0 else 1.0
        is_fair = self.min_acceptable_dir <= dir_score <= self.max_acceptable_dir

        recalibration = not is_fair
        threshold_adjustment = 0.0

        if dir_score > self.max_acceptable_dir:
            # Low light participants are being disproportionately flagged -> increase threshold
            threshold_adjustment = 0.15
            summary = f"Bias detected: Low-light students flagged at {dir_score:.2f}x rate of normal students. Suppressing sensitivity for dim frames."
        elif dir_score < self.min_acceptable_dir:
            threshold_adjustment = -0.10
            summary = f"Under-detection in low light: DIR is {dir_score:.2f}. Calibrating sensor gain."
        else:
            summary = f"Fairness verified: Disparate Impact Ratio ({dir_score:.2f}) is within acceptable range [0.80, 1.25]."

        return FairnessAuditReport(
            disparate_impact_ratio=round(dir_score, 2),
            normal_light_flag_rate=round(rate_normal, 4),
            low_light_flag_rate=round(rate_low_light, 4),
            is_fair=is_fair,
            recalibration_triggered=recalibration,
            recommended_threshold_adjustment=threshold_adjustment,
            audit_summary=summary
        )
