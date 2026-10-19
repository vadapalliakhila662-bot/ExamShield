"""
ExamShield System Configuration and Environment Variables
Pins OpenCV 5 and AWS parameters.
"""

import os
from pydantic import BaseModel, Field

class SystemSettings(BaseModel):
    # Core Vision Configuration
    OPENCV_MAJOR_VERSION: int = Field(default=5, description="Mandatory OpenCV 5 major release")
    MIN_LUMINANCE: float = Field(default=45.0, description="Minimum acceptable mean pixel intensity (0-255)")
    BLUR_LAPLACIAN_THRESHOLD: float = Field(default=100.0, description="Laplacian variance threshold for sharp frame")
    
    # Head Pose & Gaze Thresholds
    MAX_YAW_DEGREES: float = Field(default=25.0, description="Horizontal head rotation limit before anomaly flag")
    MAX_PITCH_DEGREES: float = Field(default=20.0, description="Vertical head rotation limit before anomaly flag")
    MAX_GAZE_OFFSET: float = Field(default=0.35, description="Normalized iris displacement threshold")
    
    # Temporal Anti-Bias Logic
    TRANSIENT_WINDOW_SECONDS: float = Field(default=120.0, description="Threshold for transient looking away (suppress alert)")
    SUSTAINED_WINDOW_SECONDS: float = Field(default=600.0, description="Threshold for sustained persistent absence")
    
    # AWS Infrastructure
    AWS_DEFAULT_REGION: str = Field(default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    DYNAMODB_AUDIT_TABLE: str = Field(default=os.getenv("DYNAMODB_AUDIT_TABLE", "ExamShield-Audit"))
    DYNAMODB_SESSIONS_TABLE: str = Field(default=os.getenv("DYNAMODB_SESSIONS_TABLE", "ExamShield-Sessions"))
    S3_EVIDENCE_BUCKET: str = Field(default=os.getenv("S3_EVIDENCE_BUCKET", "examshield-evidence-store"))
    BEDROCK_MODEL_ID: str = Field(
        default=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    )
    
    # COOL (Cloud-Optimized OpenCV Library)
    ENABLE_COOL: bool = Field(default=os.getenv("ENABLE_COOL_ACCELERATION", "false").lower() == "true")
    COOL_INSTANCE_TYPE: str = Field(default=os.getenv("COOL_INSTANCE_TYPE", "c7g.xlarge"))

settings = SystemSettings()
