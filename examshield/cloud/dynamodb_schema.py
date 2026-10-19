"""
ExamShield DynamoDB Schema and Telemetry Data Models
Defines schema for ExamShield-Audit and ExamShield-Sessions tables.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class ExamShieldAuditEvent(BaseModel):
    SessionID: str
    Timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    StudentId: str
    FlagType: str
    ConfidenceScore: float
    LightingMetrics: Dict[str, float]
    BlurScore: float
    HeadPose: Dict[str, float]
    GazeVector: Dict[str, float]
    ToolsDispatched: List[str]
    StudentClaim: Optional[str] = None
    ProctorStatus: str = "PENDING_REVIEW" # PENDING_REVIEW, UPHELD, DISMISSED, AUTO_SUPPRESSED
    AccommodationProfile: Dict[str, Any] = Field(default_factory=dict)
    EvidenceClipS3Uri: Optional[str] = None
    BedrockConsistencyVerdict: Optional[Dict[str, Any]] = None

DYNAMODB_AUDIT_TABLE_SPEC = {
    "TableName": "ExamShield-Audit",
    "KeySchema": [
        {"AttributeName": "SessionID", "KeyType": "HASH"},  # Partition Key
        {"AttributeName": "Timestamp", "KeyType": "RANGE"}  # Sort Key
    ],
    "AttributeDefinitions": [
        {"AttributeName": "SessionID", "AttributeType": "S"},
        {"AttributeName": "Timestamp", "AttributeType": "S"}
    ],
    "BillingMode": "PAY_PER_REQUEST"
}

DYNAMODB_SESSIONS_TABLE_SPEC = {
    "TableName": "ExamShield-Sessions",
    "KeySchema": [
        {"AttributeName": "SessionID", "KeyType": "HASH"}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "SessionID", "AttributeType": "S"}
    ],
    "BillingMode": "PAY_PER_REQUEST"
}
