"""
ExamShield Cloud & AWS Integration Module
"""
from examshield.cloud.aws_client import AWSClientManager
from examshield.cloud.dynamodb_schema import ExamShieldAuditEvent, DYNAMODB_AUDIT_TABLE_SPEC
from examshield.cloud.appeal_agent import BedrockAppealAgent, AppealEvaluationResult

__all__ = [
    "AWSClientManager",
    "ExamShieldAuditEvent",
    "DYNAMODB_AUDIT_TABLE_SPEC",
    "BedrockAppealAgent",
    "AppealEvaluationResult"
]
