"""
ExamShield AWS Client Manager
Provides resilient interfaces for AWS S3, DynamoDB, and Bedrock.
Includes local fallback emulation when live AWS credentials are not configured.
"""

from decimal import Decimal
import os
import json
import logging
from typing import Dict, Any, List, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from examshield.config import settings
from examshield.cloud.dynamodb_schema import ExamShieldAuditEvent

logger = logging.getLogger("ExamShield-AWS")

def _convert_floats_to_decimals(obj: Any) -> Any:
    """Recursively converts Python float objects to Decimals for DynamoDB serialization."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: _convert_floats_to_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_floats_to_decimals(v) for v in obj]
    return obj

class AWSClientManager:
    def __init__(self):
        self.region = settings.AWS_DEFAULT_REGION
        self.is_connected = False
        self._local_audit_store: List[Dict[str, Any]] = []

        try:
            self.session = boto3.Session(region_name=self.region)
            self.dynamodb = self.session.resource("dynamodb")
            self.s3 = self.session.client("s3")
            self.bedrock = self.session.client("bedrock-runtime")
            # Quick check to see if credentials exist
            creds = self.session.get_credentials()
            if creds is None:
                raise NoCredentialsError()
            self.is_connected = True
            logger.info("Successfully initialized live AWS clients.")
        except (NoCredentialsError, Exception) as e:
            logger.warning(f"AWS credentials not detected ({e}). Using resilient local in-memory fallback store.")
            self.dynamodb = None
            self.s3 = None
            self.bedrock = None
            self.is_connected = False

    def log_audit_event(self, event: ExamShieldAuditEvent) -> bool:
        """
        Records an audit event into DynamoDB ExamShield-Audit table.
        """
        payload = event.model_dump()
        if self.is_connected and self.dynamodb:
            try:
                table = self.dynamodb.Table(settings.DYNAMODB_AUDIT_TABLE)
                dynamo_payload = _convert_floats_to_decimals(payload)
                table.put_item(Item=dynamo_payload)
                return True
            except Exception as e:
                logger.warning(f"Failed to record DynamoDB item ({e}). Falling back to local audit buffer.")
                self._local_audit_store.append(payload)
                return True
        else:
            self._local_audit_store.append(payload)
            return True

    def query_audit_events(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Fetches all audit events for a session.
        """
        if self.is_connected and self.dynamodb:
            try:
                table = self.dynamodb.Table(settings.DYNAMODB_AUDIT_TABLE)
                from boto3.dynamodb.conditions import Key
                response = table.query(
                    KeyConditionExpression=Key("SessionID").eq(session_id)
                )
                return response.get("Items", [])
            except Exception as e:
                logger.error(f"Error querying DynamoDB: {e}")
                return [e for e in self._local_audit_store if e.get("SessionID") == session_id]
        else:
            return [e for e in self._local_audit_store if e.get("SessionID") == session_id]

    def generate_s3_upload_url(self, file_key: str, expires_in: int = 3600) -> str:
        """
        Generates a presigned S3 upload URL for edge frame uploads.
        """
        if self.is_connected and self.s3:
            try:
                return self.s3.generate_presigned_url(
                    ClientMethod="put_object",
                    Params={"Bucket": settings.S3_EVIDENCE_BUCKET, "Key": file_key},
                    ExpiresIn=expires_in
                )
            except ClientError as e:
                logger.error(f"Presigned URL generation failed: {e}")
                return f"https://{settings.S3_EVIDENCE_BUCKET}.s3.{self.region}.amazonaws.com/{file_key}"
        return f"mock://s3-local/{settings.S3_EVIDENCE_BUCKET}/{file_key}"
