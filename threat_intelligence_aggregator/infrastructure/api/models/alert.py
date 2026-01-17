"""
Alert API models (DTOs).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AlertResponse(BaseModel):
    """Alert response model."""

    alert_id: str = Field(description="Alert identifier")
    alert_type: str = Field(
        description="Alert type (NEW_CRITICAL_CVE, EMERGING_THREAT, IOC_DETECTED, etc.)"
    )
    severity: str = Field(description="Severity level (CRITICAL, HIGH, MEDIUM, LOW, INFO)")
    title: str = Field(description="Alert title")
    description: str = Field(description="Alert description")
    created_at: str = Field(description="Creation date (ISO format)")
    status: str = Field(
        description="Status (NEW, ACKNOWLEDGED, IN_PROGRESS, RESOLVED, FALSE_POSITIVE, IGNORED)"
    )
    source_entity_type: str = Field(description="Source entity type (CVE, ThreatIntel, IOC, Topic)")
    source_entity_id: str = Field(description="Source entity ID")
    related_cves: list[str] = Field(default_factory=list, description="Related CVE IDs")
    related_iocs: list[str] = Field(default_factory=list, description="Related IOC IDs")
    related_topics: list[str] = Field(default_factory=list, description="Related topic IDs")
    tags: list[str] = Field(default_factory=list, description="Tags")
    actionable_items: list[str] = Field(default_factory=list, description="Recommended actions")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    acknowledged_at: str | None = Field(default=None, description="Acknowledgement date")
    acknowledged_by: str | None = Field(default=None, description="Acknowledged by")
    resolved_at: str | None = Field(default=None, description="Resolution date")
    resolved_by: str | None = Field(default=None, description="Resolved by")
    resolution_notes: str = Field(default="", description="Resolution notes")
    notification_sent: bool = Field(description="Whether notification was sent")
    age_hours: float = Field(description="Alert age in hours")
    is_active: bool = Field(description="Whether alert is active")
    is_critical: bool = Field(description="Whether alert is critical")

    class Config:
        json_schema_extra = {
            "example": {
                "alert_id": "alert_20240115_0001",
                "alert_type": "NEW_CRITICAL_CVE",
                "severity": "CRITICAL",
                "title": "Critical RCE Vulnerability in Apache HTTP Server",
                "description": "CVE-2024-1234 allows remote code execution...",
                "created_at": "2024-01-15T10:00:00Z",
                "status": "NEW",
                "source_entity_type": "CVE",
                "source_entity_id": "CVE-2024-1234",
                "related_cves": ["CVE-2024-1234"],
                "related_iocs": [],
                "related_topics": [],
                "tags": ["rce", "apache", "web-server"],
                "actionable_items": ["Upgrade to Apache 2.4.53", "Review access logs"],
                "confidence_score": 0.95,
                "acknowledged_at": None,
                "acknowledged_by": None,
                "resolved_at": None,
                "resolved_by": None,
                "resolution_notes": "",
                "notification_sent": False,
                "age_hours": 2.5,
                "is_active": True,
                "is_critical": True,
            }
        }


class AlertFilterParams(BaseModel):
    """Alert filter parameters."""

    status: str | None = Field(default=None, description="Filter by status")
    severity: str | None = Field(default=None, description="Filter by severity")
    alert_type: str | None = Field(default=None, description="Filter by alert type")
    is_active: bool | None = Field(default=None, description="Filter by active status")
    is_critical: bool | None = Field(default=None, description="Filter critical alerts only")
    min_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum confidence score"
    )


class AlertAcknowledgeRequest(BaseModel):
    """Alert acknowledgement request."""

    acknowledged_by: str = Field(min_length=1, description="User who acknowledges the alert")


class AlertResolveRequest(BaseModel):
    """Alert resolution request."""

    resolved_by: str = Field(min_length=1, description="User who resolves the alert")
    resolution_notes: str = Field(default="", description="Resolution notes")


class AlertFalsePositiveRequest(BaseModel):
    """Mark alert as false positive."""

    marked_by: str = Field(min_length=1, description="User who marks as false positive")
    notes: str = Field(default="", description="Explanation notes")


class AlertStatsResponse(BaseModel):
    """Alert statistics."""

    total_alerts: int
    by_status: dict[str, int]
    by_severity: dict[str, int]
    by_type: dict[str, int]
    active_count: int
    critical_count: int
    recent_24h: int
    avg_resolution_time_hours: float
