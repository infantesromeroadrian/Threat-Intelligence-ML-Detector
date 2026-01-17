"""
IOC (Indicator of Compromise) API models (DTOs).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class IOCResponse(BaseModel):
    """IOC response model."""

    ioc_id: str = Field(description="IOC identifier")
    ioc_type: str = Field(
        description="IOC type (IP_ADDRESS, DOMAIN, URL, EMAIL, FILE_HASH, CVE_ID)"
    )
    value: str = Field(description="IOC value")
    source: str = Field(description="Source of IOC")
    first_seen: str = Field(description="First seen date (ISO format)")
    last_seen: str = Field(description="Last seen date (ISO format)")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    threat_level: str = Field(description="Threat level (CRITICAL, HIGH, MEDIUM, LOW, INFO)")
    is_active: bool = Field(description="Whether IOC is currently active")
    tags: list[str] = Field(default_factory=list, description="Associated tags")
    related_cves: list[str] = Field(default_factory=list, description="Related CVE IDs")
    context: str | None = Field(default=None, description="Context where IOC was found")

    class Config:
        json_schema_extra = {
            "example": {
                "ioc_id": "ioc_20240115_0001",
                "ioc_type": "IP_ADDRESS",
                "value": "192.0.2.1",
                "source": "OTX",
                "first_seen": "2024-01-15T10:00:00Z",
                "last_seen": "2024-01-15T10:00:00Z",
                "confidence_score": 0.95,
                "threat_level": "HIGH",
                "is_active": True,
                "tags": ["malware", "c2"],
                "related_cves": ["CVE-2024-1234"],
                "context": "Observed in ransomware campaign targeting healthcare",
            }
        }


class IOCFilterParams(BaseModel):
    """IOC filter parameters."""

    ioc_type: str | None = Field(default=None, description="Filter by IOC type")
    threat_level: str | None = Field(default=None, description="Filter by threat level")
    min_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum confidence score"
    )
    is_active: bool | None = Field(default=None, description="Filter by active status")
    tag: str | None = Field(default=None, description="Filter by tag")
    source: str | None = Field(default=None, description="Filter by source")
    search: str | None = Field(default=None, description="Search in value or context")


class IOCCreateRequest(BaseModel):
    """IOC creation request."""

    ioc_type: str = Field(description="IOC type")
    value: str = Field(min_length=1, description="IOC value")
    source: str = Field(description="Source of IOC")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score")
    threat_level: str = Field(description="Threat level")
    tags: list[str] = Field(default_factory=list)
    related_cves: list[str] = Field(default_factory=list)
    context: str | None = Field(default=None)


class IOCStatsResponse(BaseModel):
    """IOC statistics."""

    total_iocs: int
    by_type: dict[str, int]
    by_threat_level: dict[str, int]
    active_count: int
    recent_24h: int
