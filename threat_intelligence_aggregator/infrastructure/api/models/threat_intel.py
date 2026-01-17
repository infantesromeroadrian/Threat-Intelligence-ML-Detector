"""
Threat Intelligence API models (DTOs).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ThreatIntelResponse(BaseModel):
    """Threat intelligence document response."""

    document_id: str = Field(description="Document identifier")
    title: str = Field(description="Document title")
    content: str = Field(description="Full text content")
    threat_type: str = Field(
        description="Threat type (MALWARE, RANSOMWARE, PHISHING, APT, BOTNET, etc.)"
    )
    severity: str = Field(description="Severity level (CRITICAL, HIGH, MEDIUM, LOW, INFO)")
    source: str = Field(description="Source (e.g., 'OTX', 'MITRE')")
    published_date: str = Field(description="Publication date (ISO format)")
    collected_at: str = Field(description="Collection date (ISO format)")
    author: str | None = Field(default=None, description="Document author")
    tags: list[str] = Field(default_factory=list, description="Tags")
    related_cves: list[str] = Field(default_factory=list, description="Related CVE IDs")
    iocs_count: int = Field(description="Number of extracted IOCs")
    url: str | None = Field(default=None, description="Original URL")
    tlp: str = Field(description="Traffic Light Protocol (WHITE, GREEN, AMBER, RED)")
    confidence_score: float | None = Field(default=None, description="Confidence score (0.0-1.0)")
    word_count: int = Field(description="Word count")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "otx_pulse_20240115_0001",
                "title": "Ransomware Campaign Targeting Healthcare Sector",
                "content": "A sophisticated ransomware campaign has been observed...",
                "threat_type": "RANSOMWARE",
                "severity": "CRITICAL",
                "source": "AlienVault OTX",
                "published_date": "2024-01-15T10:00:00Z",
                "collected_at": "2024-01-15T11:00:00Z",
                "author": "AlienVault Labs",
                "tags": ["ransomware", "healthcare", "encryption"],
                "related_cves": ["CVE-2024-1234"],
                "iocs_count": 25,
                "url": "https://otx.alienvault.com/pulse/abc123",
                "tlp": "WHITE",
                "confidence_score": 0.95,
                "word_count": 350,
            }
        }


class ThreatIntelFilterParams(BaseModel):
    """Threat intelligence filter parameters."""

    threat_type: str | None = Field(default=None, description="Filter by threat type")
    severity: str | None = Field(default=None, description="Filter by severity")
    source: str | None = Field(default=None, description="Filter by source")
    tag: str | None = Field(default=None, description="Filter by tag")
    keyword: str | None = Field(default=None, description="Search keyword in title/content")
    min_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum confidence score"
    )
    start_date: str | None = Field(default=None, description="Start date (ISO format)")
    end_date: str | None = Field(default=None, description="End date (ISO format)")


class ThreatIntelStatsResponse(BaseModel):
    """Threat intelligence statistics."""

    total_documents: int
    by_threat_type: dict[str, int]
    by_severity: dict[str, int]
    by_source: dict[str, int]
    recent_24h: int
    recent_7d: int
    avg_iocs_per_document: float
