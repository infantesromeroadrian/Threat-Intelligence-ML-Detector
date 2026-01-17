"""
ThreatIntel (Threat Intelligence Document) domain entity.

Pure Python domain entity with no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ThreatType(str, Enum):
    """Types of threat intelligence."""

    MALWARE = "MALWARE"
    RANSOMWARE = "RANSOMWARE"
    PHISHING = "PHISHING"
    APT = "APT"  # Advanced Persistent Threat
    BOTNET = "BOTNET"
    EXPLOIT = "EXPLOIT"
    VULNERABILITY = "VULNERABILITY"
    DDoS = "DDoS"
    DATA_BREACH = "DATA_BREACH"
    INSIDER_THREAT = "INSIDER_THREAT"
    UNKNOWN = "UNKNOWN"


class ThreatSeverity(str, Enum):
    """Severity levels for threat intelligence."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class ThreatIntel:
    """
    Threat Intelligence document entity.

    Represents a threat intelligence report from various sources.
    """

    document_id: str  # Unique identifier
    title: str
    content: str  # Full text content
    threat_type: ThreatType
    severity: ThreatSeverity
    source: str  # e.g., "OTX", "MITRE", "Manual Analysis"
    published_date: datetime
    collected_at: datetime
    author: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    related_cves: list[str] = field(default_factory=list)
    iocs_count: int = 0  # Number of IOCs extracted from this document
    url: Optional[str] = None  # Original URL
    tlp: str = "WHITE"  # Traffic Light Protocol (WHITE, GREEN, AMBER, RED)
    confidence_score: Optional[float] = None  # 0.0-1.0
    raw_data: Optional[dict[str, object]] = None

    def __post_init__(self) -> None:
        """Validate threat intel data."""
        if not self.document_id:
            raise ValueError("Document ID cannot be empty")
        if not self.title:
            raise ValueError("Title cannot be empty")
        if not self.content:
            raise ValueError("Content cannot be empty")
        if self.confidence_score is not None:
            if not 0.0 <= self.confidence_score <= 1.0:
                raise ValueError(
                    f"Confidence score must be between 0.0 and 1.0, got {self.confidence_score}"
                )

    @property
    def is_high_severity(self) -> bool:
        """Check if threat is high or critical severity."""
        return self.severity in (ThreatSeverity.CRITICAL, ThreatSeverity.HIGH)

    @property
    def is_critical(self) -> bool:
        """Check if threat is critical."""
        return self.severity == ThreatSeverity.CRITICAL

    @property
    def has_iocs(self) -> bool:
        """Check if document has extracted IOCs."""
        return self.iocs_count > 0

    @property
    def is_restricted(self) -> bool:
        """Check if document has sharing restrictions (TLP RED/AMBER)."""
        return self.tlp in ("RED", "AMBER")

    @property
    def word_count(self) -> int:
        """Get word count of content."""
        return len(self.content.split())

    def to_dict(self) -> dict[str, object]:
        """Convert threat intel to dictionary."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "content": self.content,
            "threat_type": self.threat_type.value,
            "severity": self.severity.value,
            "source": self.source,
            "published_date": self.published_date.isoformat(),
            "collected_at": self.collected_at.isoformat(),
            "author": self.author,
            "tags": self.tags,
            "related_cves": self.related_cves,
            "iocs_count": self.iocs_count,
            "url": self.url,
            "tlp": self.tlp,
            "confidence_score": self.confidence_score,
            "word_count": self.word_count,
        }

    def __str__(self) -> str:
        """String representation."""
        return f"[{self.threat_type.value}] {self.title} ({self.severity.value})"
