"""
IOC (Indicator of Compromise) domain entity.

Pure Python domain entity with no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class IOCType(str, Enum):
    """Types of Indicators of Compromise."""

    IP_ADDRESS = "IP_ADDRESS"
    DOMAIN = "DOMAIN"
    URL = "URL"
    EMAIL = "EMAIL"
    FILE_HASH_MD5 = "FILE_HASH_MD5"
    FILE_HASH_SHA1 = "FILE_HASH_SHA1"
    FILE_HASH_SHA256 = "FILE_HASH_SHA256"
    FILE_PATH = "FILE_PATH"
    REGISTRY_KEY = "REGISTRY_KEY"
    CVE_ID = "CVE_ID"
    CWE_ID = "CWE_ID"
    MUTEX = "MUTEX"
    USER_AGENT = "USER_AGENT"
    UNKNOWN = "UNKNOWN"


class IOCConfidence(str, Enum):
    """Confidence level for IOC detection."""

    HIGH = "HIGH"  # 90-100%
    MEDIUM = "MEDIUM"  # 70-89%
    LOW = "LOW"  # 50-69%
    UNKNOWN = "UNKNOWN"  # <50%


@dataclass
class IOC:
    """
    IOC (Indicator of Compromise) entity.

    Represents a security indicator extracted from threat intelligence.
    """

    value: str  # The actual IOC value (IP, hash, URL, etc.)
    ioc_type: IOCType
    confidence: IOCConfidence
    source_document_id: str  # Reference to CVE/ThreatIntel document
    extracted_at: datetime
    context: str = ""  # Surrounding text where IOC was found
    tags: list[str] = field(default_factory=list)  # e.g., ["malware", "ransomware"]
    related_cves: list[str] = field(default_factory=list)  # Related CVE IDs
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    reputation_score: Optional[float] = None  # 0.0-1.0 (0=clean, 1=malicious)

    def __post_init__(self) -> None:
        """Validate IOC data."""
        if not self.value:
            raise ValueError("IOC value cannot be empty")
        if self.reputation_score is not None:
            if not 0.0 <= self.reputation_score <= 1.0:
                raise ValueError(
                    f"Reputation score must be between 0.0 and 1.0, got {self.reputation_score}"
                )

    @property
    def is_high_confidence(self) -> bool:
        """Check if IOC has high confidence."""
        return self.confidence == IOCConfidence.HIGH

    @property
    def is_hash(self) -> bool:
        """Check if IOC is a file hash."""
        return self.ioc_type in (
            IOCType.FILE_HASH_MD5,
            IOCType.FILE_HASH_SHA1,
            IOCType.FILE_HASH_SHA256,
        )

    @property
    def is_network_indicator(self) -> bool:
        """Check if IOC is a network indicator."""
        return self.ioc_type in (IOCType.IP_ADDRESS, IOCType.DOMAIN, IOCType.URL)

    @property
    def is_malicious(self) -> bool:
        """Check if IOC is considered malicious based on reputation."""
        return self.reputation_score is not None and self.reputation_score >= 0.7

    def to_dict(self) -> dict[str, object]:
        """Convert IOC to dictionary."""
        return {
            "value": self.value,
            "ioc_type": self.ioc_type.value,
            "confidence": self.confidence.value,
            "source_document_id": self.source_document_id,
            "extracted_at": self.extracted_at.isoformat(),
            "context": self.context,
            "tags": self.tags,
            "related_cves": self.related_cves,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "reputation_score": self.reputation_score,
        }

    def __str__(self) -> str:
        """String representation."""
        return f"[{self.ioc_type.value}] {self.value} (Confidence: {self.confidence.value})"
