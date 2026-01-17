"""
CVE (Common Vulnerabilities and Exposures) domain entity.

Pure Python domain entity with no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class CVESeverity(str, Enum):
    """CVE severity levels based on CVSS score."""

    CRITICAL = "CRITICAL"  # CVSS 9.0-10.0
    HIGH = "HIGH"  # CVSS 7.0-8.9
    MEDIUM = "MEDIUM"  # CVSS 4.0-6.9
    LOW = "LOW"  # CVSS 0.1-3.9
    NONE = "NONE"  # CVSS 0.0
    UNKNOWN = "UNKNOWN"  # No CVSS score available


@dataclass(frozen=True)
class CVSS:
    """CVSS (Common Vulnerability Scoring System) score."""

    version: str  # e.g., "3.1", "2.0"
    base_score: float  # 0.0-10.0
    vector_string: str  # e.g., "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    exploitability_score: Optional[float] = None
    impact_score: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate CVSS score."""
        if not 0.0 <= self.base_score <= 10.0:
            raise ValueError(f"CVSS base_score must be between 0.0 and 10.0, got {self.base_score}")


@dataclass
class CVE:
    """
    CVE (Common Vulnerabilities and Exposures) entity.

    Represents a security vulnerability from the NVD database.
    """

    cve_id: str  # e.g., "CVE-2024-1234"
    description: str
    published_date: datetime
    last_modified_date: datetime
    severity: CVESeverity
    cvss: Optional[CVSS] = None
    cwe_ids: list[str] = field(default_factory=list)  # e.g., ["CWE-79", "CWE-89"]
    references: list[str] = field(default_factory=list)  # URLs
    affected_vendors: list[str] = field(default_factory=list)
    affected_products: list[str] = field(default_factory=list)
    source: str = "NVD"  # Data source (NVD, MITRE, etc.)
    raw_data: Optional[dict[str, object]] = None  # Original JSON data

    def __post_init__(self) -> None:
        """Validate CVE data."""
        if not self.cve_id.startswith("CVE-"):
            raise ValueError(f"Invalid CVE ID format: {self.cve_id}")
        if not self.description:
            raise ValueError("CVE description cannot be empty")

    @property
    def is_critical(self) -> bool:
        """Check if CVE is critical severity."""
        return self.severity == CVESeverity.CRITICAL

    @property
    def is_high_or_critical(self) -> bool:
        """Check if CVE is high or critical severity."""
        return self.severity in (CVESeverity.CRITICAL, CVESeverity.HIGH)

    @property
    def cvss_score(self) -> Optional[float]:
        """Get CVSS base score if available."""
        return self.cvss.base_score if self.cvss else None

    @property
    def year(self) -> int:
        """Extract year from CVE ID."""
        try:
            return int(self.cve_id.split("-")[1])
        except (IndexError, ValueError):
            return 0

    def to_dict(self) -> dict[str, object]:
        """Convert CVE to dictionary."""
        return {
            "cve_id": self.cve_id,
            "description": self.description,
            "published_date": self.published_date.isoformat(),
            "last_modified_date": self.last_modified_date.isoformat(),
            "severity": self.severity.value,
            "cvss_score": self.cvss_score,
            "cwe_ids": self.cwe_ids,
            "references": self.references,
            "affected_vendors": self.affected_vendors,
            "affected_products": self.affected_products,
            "source": self.source,
        }

    def __str__(self) -> str:
        """String representation."""
        cvss_str = f"CVSS {self.cvss_score}" if self.cvss_score else "No CVSS"
        return f"{self.cve_id} [{self.severity.value}] - {cvss_str}"
