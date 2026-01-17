"""
CVE API models (DTOs) for requests and responses.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# =============================================================================
# CVSS Models
# =============================================================================


class CVSSResponse(BaseModel):
    """CVSS score response model."""

    version: str = Field(description="CVSS version (e.g., '3.1')")
    base_score: float = Field(ge=0.0, le=10.0, description="CVSS base score")
    exploitability_score: float | None = Field(default=None, description="Exploitability score")
    impact_score: float | None = Field(default=None, description="Impact score")
    vector_string: str | None = Field(default=None, description="CVSS vector string")

    class Config:
        json_schema_extra = {
            "example": {
                "version": "3.1",
                "base_score": 9.8,
                "exploitability_score": 3.9,
                "impact_score": 5.9,
                "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            }
        }


# =============================================================================
# CVE Response Models
# =============================================================================


class CVEResponse(BaseModel):
    """CVE response model."""

    cve_id: str = Field(description="CVE identifier")
    description: str = Field(description="CVE description")
    published_date: str = Field(description="Publication date (ISO format)")
    last_modified_date: str = Field(description="Last modification date (ISO format)")
    severity: str = Field(description="Severity level (CRITICAL, HIGH, MEDIUM, LOW)")
    cvss: CVSSResponse | None = Field(default=None, description="CVSS scores")
    cwe_ids: list[str] = Field(default_factory=list, description="CWE identifiers")
    references: list[str] = Field(default_factory=list, description="Reference URLs")
    affected_vendors: list[str] = Field(default_factory=list, description="Affected vendors")
    affected_products: list[str] = Field(default_factory=list, description="Affected products")
    source: str = Field(description="Data source (e.g., 'NVD')")

    class Config:
        json_schema_extra = {
            "example": {
                "cve_id": "CVE-2024-1234",
                "description": "Remote code execution vulnerability in Apache HTTP Server",
                "published_date": "2024-01-15T10:00:00Z",
                "last_modified_date": "2024-01-16T14:30:00Z",
                "severity": "CRITICAL",
                "cvss": {
                    "version": "3.1",
                    "base_score": 9.8,
                    "exploitability_score": 3.9,
                    "impact_score": 5.9,
                    "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                },
                "cwe_ids": ["CWE-787"],
                "references": ["https://nvd.nist.gov/vuln/detail/CVE-2024-1234"],
                "affected_vendors": ["Apache"],
                "affected_products": ["HTTP Server"],
                "source": "NVD",
            }
        }


class CVEListResponse(BaseModel):
    """List of CVEs response."""

    items: list[CVEResponse]
    total: int


class CVESummaryResponse(BaseModel):
    """CVE summary statistics."""

    total_cves: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    recent_24h: int
    recent_7d: int


# =============================================================================
# CVE Request Models
# =============================================================================


class CVEFilterParams(BaseModel):
    """CVE filter parameters."""

    severity: str | None = Field(
        default=None, description="Filter by severity (CRITICAL, HIGH, MEDIUM, LOW)"
    )
    keyword: str | None = Field(default=None, description="Search keyword in description")
    vendor: str | None = Field(default=None, description="Filter by affected vendor")
    product: str | None = Field(default=None, description="Filter by affected product")
    cwe_id: str | None = Field(default=None, description="Filter by CWE ID")
    min_cvss: float | None = Field(default=None, ge=0.0, le=10.0, description="Minimum CVSS score")
    start_date: str | None = Field(default=None, description="Start date (ISO format)")
    end_date: str | None = Field(default=None, description="End date (ISO format)")


class CVECreateRequest(BaseModel):
    """CVE creation request (for manual entry)."""

    cve_id: str = Field(description="CVE identifier", pattern=r"^CVE-\d{4}-\d{4,}$")
    description: str = Field(min_length=10, description="CVE description")
    severity: str = Field(description="Severity level")
    cvss_base_score: float | None = Field(default=None, ge=0.0, le=10.0)
    cwe_ids: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    affected_vendors: list[str] = Field(default_factory=list)
    affected_products: list[str] = Field(default_factory=list)
