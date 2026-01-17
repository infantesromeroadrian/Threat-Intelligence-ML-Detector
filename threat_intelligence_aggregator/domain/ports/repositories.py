"""
Repository ports (interfaces) for data persistence.

These are abstract interfaces that infrastructure adapters must implement.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Protocol

from ..entities import Alert, CVE, IOC, ThreatIntel, Topic


class CVERepository(Protocol):
    """Repository interface for CVE persistence."""

    def save(self, cve: CVE) -> None:
        """Save a CVE to the repository."""
        ...

    def save_many(self, cves: list[CVE]) -> None:
        """Save multiple CVEs to the repository."""
        ...

    def find_by_id(self, cve_id: str) -> Optional[CVE]:
        """Find a CVE by its ID."""
        ...

    def find_by_severity(self, severity: str) -> list[CVE]:
        """Find CVEs by severity level."""
        ...

    def find_by_date_range(self, start_date: datetime, end_date: datetime) -> list[CVE]:
        """Find CVEs published within a date range."""
        ...

    def find_recent(self, limit: int = 100) -> list[CVE]:
        """Find the most recent CVEs."""
        ...

    def exists(self, cve_id: str) -> bool:
        """Check if a CVE exists in the repository."""
        ...

    def count(self) -> int:
        """Count total CVEs in repository."""
        ...

    def delete(self, cve_id: str) -> None:
        """Delete a CVE from the repository."""
        ...


class IOCRepository(Protocol):
    """Repository interface for IOC persistence."""

    def save(self, ioc: IOC) -> None:
        """Save an IOC to the repository."""
        ...

    def save_many(self, iocs: list[IOC]) -> None:
        """Save multiple IOCs to the repository."""
        ...

    def find_by_value(self, value: str) -> Optional[IOC]:
        """Find an IOC by its value."""
        ...

    def find_by_type(self, ioc_type: str) -> list[IOC]:
        """Find IOCs by type."""
        ...

    def find_by_source(self, source_document_id: str) -> list[IOC]:
        """Find IOCs extracted from a specific document."""
        ...

    def find_by_confidence(self, min_confidence: str) -> list[IOC]:
        """Find IOCs with minimum confidence level."""
        ...

    def find_recent(self, limit: int = 100) -> list[IOC]:
        """Find the most recently extracted IOCs."""
        ...

    def count(self) -> int:
        """Count total IOCs in repository."""
        ...

    def delete(self, value: str, ioc_type: str) -> None:
        """Delete an IOC from the repository."""
        ...


class ThreatIntelRepository(Protocol):
    """Repository interface for threat intelligence documents."""

    def save(self, threat_intel: ThreatIntel) -> None:
        """Save a threat intelligence document."""
        ...

    def save_many(self, threat_intels: list[ThreatIntel]) -> None:
        """Save multiple threat intelligence documents."""
        ...

    def find_by_id(self, document_id: str) -> Optional[ThreatIntel]:
        """Find a threat intelligence document by ID."""
        ...

    def find_by_threat_type(self, threat_type: str) -> list[ThreatIntel]:
        """Find documents by threat type."""
        ...

    def find_by_severity(self, severity: str) -> list[ThreatIntel]:
        """Find documents by severity."""
        ...

    def find_by_source(self, source: str) -> list[ThreatIntel]:
        """Find documents from a specific source."""
        ...

    def find_recent(self, limit: int = 100) -> list[ThreatIntel]:
        """Find the most recently collected documents."""
        ...

    def search_content(self, query: str) -> list[ThreatIntel]:
        """Search documents by content."""
        ...

    def count(self) -> int:
        """Count total documents in repository."""
        ...

    def delete(self, document_id: str) -> None:
        """Delete a document from the repository."""
        ...


class TopicRepository(Protocol):
    """Repository interface for discovered topics."""

    def save(self, topic: Topic) -> None:
        """Save a topic."""
        ...

    def save_many(self, topics: list[Topic]) -> None:
        """Save multiple topics."""
        ...

    def find_by_id(self, topic_id: str) -> Optional[Topic]:
        """Find a topic by ID."""
        ...

    def find_by_number(self, topic_number: int) -> Optional[Topic]:
        """Find a topic by its number."""
        ...

    def find_all(self) -> list[Topic]:
        """Find all topics."""
        ...

    def find_significant(self) -> list[Topic]:
        """Find significant topics (>= 5 docs, coherence > 0.4)."""
        ...

    def find_labeled(self) -> list[Topic]:
        """Find topics that have been manually labeled."""
        ...

    def update_label(self, topic_id: str, label: str) -> None:
        """Update the label for a topic."""
        ...

    def count(self) -> int:
        """Count total topics."""
        ...

    def delete(self, topic_id: str) -> None:
        """Delete a topic."""
        ...


class AlertRepository(Protocol):
    """Repository interface for security alerts."""

    def save(self, alert: Alert) -> None:
        """Save an alert."""
        ...

    def save_many(self, alerts: list[Alert]) -> None:
        """Save multiple alerts."""
        ...

    def find_by_id(self, alert_id: str) -> Optional[Alert]:
        """Find an alert by ID."""
        ...

    def find_by_status(self, status: str) -> list[Alert]:
        """Find alerts by status."""
        ...

    def find_by_severity(self, severity: str) -> list[Alert]:
        """Find alerts by severity."""
        ...

    def find_active(self) -> list[Alert]:
        """Find all active alerts (not resolved/ignored)."""
        ...

    def find_critical(self) -> list[Alert]:
        """Find all critical alerts."""
        ...

    def find_recent(self, limit: int = 100) -> list[Alert]:
        """Find most recent alerts."""
        ...

    def update_status(self, alert_id: str, status: str) -> None:
        """Update alert status."""
        ...

    def count(self) -> int:
        """Count total alerts."""
        ...

    def count_by_status(self, status: str) -> int:
        """Count alerts by status."""
        ...

    def delete(self, alert_id: str) -> None:
        """Delete an alert."""
        ...
