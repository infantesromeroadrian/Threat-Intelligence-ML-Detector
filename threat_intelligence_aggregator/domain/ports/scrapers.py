"""
Scraper ports (interfaces) for external data collection.

These are abstract interfaces that infrastructure adapters must implement.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..entities import CVE, ThreatIntel


class CVEScraperPort(Protocol):
    """Interface for CVE scrapers (NVD, MITRE, etc.)."""

    def scrape_recent(self, days: int = 7) -> list[CVE]:
        """
        Scrape recent CVEs from the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of CVE entities
        """
        ...

    def scrape_by_date_range(self, start_date: datetime, end_date: datetime) -> list[CVE]:
        """
        Scrape CVEs within a date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of CVE entities
        """
        ...

    def scrape_by_id(self, cve_id: str) -> CVE | None:
        """
        Scrape a specific CVE by ID.

        Args:
            cve_id: CVE identifier (e.g., "CVE-2024-1234")

        Returns:
            CVE entity or None if not found
        """
        ...

    def scrape_by_keyword(self, keyword: str, limit: int = 100) -> list[CVE]:
        """
        Scrape CVEs matching a keyword.

        Args:
            keyword: Search keyword
            limit: Maximum number of results

        Returns:
            List of CVE entities
        """
        ...

    def get_source_name(self) -> str:
        """Get the name of the data source (e.g., 'NVD', 'MITRE')."""
        ...


class ThreatFeedScraperPort(Protocol):
    """Interface for threat intelligence feed scrapers (OTX, etc.)."""

    def scrape_recent(self, hours: int = 24) -> list[ThreatIntel]:
        """
        Scrape recent threat intelligence from the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            List of ThreatIntel entities
        """
        ...

    def scrape_by_threat_type(self, threat_type: str, limit: int = 100) -> list[ThreatIntel]:
        """
        Scrape threat intelligence by type.

        Args:
            threat_type: Type of threat (e.g., "malware", "phishing")
            limit: Maximum number of results

        Returns:
            List of ThreatIntel entities
        """
        ...

    def scrape_by_tag(self, tag: str, limit: int = 100) -> list[ThreatIntel]:
        """
        Scrape threat intelligence by tag.

        Args:
            tag: Tag to search for
            limit: Maximum number of results

        Returns:
            List of ThreatIntel entities
        """
        ...

    def scrape_by_id(self, pulse_id: str) -> ThreatIntel | None:
        """
        Scrape a specific threat intelligence document by ID.

        Args:
            pulse_id: Document/pulse identifier

        Returns:
            ThreatIntel entity or None if not found
        """
        ...

    def get_source_name(self) -> str:
        """Get the name of the data source (e.g., 'OTX', 'MISP')."""
        ...
