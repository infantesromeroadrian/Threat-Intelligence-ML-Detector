"""
Extractor ports (interfaces) for information extraction.

These are abstract interfaces that infrastructure adapters must implement.
"""

from __future__ import annotations

from typing import Protocol

from ..entities import IOC


class IOCExtractorPort(Protocol):
    """Interface for IOC extraction from text."""

    def extract_from_text(self, text: str, source_document_id: str) -> list[IOC]:
        """
        Extract IOCs from text content.

        Args:
            text: Text content to analyze
            source_document_id: ID of the source document

        Returns:
            List of extracted IOC entities
        """
        ...

    def extract_ip_addresses(self, text: str) -> list[str]:
        """
        Extract IP addresses from text.

        Args:
            text: Text content to analyze

        Returns:
            List of IP addresses
        """
        ...

    def extract_domains(self, text: str) -> list[str]:
        """
        Extract domain names from text.

        Args:
            text: Text content to analyze

        Returns:
            List of domain names
        """
        ...

    def extract_urls(self, text: str) -> list[str]:
        """
        Extract URLs from text.

        Args:
            text: Text content to analyze

        Returns:
            List of URLs
        """
        ...

    def extract_file_hashes(self, text: str) -> dict[str, list[str]]:
        """
        Extract file hashes from text.

        Args:
            text: Text content to analyze

        Returns:
            Dictionary with hash types as keys and lists of hashes as values
            Example: {"md5": [...], "sha1": [...], "sha256": [...]}
        """
        ...

    def extract_emails(self, text: str) -> list[str]:
        """
        Extract email addresses from text.

        Args:
            text: Text content to analyze

        Returns:
            List of email addresses
        """
        ...

    def extract_cve_ids(self, text: str) -> list[str]:
        """
        Extract CVE identifiers from text.

        Args:
            text: Text content to analyze

        Returns:
            List of CVE IDs (e.g., ["CVE-2024-1234", ...])
        """
        ...

    def get_extractor_name(self) -> str:
        """Get the name of the extractor implementation."""
        ...
