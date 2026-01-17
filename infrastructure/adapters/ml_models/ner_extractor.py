"""
spaCy NER IOC Extractor implementation.

Implements IOCExtractorPort using spaCy + custom regex patterns.
Extracts IP addresses, domains, URLs, file hashes, CVE IDs, etc.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from ....domain.entities import IOC, IOCConfidence, IOCType
from ....domain.ports.extractors import IOCExtractorPort
from ...config.logging_config import get_logger
from ...config.settings import settings

logger = get_logger(__name__)


class NERIOCExtractor:
    """
    IOC extractor using spaCy NER + regex patterns.

    Extracts various types of IOCs from text:
    - IP addresses (IPv4/IPv6)
    - Domain names
    - URLs
    - File hashes (MD5, SHA1, SHA256)
    - Email addresses
    - CVE IDs
    """

    # Regex patterns for IOC extraction
    PATTERNS = {
        "ipv4": re.compile(
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
        ),
        "domain": re.compile(
            r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b"
        ),
        "url": re.compile(
            r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}"
            r"\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)"
        ),
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "md5": re.compile(r"\b[a-fA-F0-9]{32}\b"),
        "sha1": re.compile(r"\b[a-fA-F0-9]{40}\b"),
        "sha256": re.compile(r"\b[a-fA-F0-9]{64}\b"),
        "cve": re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE),
    }

    # Known benign domains/IPs to filter out
    BENIGN_DOMAINS = {
        "example.com",
        "example.org",
        "localhost",
        "test.com",
        "domain.com",
    }

    BENIGN_IPS = {
        "127.0.0.1",
        "0.0.0.0",
        "255.255.255.255",
    }

    def __init__(self) -> None:
        """Initialize NER IOC extractor."""
        self.spacy_model_name = settings.spacy_model
        self.nlp: Any = None

        try:
            import spacy

            self.nlp = spacy.load(self.spacy_model_name)
            logger.info(
                f"✅ Loaded spaCy model: {self.spacy_model_name}",
                source="NERIOCExtractor",
            )
        except ImportError:
            logger.warning(
                "⚠️  spaCy not installed - using regex-only extraction",
                source="NERIOCExtractor",
            )
        except OSError:
            logger.warning(
                f"⚠️  spaCy model '{self.spacy_model_name}' not found - using regex-only extraction",
                source="NERIOCExtractor",
            )

    def extract_from_text(self, text: str, source_document_id: str) -> list[IOC]:
        """
        Extract all IOCs from text.

        Args:
            text: Text content to analyze
            source_document_id: ID of the source document

        Returns:
            List of extracted IOC entities
        """
        logger.info(
            f"🔍 Extracting IOCs from text (length: {len(text)})",
            source="NERIOCExtractor",
            document_id=source_document_id,
        )

        iocs: list[IOC] = []

        # Extract different types of IOCs
        iocs.extend(self._extract_ips(text, source_document_id))
        iocs.extend(self._extract_domains(text, source_document_id))
        iocs.extend(self._extract_urls(text, source_document_id))
        iocs.extend(self._extract_emails(text, source_document_id))
        iocs.extend(self._extract_hashes(text, source_document_id))
        iocs.extend(self._extract_cves(text, source_document_id))

        logger.info(
            f"✅ Extracted {len(iocs)} IOCs",
            source="NERIOCExtractor",
            count=len(iocs),
        )

        return iocs

    def extract_ip_addresses(self, text: str) -> list[str]:
        """Extract IP addresses from text."""
        matches = self.PATTERNS["ipv4"].findall(text)
        # Filter benign IPs
        return [ip for ip in matches if ip not in self.BENIGN_IPS]

    def extract_domains(self, text: str) -> list[str]:
        """Extract domain names from text."""
        matches = self.PATTERNS["domain"].findall(text)
        # Filter benign domains and IPs
        domains = []
        for match in matches:
            if match.lower() not in self.BENIGN_DOMAINS:
                # Exclude if it's an IP address
                if not self.PATTERNS["ipv4"].match(match):
                    domains.append(match)
        return domains

    def extract_urls(self, text: str) -> list[str]:
        """Extract URLs from text."""
        return self.PATTERNS["url"].findall(text)

    def extract_file_hashes(self, text: str) -> dict[str, list[str]]:
        """Extract file hashes from text."""
        return {
            "md5": self.PATTERNS["md5"].findall(text),
            "sha1": self.PATTERNS["sha1"].findall(text),
            "sha256": self.PATTERNS["sha256"].findall(text),
        }

    def extract_emails(self, text: str) -> list[str]:
        """Extract email addresses from text."""
        return self.PATTERNS["email"].findall(text)

    def extract_cve_ids(self, text: str) -> list[str]:
        """Extract CVE identifiers from text."""
        matches = self.PATTERNS["cve"].findall(text)
        # Normalize to uppercase
        return [cve.upper() for cve in matches]

    def get_extractor_name(self) -> str:
        """Get the name of the extractor implementation."""
        return "spaCy NER + Regex IOC Extractor"

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _extract_ips(self, text: str, source_id: str) -> list[IOC]:
        """Extract IP address IOCs."""
        ips = self.extract_ip_addresses(text)
        iocs = []

        for ip in ips:
            # Get context (surrounding text)
            context = self._get_context(text, ip)

            ioc = IOC(
                value=ip,
                ioc_type=IOCType.IP_ADDRESS,
                confidence=IOCConfidence.MEDIUM,  # Regex extraction = medium confidence
                source_document_id=source_id,
                extracted_at=datetime.utcnow(),
                context=context,
            )
            iocs.append(ioc)

        return iocs

    def _extract_domains(self, text: str, source_id: str) -> list[IOC]:
        """Extract domain name IOCs."""
        domains = self.extract_domains(text)
        iocs = []

        for domain in domains:
            context = self._get_context(text, domain)

            ioc = IOC(
                value=domain,
                ioc_type=IOCType.DOMAIN,
                confidence=IOCConfidence.MEDIUM,
                source_document_id=source_id,
                extracted_at=datetime.utcnow(),
                context=context,
            )
            iocs.append(ioc)

        return iocs

    def _extract_urls(self, text: str, source_id: str) -> list[IOC]:
        """Extract URL IOCs."""
        urls = self.extract_urls(text)
        iocs = []

        for url in urls:
            context = self._get_context(text, url)

            # Higher confidence for malicious-looking URLs
            confidence = (
                IOCConfidence.HIGH
                if any(
                    keyword in url.lower()
                    for keyword in ["malware", "phishing", "exploit", "payload"]
                )
                else IOCConfidence.MEDIUM
            )

            ioc = IOC(
                value=url,
                ioc_type=IOCType.URL,
                confidence=confidence,
                source_document_id=source_id,
                extracted_at=datetime.utcnow(),
                context=context,
            )
            iocs.append(ioc)

        return iocs

    def _extract_emails(self, text: str, source_id: str) -> list[IOC]:
        """Extract email IOCs."""
        emails = self.extract_emails(text)
        iocs = []

        for email in emails:
            context = self._get_context(text, email)

            ioc = IOC(
                value=email,
                ioc_type=IOCType.EMAIL,
                confidence=IOCConfidence.MEDIUM,
                source_document_id=source_id,
                extracted_at=datetime.utcnow(),
                context=context,
            )
            iocs.append(ioc)

        return iocs

    def _extract_hashes(self, text: str, source_id: str) -> list[IOC]:
        """Extract file hash IOCs."""
        hashes = self.extract_file_hashes(text)
        iocs = []

        # MD5 hashes
        for md5 in hashes["md5"]:
            context = self._get_context(text, md5)
            ioc = IOC(
                value=md5,
                ioc_type=IOCType.FILE_HASH_MD5,
                confidence=IOCConfidence.HIGH,  # Hashes are usually high confidence
                source_document_id=source_id,
                extracted_at=datetime.utcnow(),
                context=context,
            )
            iocs.append(ioc)

        # SHA1 hashes
        for sha1 in hashes["sha1"]:
            context = self._get_context(text, sha1)
            ioc = IOC(
                value=sha1,
                ioc_type=IOCType.FILE_HASH_SHA1,
                confidence=IOCConfidence.HIGH,
                source_document_id=source_id,
                extracted_at=datetime.utcnow(),
                context=context,
            )
            iocs.append(ioc)

        # SHA256 hashes
        for sha256 in hashes["sha256"]:
            context = self._get_context(text, sha256)
            ioc = IOC(
                value=sha256,
                ioc_type=IOCType.FILE_HASH_SHA256,
                confidence=IOCConfidence.HIGH,
                source_document_id=source_id,
                extracted_at=datetime.utcnow(),
                context=context,
            )
            iocs.append(ioc)

        return iocs

    def _extract_cves(self, text: str, source_id: str) -> list[IOC]:
        """Extract CVE ID IOCs."""
        cve_ids = self.extract_cve_ids(text)
        iocs = []

        for cve_id in cve_ids:
            context = self._get_context(text, cve_id)

            ioc = IOC(
                value=cve_id,
                ioc_type=IOCType.CVE_ID,
                confidence=IOCConfidence.HIGH,  # CVE IDs are very reliable
                source_document_id=source_id,
                extracted_at=datetime.utcnow(),
                context=context,
            )
            iocs.append(ioc)

        return iocs

    @staticmethod
    def _get_context(text: str, value: str, window: int = 100) -> str:
        """
        Extract context around a value in text.

        Args:
            text: Full text
            value: Value to find
            window: Characters before/after to include

        Returns:
            Context string
        """
        try:
            idx = text.find(value)
            if idx == -1:
                return ""

            start = max(0, idx - window)
            end = min(len(text), idx + len(value) + window)

            context = text[start:end].strip()

            # Add ellipsis if truncated
            if start > 0:
                context = "..." + context
            if end < len(text):
                context = context + "..."

            return context
        except Exception:
            return ""
