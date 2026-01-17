"""
AlienVault OTX (Open Threat Exchange) scraper adapter.

Scrapes threat intelligence and IOCs from OTX API v1.
Uses real API with proper authentication.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import requests

from ....domain.entities import IOC, ThreatIntel, ThreatSeverity, ThreatType, IOCType, IOCConfidence
from ....infrastructure.config.logging_config import get_logger
from ....infrastructure.config.settings import settings

logger = get_logger(__name__)


class OTXScraper:
    """
    AlienVault OTX threat feed scraper.

    Connects to OTX API v1 to retrieve:
    - Pulses (threat intelligence feeds)
    - IOCs (indicators of compromise)
    - Tags and metadata
    """

    def __init__(self) -> None:
        """Initialize OTX scraper."""
        self.api_key = settings.otx_api_key
        self.base_url = "https://otx.alienvault.com/api/v1"
        self.session = requests.Session()

        if self.api_key:
            self.session.headers.update({"X-OTX-API-KEY": self.api_key})
            logger.info(
                "✅ OTX API key configured - using real API",
                source="OTXScraper",
            )
        else:
            logger.warning(
                "⚠️  OTX API key not configured",
                source="OTXScraper",
            )

        self.session.headers.update(
            {"Content-Type": "application/json", "User-Agent": "ThreatIntelAggregator/0.1.0"}
        )

        # Rate limiting (OTX has generous limits but still need to be respectful)
        self.rate_limit_delay = 0.5

    def scrape_recent(self, hours: int = 24) -> list[ThreatIntel]:
        """
        Scrape recent threat intelligence from the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            List of ThreatIntel entities
        """
        logger.info(
            "🔍 Scraping recent OTX threat intelligence",
            source="OTXScraper",
            hours=hours,
        )

        all_threats: list[ThreatIntel] = []

        try:
            # Get subscribed pulses (most relevant)
            endpoint = f"{self.base_url}/pulses/subscribed"
            params = {
                "limit": 50,
                "modified_since": (datetime.utcnow() - timedelta(hours=hours)).isoformat(),
            }

            time.sleep(self.rate_limit_delay)
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            pulses = data.get("results", [])

            for pulse in pulses:
                threat = self._parse_pulse_to_threat_intel(pulse)
                if threat:
                    all_threats.append(threat)

        except requests.exceptions.RequestException as e:
            logger.error(
                f"❌ Error fetching recent OTX threats: {e}",
                source="OTXScraper",
            )

        logger.info(
            f"✅ Scraped {len(all_threats)} OTX threats",
            source="OTXScraper",
            count=len(all_threats),
        )

        return all_threats

    def scrape_by_threat_type(self, threat_type: str, limit: int = 100) -> list[ThreatIntel]:
        """
        Scrape threat intelligence by type.

        Args:
            threat_type: Type of threat (e.g., "malware", "phishing")
            limit: Maximum number of results

        Returns:
            List of ThreatIntel entities
        """
        logger.info(
            "🔍 Scraping OTX by threat type",
            source="OTXScraper",
            threat_type=threat_type,
            limit=limit,
        )

        all_threats: list[ThreatIntel] = []

        try:
            endpoint = f"{self.base_url}/pulses/subscribed"
            params = {"limit": limit}

            time.sleep(self.rate_limit_delay)
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            pulses = data.get("results", [])

            for pulse in pulses:
                # Filter by tags matching threat type
                tags = pulse.get("tags", [])
                if threat_type.lower() in [tag.lower() for tag in tags]:
                    threat = self._parse_pulse_to_threat_intel(pulse)
                    if threat:
                        all_threats.append(threat)

        except requests.exceptions.RequestException as e:
            logger.error(
                f"❌ Error fetching OTX threats by type: {e}",
                source="OTXScraper",
            )

        logger.info(
            f"✅ Scraped {len(all_threats)} OTX threats for type '{threat_type}'",
            source="OTXScraper",
            count=len(all_threats),
        )

        return all_threats

    def scrape_by_tag(self, tag: str, limit: int = 100) -> list[ThreatIntel]:
        """
        Scrape threat intelligence by tag.

        Args:
            tag: Tag to search for
            limit: Maximum number of results

        Returns:
            List of ThreatIntel entities
        """
        logger.info(
            "🔍 Scraping OTX by tag",
            source="OTXScraper",
            tag=tag,
            limit=limit,
        )

        all_threats: list[ThreatIntel] = []

        try:
            endpoint = f"{self.base_url}/pulses/subscribed"
            params = {"limit": limit}

            time.sleep(self.rate_limit_delay)
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            pulses = data.get("results", [])

            for pulse in pulses:
                tags = pulse.get("tags", [])
                if tag.lower() in [t.lower() for t in tags]:
                    threat = self._parse_pulse_to_threat_intel(pulse)
                    if threat:
                        all_threats.append(threat)

        except requests.exceptions.RequestException as e:
            logger.error(
                f"❌ Error fetching OTX threats by tag: {e}",
                source="OTXScraper",
            )

        logger.info(
            f"✅ Scraped {len(all_threats)} OTX threats for tag '{tag}'",
            source="OTXScraper",
            count=len(all_threats),
        )

        return all_threats

    def scrape_iocs_from_pulse(self, pulse_id: str) -> list[IOC]:
        """
        Extract IOCs from a specific pulse.

        Args:
            pulse_id: Pulse identifier

        Returns:
            List of IOC entities
        """
        logger.info(
            "🔍 Extracting IOCs from pulse",
            source="OTXScraper",
            pulse_id=pulse_id,
        )

        all_iocs: list[IOC] = []

        try:
            endpoint = f"{self.base_url}/pulses/{pulse_id}/indicators"

            time.sleep(self.rate_limit_delay)
            response = self.session.get(endpoint, timeout=30)
            response.raise_for_status()

            data = response.json()
            indicators = data.get("results", [])

            for indicator in indicators:
                ioc = self._parse_indicator_to_ioc(indicator, pulse_id)
                if ioc:
                    all_iocs.append(ioc)

        except requests.exceptions.RequestException as e:
            logger.error(
                f"❌ Error fetching IOCs from pulse {pulse_id}: {e}",
                source="OTXScraper",
            )

        logger.info(
            f"✅ Extracted {len(all_iocs)} IOCs from pulse",
            source="OTXScraper",
            count=len(all_iocs),
        )

        return all_iocs

    def get_pulse_details(self, pulse_id: str) -> dict[str, Any] | None:
        """
        Get detailed information about a specific pulse.

        Args:
            pulse_id: Pulse identifier

        Returns:
            Pulse data dictionary or None if not found
        """
        try:
            endpoint = f"{self.base_url}/pulses/{pulse_id}"

            time.sleep(self.rate_limit_delay)
            response = self.session.get(endpoint, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(
                f"❌ Error fetching pulse {pulse_id}: {e}",
                source="OTXScraper",
            )
            return None

    # =========================================================================
    # Private Methods - Parsing
    # =========================================================================

    def _parse_pulse_to_threat_intel(self, pulse: dict[str, Any]) -> ThreatIntel | None:
        """
        Parse OTX pulse to ThreatIntel entity.

        Args:
            pulse: Pulse data from OTX API

        Returns:
            ThreatIntel entity or None if parsing fails
        """
        try:
            pulse_id = pulse.get("id", "")
            title = pulse.get("name", "Untitled Threat")
            description = pulse.get("description", "No description available")
            created = pulse.get("created", "")
            modified = pulse.get("modified", "")

            # Parse dates
            published_date = self._parse_datetime(created)
            collected_at = datetime.utcnow()

            # Determine threat type from tags
            tags = pulse.get("tags", [])
            threat_type = self._infer_threat_type_from_tags(tags)

            # Determine severity (OTX doesn't have severity, infer from tags)
            severity = self._infer_severity_from_pulse(pulse)

            # Extract IOC count
            indicator_count = pulse.get("indicator_count", 0)

            # Extract related CVEs
            related_cves = []
            references = pulse.get("references", [])
            if isinstance(references, list):
                for ref in references:
                    # Skip if ref is not a dict (could be string or other type)
                    if not isinstance(ref, dict):
                        continue
                    if ref.get("type") == "CVE" and ref.get("value"):
                        related_cves.append(ref.get("value", ""))

            # Build URL
            url = f"https://otx.alienvault.com/pulse/{pulse_id}"

            # Get author
            author_data = pulse.get("author", {})
            if isinstance(author_data, dict):
                author = author_data.get("username", "Unknown")
            elif isinstance(author_data, str):
                author = author_data
            else:
                author = "Unknown"

            return ThreatIntel(
                document_id=f"otx-{pulse_id}",
                title=title,
                content=description,
                threat_type=threat_type,
                severity=severity,
                source="AlienVault OTX",
                published_date=published_date,
                collected_at=collected_at,
                author=author,
                tags=tags,
                related_cves=related_cves,
                iocs_count=indicator_count,
                url=url,
                tlp="WHITE",  # Default TLP for OTX
                raw_data={"pulse_id": pulse_id},  # Store pulse_id for IOC extraction
            )

        except Exception as e:
            logger.error(
                f"❌ Error parsing OTX pulse: {e}",
                source="OTXScraper",
                error=str(e),
            )
            return None

    def _parse_indicator_to_ioc(self, indicator: dict[str, Any], pulse_id: str) -> IOC | None:
        """
        Parse OTX indicator to IOC entity.

        Args:
            indicator: Indicator data from OTX API
            pulse_id: Associated pulse ID

        Returns:
            IOC entity or None if parsing fails
        """
        try:
            ioc_value = indicator.get("indicator", "")
            ioc_type_str = indicator.get("type", "")

            # Map OTX type to our IOCType
            ioc_type = self._map_otx_type_to_ioc_type(ioc_type_str)
            if not ioc_type:
                return None

            # Build context
            context = indicator.get("description", "")
            if not context:
                context = f"{ioc_type_str} indicator from OTX pulse"

            # Parse dates
            created = indicator.get("created", "")
            first_seen = self._parse_datetime(created)
            last_seen = first_seen  # OTX doesn't provide last_seen
            extracted_at = datetime.utcnow()

            # Confidence (OTX indicators are generally high quality)
            confidence = IOCConfidence.HIGH

            return IOC(
                value=ioc_value,
                ioc_type=ioc_type,
                confidence=confidence,
                source_document_id=f"otx-{pulse_id}",
                extracted_at=extracted_at,
                context=context,
                tags=[pulse_id],
                first_seen=first_seen,
                last_seen=last_seen,
                reputation_score=0.7,  # Default high reputation for OTX
            )

        except Exception as e:
            logger.error(
                f"❌ Error parsing OTX indicator: {e}",
                source="OTXScraper",
                error=str(e),
            )
            return None

    def _infer_threat_type_from_tags(self, tags: list[str]) -> ThreatType:
        """Infer threat type from pulse tags."""
        tags_lower = [tag.lower() for tag in tags]

        if any(tag in tags_lower for tag in ["malware", "trojan", "virus", "worm"]):
            return ThreatType.MALWARE
        elif any(tag in tags_lower for tag in ["phishing", "spear phishing", "fraud"]):
            return ThreatType.PHISHING
        elif any(tag in tags_lower for tag in ["apt", "advanced persistent", "espionage"]):
            return ThreatType.APT
        elif any(tag in tags_lower for tag in ["ransomware", "crypto", "locker"]):
            return ThreatType.RANSOMWARE
        elif any(tag in tags_lower for tag in ["exploit", "vulnerability", "cve"]):
            return ThreatType.EXPLOIT
        elif any(tag in tags_lower for tag in ["botnet", "bot", "c2"]):
            return ThreatType.BOTNET
        elif any(tag in tags_lower for tag in ["ddos", "dos", "amplification"]):
            return ThreatType.DDoS
        else:
            return ThreatType.VULNERABILITY  # Default fallback

    def _infer_severity_from_pulse(self, pulse: dict[str, Any]) -> ThreatSeverity:
        """Infer severity from pulse metadata."""
        # Check tags for severity indicators
        tags = pulse.get("tags", [])
        tags_lower = [tag.lower() for tag in tags]

        if any(tag in tags_lower for tag in ["critical", "severe", "high"]):
            return ThreatSeverity.CRITICAL
        elif any(tag in tags_lower for tag in ["important", "moderate"]):
            return ThreatSeverity.HIGH
        elif any(tag in tags_lower for tag in ["medium", "normal"]):
            return ThreatSeverity.MEDIUM
        else:
            # Default based on indicator count
            indicator_count = pulse.get("indicator_count", 0)
            if indicator_count > 100:
                return ThreatSeverity.HIGH
            elif indicator_count > 20:
                return ThreatSeverity.MEDIUM
            else:
                return ThreatSeverity.LOW

    def _map_otx_type_to_ioc_type(self, otx_type: str) -> IOCType | None:
        """Map OTX indicator type to our IOCType."""
        type_mapping = {
            "IPv4": IOCType.IP_ADDRESS,
            "IPv6": IOCType.IP_ADDRESS,
            "domain": IOCType.DOMAIN,
            "hostname": IOCType.DOMAIN,
            "URL": IOCType.URL,
            "MD5": IOCType.FILE_HASH_MD5,
            "SHA1": IOCType.FILE_HASH_SHA1,
            "SHA256": IOCType.FILE_HASH_SHA256,
            "email": IOCType.EMAIL,
            "FilePath": IOCType.FILE_PATH,
        }

        return type_mapping.get(otx_type)

    def _parse_datetime(self, date_str: str) -> datetime:
        """Parse datetime string from OTX API."""
        try:
            # OTX uses ISO 8601 format
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return datetime.utcnow()

    def get_source_name(self) -> str:
        """Get the name of the data source."""
        return "AlienVault OTX"
