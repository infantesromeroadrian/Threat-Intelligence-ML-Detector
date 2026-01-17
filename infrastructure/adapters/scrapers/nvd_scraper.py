"""
NVD CVE Scraper implementation.

Implements CVEScraperPort for scraping CVEs from NVD.
Uses real NVD API v2.0 with proper authentication and rate limiting.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import requests

from ....domain.entities import CVE, CVSS, CVESeverity
from ....domain.ports.scrapers import CVEScraperPort
from ...config.logging_config import get_logger
from ...config.settings import settings

logger = get_logger(__name__)


class NVDScraper:
    """
    NVD CVE scraper implementation using real NVD API v2.0.

    Rate limits:
    - With API key: 50 requests per 30 seconds
    - Without API key: 5 requests per 30 seconds (fallback to mock)
    """

    def __init__(self) -> None:
        """Initialize NVD scraper."""
        self.api_key = settings.nvd_api_key
        self.base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        self.session = requests.Session()

        if self.api_key:
            self.session.headers.update({"apiKey": self.api_key})
            self.rate_limit_delay = 0.6  # 50 requests per 30s = ~0.6s delay
            logger.info(
                "✅ NVD API key configured - using real API",
                source="NVDScraper",
            )
        else:
            logger.warning(
                "⚠️  NVD API key not configured - limited rate (5 req/30s)",
                source="NVDScraper",
            )
            self.rate_limit_delay = 6.0  # 5 requests per 30s = 6s delay

    def scrape_recent(self, days: int = 7) -> list[CVE]:
        """
        Scrape recent CVEs from the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of CVE entities
        """
        logger.info(
            f"📥 Scraping CVEs from last {days} days",
            source="NVDScraper",
            days=days,
        )

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        cves = self.scrape_by_date_range(start_date, end_date)

        logger.info(
            f"✅ Scraped {len(cves)} CVEs",
            source="NVDScraper",
            count=len(cves),
        )

        return cves

    def scrape_by_date_range(self, start_date: datetime, end_date: datetime) -> list[CVE]:
        """
        Scrape CVEs within a date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of CVE entities
        """
        logger.info(
            f"📥 Scraping CVEs from {start_date.date()} to {end_date.date()}",
            source="NVDScraper",
        )

        all_cves: list[CVE] = []
        start_index = 0
        results_per_page = 2000  # NVD max per request

        # Format dates for NVD API (ISO 8601 with timezone)
        # NVD API requires UTC timezone and won't accept future dates
        pub_start_date = start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        pub_end_date = end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        while True:
            # Try with pubStartDate first, fallback to lastModStartDate if 404
            params = {
                "lastModStartDate": pub_start_date,
                "lastModEndDate": pub_end_date,
                "startIndex": start_index,
                "resultsPerPage": results_per_page,
            }

            try:
                # Respect rate limiting
                time.sleep(self.rate_limit_delay)

                response = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()

                data = response.json()

                # Parse CVEs from response
                vulnerabilities = data.get("vulnerabilities", [])
                if not vulnerabilities:
                    break

                for vuln_item in vulnerabilities:
                    cve = self._parse_cve_from_nvd(vuln_item)
                    if cve:
                        all_cves.append(cve)

                # Check if there are more results
                total_results = data.get("totalResults", 0)
                start_index += results_per_page

                if start_index >= total_results:
                    break

                logger.info(
                    f"📊 Progress: {start_index}/{total_results} CVEs",
                    source="NVDScraper",
                )

            except requests.exceptions.RequestException as e:
                logger.error(
                    f"❌ Error fetching CVEs from NVD: {e}",
                    source="NVDScraper",
                )
                break

        logger.info(
            f"✅ Scraped {len(all_cves)} CVEs in date range",
            source="NVDScraper",
            count=len(all_cves),
        )

        return all_cves

    def scrape_by_id(self, cve_id: str) -> CVE | None:
        """
        Scrape a specific CVE by ID.

        Args:
            cve_id: CVE identifier (e.g., "CVE-2024-1234")

        Returns:
            CVE entity or None if not found
        """
        logger.info(
            f"📥 Scraping CVE {cve_id}",
            source="NVDScraper",
            cve_id=cve_id,
        )

        try:
            # Respect rate limiting
            time.sleep(self.rate_limit_delay)

            params = {"cveId": cve_id}
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            vulnerabilities = data.get("vulnerabilities", [])

            if not vulnerabilities:
                logger.warning(
                    f"⚠️  CVE {cve_id} not found",
                    source="NVDScraper",
                )
                return None

            cve = self._parse_cve_from_nvd(vulnerabilities[0])

            logger.info(
                f"✅ Scraped CVE {cve_id}",
                source="NVDScraper",
            )

            return cve

        except requests.exceptions.RequestException as e:
            logger.error(
                f"❌ Error fetching CVE {cve_id}: {e}",
                source="NVDScraper",
            )
            return None

    def scrape_by_keyword(self, keyword: str, limit: int = 100) -> list[CVE]:
        """
        Scrape CVEs matching a keyword.

        Args:
            keyword: Search keyword
            limit: Maximum number of results

        Returns:
            List of CVE entities
        """
        logger.info(
            f"📥 Scraping CVEs with keyword: {keyword}",
            source="NVDScraper",
            keyword=keyword,
            limit=limit,
        )

        all_cves: list[CVE] = []
        start_index = 0
        results_per_page = min(limit, 2000)

        try:
            # Respect rate limiting
            time.sleep(self.rate_limit_delay)

            params = {
                "keywordSearch": keyword,
                "startIndex": start_index,
                "resultsPerPage": results_per_page,
            }

            response = self.session.get(
                self.base_url,
                params=params,
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            vulnerabilities = data.get("vulnerabilities", [])

            for vuln_item in vulnerabilities:
                cve = self._parse_cve_from_nvd(vuln_item)
                if cve:
                    all_cves.append(cve)
                    if len(all_cves) >= limit:
                        break

        except requests.exceptions.RequestException as e:
            logger.error(
                f"❌ Error searching CVEs with keyword '{keyword}': {e}",
                source="NVDScraper",
            )

        logger.info(
            f"✅ Scraped {len(all_cves)} CVEs matching '{keyword}'",
            source="NVDScraper",
            count=len(all_cves),
        )

        return all_cves

    def get_source_name(self) -> str:
        """Get the name of the data source."""
        return "NVD"

    # =========================================================================
    # Private Methods - NVD API Parsing
    # =========================================================================

    def _parse_cve_from_nvd(self, vuln_item: dict[str, Any]) -> CVE | None:
        """
        Parse a CVE from NVD API response.

        Args:
            vuln_item: Vulnerability item from NVD API

        Returns:
            CVE entity or None if parsing fails
        """
        try:
            cve_data = vuln_item.get("cve", {})

            # Extract basic info
            cve_id = cve_data.get("id", "")

            # Extract description (English)
            descriptions = cve_data.get("descriptions", [])
            description = next(
                (d.get("value", "") for d in descriptions if d.get("lang") == "en"),
                "No description available",
            )

            # Extract dates
            published_str = cve_data.get("published", "")
            modified_str = cve_data.get("lastModified", "")

            published_date = self._parse_datetime(published_str)
            last_modified_date = self._parse_datetime(modified_str)

            # Extract CVSS metrics (prefer v3.1, fallback to v3.0, v2)
            cvss, severity = self._extract_cvss_and_severity(cve_data.get("metrics", {}))

            # Extract CWE IDs
            cwe_ids = self._extract_cwe_ids(cve_data.get("weaknesses", []))

            # Extract references
            references = [ref.get("url", "") for ref in cve_data.get("references", [])]

            # Extract affected vendors/products
            affected_vendors, affected_products = self._extract_affected_config(
                cve_data.get("configurations", [])
            )

            return CVE(
                cve_id=cve_id,
                description=description,
                published_date=published_date,
                last_modified_date=last_modified_date,
                severity=severity,
                cvss=cvss,
                cwe_ids=cwe_ids,
                references=references,
                affected_vendors=affected_vendors,
                affected_products=affected_products,
                source="NVD",
            )

        except Exception as e:
            logger.error(
                f"❌ Error parsing CVE from NVD: {e}",
                source="NVDScraper",
                error=str(e),
            )
            return None

    def _extract_cvss_and_severity(self, metrics: dict[str, Any]) -> tuple[CVSS, CVESeverity]:
        """Extract CVSS and severity from metrics."""
        # Try CVSS v3.1
        if "cvssMetricV31" in metrics and metrics["cvssMetricV31"]:
            cvss_data = metrics["cvssMetricV31"][0]["cvssData"]
            base_score = cvss_data.get("baseScore", 0.0)
            vector_string = cvss_data.get("vectorString", "")

            cvss = CVSS(
                version="3.1",
                base_score=base_score,
                vector_string=vector_string,
                exploitability_score=metrics["cvssMetricV31"][0].get("exploitabilityScore"),
                impact_score=metrics["cvssMetricV31"][0].get("impactScore"),
            )
            severity = self._cvss_score_to_severity(base_score)
            return cvss, severity

        # Try CVSS v3.0
        if "cvssMetricV30" in metrics and metrics["cvssMetricV30"]:
            cvss_data = metrics["cvssMetricV30"][0]["cvssData"]
            base_score = cvss_data.get("baseScore", 0.0)
            vector_string = cvss_data.get("vectorString", "")

            cvss = CVSS(
                version="3.0",
                base_score=base_score,
                vector_string=vector_string,
                exploitability_score=metrics["cvssMetricV30"][0].get("exploitabilityScore"),
                impact_score=metrics["cvssMetricV30"][0].get("impactScore"),
            )
            severity = self._cvss_score_to_severity(base_score)
            return cvss, severity

        # Try CVSS v2
        if "cvssMetricV2" in metrics and metrics["cvssMetricV2"]:
            cvss_data = metrics["cvssMetricV2"][0]["cvssData"]
            base_score = cvss_data.get("baseScore", 0.0)
            vector_string = cvss_data.get("vectorString", "")

            cvss = CVSS(
                version="2.0",
                base_score=base_score,
                vector_string=vector_string,
            )
            severity = self._cvss_score_to_severity(base_score)
            return cvss, severity

        # No CVSS data available
        cvss = CVSS(version="3.1", base_score=0.0, vector_string="")
        return cvss, CVESeverity.UNKNOWN

    def _cvss_score_to_severity(self, score: float) -> CVESeverity:
        """Convert CVSS score to severity level."""
        if score >= 9.0:
            return CVESeverity.CRITICAL
        elif score >= 7.0:
            return CVESeverity.HIGH
        elif score >= 4.0:
            return CVESeverity.MEDIUM
        elif score > 0.0:
            return CVESeverity.LOW
        else:
            return CVESeverity.UNKNOWN

    def _extract_cwe_ids(self, weaknesses: list[dict[str, Any]]) -> list[str]:
        """Extract CWE IDs from weaknesses."""
        cwe_ids = []

        for weakness in weaknesses:
            descriptions = weakness.get("description", [])
            for desc in descriptions:
                if desc.get("lang") == "en":
                    value = desc.get("value", "")
                    if value.startswith("CWE-"):
                        cwe_ids.append(value)

        return cwe_ids

    def _extract_affected_config(
        self, configurations: list[dict[str, Any]]
    ) -> tuple[list[str], list[str]]:
        """Extract affected vendors and products from configurations."""
        vendors = set()
        products = set()

        for config in configurations:
            nodes = config.get("nodes", [])
            for node in nodes:
                cpe_matches = node.get("cpeMatch", [])
                for cpe in cpe_matches:
                    criteria = cpe.get("criteria", "")
                    # CPE format: cpe:2.3:a:vendor:product:version:...
                    parts = criteria.split(":")
                    if len(parts) >= 5:
                        vendor = parts[3]
                        product = parts[4]
                        if vendor and vendor != "*":
                            vendors.add(vendor)
                        if product and product != "*":
                            products.add(product)

        return list(vendors), list(products)

    def _parse_datetime(self, date_str: str) -> datetime:
        """Parse datetime string from NVD API."""
        try:
            # NVD uses ISO 8601 format
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return datetime.utcnow()
