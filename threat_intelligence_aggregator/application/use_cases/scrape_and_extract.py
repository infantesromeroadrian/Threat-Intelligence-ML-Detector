"""
Scrape CVEs and Extract IOCs Use Case.

End-to-end pipeline:
1. Scrape CVEs from NVD
2. Extract IOCs from CVE descriptions
3. Save CVEs and IOCs to database
4. Generate alerts for critical items
"""

from __future__ import annotations

from datetime import datetime

from ...domain.entities import CVE, IOC
from ...domain.ports.extractors import IOCExtractorPort
from ...domain.ports.repositories import CVERepository, IOCRepository
from ...domain.ports.scrapers import CVEScraperPort
from ...domain.services import AlertService, IOCExtractorService
from ...infrastructure.config.logging_config import get_logger

logger = get_logger(__name__)


class ScrapeAndExtractUseCase:
    """
    Use case for scraping CVEs and extracting IOCs.

    Orchestrates the complete pipeline from scraping to storage.
    """

    def __init__(
        self,
        cve_scraper: CVEScraperPort,
        ioc_extractor: IOCExtractorPort,
        cve_repository: CVERepository,
        ioc_repository: IOCRepository,
    ) -> None:
        """
        Initialize use case with dependencies.

        Args:
            cve_scraper: CVE scraper implementation
            ioc_extractor: IOC extractor implementation
            cve_repository: CVE repository implementation
            ioc_repository: IOC repository implementation
        """
        self.cve_scraper = cve_scraper
        self.ioc_extractor = ioc_extractor
        self.cve_repository = cve_repository
        self.ioc_repository = ioc_repository

        # Domain services
        self.ioc_service = IOCExtractorService()
        self.alert_service = AlertService()

    def execute(self, days: int = 7) -> dict[str, int | list[str]]:
        """
        Execute the complete pipeline.

        Args:
            days: Number of days to look back for CVEs

        Returns:
            Dictionary with execution statistics
        """
        logger.info(
            "🚀 Starting CVE scrape and IOC extraction pipeline",
            source="ScrapeAndExtractUseCase",
            days=days,
        )

        start_time = datetime.utcnow()

        # Step 1: Scrape CVEs
        logger.info("📥 Step 1: Scraping CVEs...")
        cves = self.cve_scraper.scrape_recent(days=days)
        logger.info(f"✅ Scraped {len(cves)} CVEs", count=len(cves))

        # Step 2: Save CVEs to database
        logger.info("💾 Step 2: Saving CVEs to database...")
        self.cve_repository.save_many(cves)
        logger.info(f"✅ Saved {len(cves)} CVEs", count=len(cves))

        # Step 3: Extract IOCs from CVEs
        logger.info("🔍 Step 3: Extracting IOCs from CVEs...")
        all_iocs: list[IOC] = []

        for cve in cves:
            # Extract IOCs from CVE description
            iocs = self.ioc_extractor.extract_from_text(
                text=cve.description, source_document_id=cve.cve_id
            )

            # Link IOCs to CVE
            for ioc in iocs:
                ioc.related_cves.append(cve.cve_id)

            all_iocs.extend(iocs)

        logger.info(f"✅ Extracted {len(all_iocs)} IOCs", count=len(all_iocs))

        # Step 4: Deduplicate IOCs
        logger.info("🔄 Step 4: Deduplicating IOCs...")
        unique_iocs = self.ioc_service.deduplicate_iocs(all_iocs)
        logger.info(f"✅ Deduplicated to {len(unique_iocs)} unique IOCs", count=len(unique_iocs))

        # Step 5: Filter high-confidence IOCs
        logger.info("🎯 Step 5: Filtering high-confidence IOCs...")
        high_conf_iocs = self.ioc_service.filter_high_confidence_iocs(unique_iocs)
        logger.info(
            f"✅ Found {len(high_conf_iocs)} high-confidence IOCs", count=len(high_conf_iocs)
        )

        # Step 6: Save IOCs to database
        logger.info("💾 Step 6: Saving IOCs to database...")
        self.ioc_repository.save_many(unique_iocs)
        logger.info(f"✅ Saved {len(unique_iocs)} IOCs", count=len(unique_iocs))

        # Step 7: Identify critical CVEs for alerts
        logger.info("🚨 Step 7: Identifying items for alerts...")
        critical_cves = [cve for cve in cves if self.alert_service.should_alert_for_cve(cve)]
        alert_iocs = [ioc for ioc in high_conf_iocs if self.alert_service.should_alert_for_ioc(ioc)]

        logger.info(
            f"✅ Found {len(critical_cves)} critical CVEs and {len(alert_iocs)} alert-worthy IOCs",
            critical_cves=len(critical_cves),
            alert_iocs=len(alert_iocs),
        )

        # Calculate statistics
        duration = (datetime.utcnow() - start_time).total_seconds()
        ioc_stats = self.ioc_service.calculate_ioc_statistics(unique_iocs)

        stats = {
            "cves_scraped": len(cves),
            "iocs_extracted": len(all_iocs),
            "iocs_unique": len(unique_iocs),
            "iocs_high_confidence": len(high_conf_iocs),
            "critical_cves": len(critical_cves),
            "alert_worthy_iocs": len(alert_iocs),
            "duration_seconds": duration,
            **ioc_stats,
        }

        logger.info(
            f"🎉 Pipeline completed successfully in {duration:.2f}s",
            source="ScrapeAndExtractUseCase",
            stats=stats,
        )

        return stats
