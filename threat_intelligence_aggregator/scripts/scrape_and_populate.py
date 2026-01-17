#!/usr/bin/env python3
"""
Scrape threat intelligence and populate database.

This script:
1. Scrapes CVEs from NVD API
2. Scrapes threats and IOCs from AlienVault OTX
3. Saves data to SQLite database
4. Generates alerts for critical threats
5. Sends notifications via Slack/Email (if configured)

Usage:
    python scripts/scrape_and_populate.py [--days 7] [--hours 24] [--notify]

Examples:
    # Scrape last 7 days of CVEs and 24 hours of threats
    python scripts/scrape_and_populate.py

    # Scrape last 30 days with notifications
    python scripts/scrape_and_populate.py --days 30 --notify

    # Quick test (1 day, no notifications)
    python scripts/scrape_and_populate.py --days 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from threat_intelligence_aggregator.domain.entities.alert import (
    Alert,
    AlertSeverity,
    AlertStatus,
    AlertType,
)
from threat_intelligence_aggregator.domain.entities.cve import CVE
from threat_intelligence_aggregator.domain.entities.ioc import IOC
from threat_intelligence_aggregator.domain.entities.threat_intel import ThreatIntel
from threat_intelligence_aggregator.infrastructure.adapters.notifiers.email_notifier import (
    EmailNotifier,
)
from threat_intelligence_aggregator.infrastructure.adapters.notifiers.slack_notifier import (
    SlackNotifier,
)
from threat_intelligence_aggregator.infrastructure.adapters.repositories.sqlite_alert_repo import (
    SQLiteAlertRepository,
)
from threat_intelligence_aggregator.infrastructure.adapters.repositories.sqlite_cve_repo import (
    SQLiteCVERepository,
)
from threat_intelligence_aggregator.infrastructure.adapters.repositories.sqlite_ioc_repo import (
    SQLiteIOCRepository,
)
from threat_intelligence_aggregator.infrastructure.adapters.repositories.sqlite_threat_intel_repo import (
    SQLiteThreatIntelRepository,
)
from threat_intelligence_aggregator.infrastructure.adapters.scrapers.nvd_scraper import (
    NVDScraper,
)
from threat_intelligence_aggregator.infrastructure.adapters.scrapers.otx_scraper import (
    OTXScraper,
)
from threat_intelligence_aggregator.infrastructure.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ScraperStats:
    """Statistics for scraping session."""

    def __init__(self) -> None:
        self.cves_scraped: int = 0
        self.cves_saved: int = 0
        self.threats_scraped: int = 0
        self.threats_saved: int = 0
        self.iocs_scraped: int = 0
        self.iocs_saved: int = 0
        self.alerts_generated: int = 0
        self.alerts_by_severity: dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }
        self.errors: list[str] = []

    def add_error(self, error: str) -> None:
        """Add error to list."""
        self.errors.append(error)
        logger.error("❌ %s", error)

    def increment_alert_severity(self, severity: AlertSeverity) -> None:
        """Increment alert count by severity."""
        self.alerts_by_severity[severity.value] += 1

    def print_summary(self) -> None:
        """Print scraping statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 SCRAPING SUMMARY")
        logger.info("=" * 60)
        logger.info("CVEs:        %d scraped, %d saved", self.cves_scraped, self.cves_saved)
        logger.info("Threats:     %d scraped, %d saved", self.threats_scraped, self.threats_saved)
        logger.info("IOCs:        %d scraped, %d saved", self.iocs_scraped, self.iocs_saved)
        logger.info("Alerts:      %d generated", self.alerts_generated)

        if self.alerts_generated > 0:
            logger.info("  - 🔴 Critical: %d", self.alerts_by_severity["critical"])
            logger.info("  - 🟠 High:     %d", self.alerts_by_severity["high"])
            logger.info("  - 🟡 Medium:   %d", self.alerts_by_severity["medium"])
            logger.info("  - 🔵 Low:      %d", self.alerts_by_severity["low"])
            logger.info("  - ⚪ Info:     %d", self.alerts_by_severity["info"])

        if self.errors:
            logger.info("\n⚠️  Errors encountered: %d", len(self.errors))
            for error in self.errors[:5]:  # Show first 5 errors
                logger.info("  - %s", error)
            if len(self.errors) > 5:
                logger.info("  ... and %d more errors", len(self.errors) - 5)

        logger.info("=" * 60)


def scrape_nvd_cves(
    scraper: NVDScraper,
    cve_repo: SQLiteCVERepository,
    alert_repo: SQLiteAlertRepository,
    stats: ScraperStats,
    days: int = 7,
) -> list[Alert]:
    """
    Scrape CVEs from NVD and save to database.

    Args:
        scraper: NVD scraper instance
        cve_repo: CVE repository
        alert_repo: Alert repository
        stats: Statistics tracker
        days: Number of days to scrape

    Returns:
        List of generated alerts
    """
    logger.info("📥 Scraping CVEs from last %d days...", days)
    alerts: list[Alert] = []

    try:
        # Scrape CVEs
        cves = scraper.scrape_recent(days=days)
        stats.cves_scraped = len(cves)
        logger.info("✅ Scraped %d CVEs", len(cves))

        if not cves:
            logger.warning("⚠️  No CVEs found")
            return alerts

        # Save CVEs to database
        for cve in cves:
            try:
                saved_cve = cve_repo.save(cve)
                stats.cves_saved += 1

                # Generate alert for critical CVEs (CVSS >= 9.0)
                if saved_cve.cvss and saved_cve.cvss.base_score >= 9.0:
                    alert = Alert(
                        id=f"alert-cve-{saved_cve.id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        alert_type=AlertType.CVE_CRITICAL,
                        severity=AlertSeverity.CRITICAL,
                        title=f"Critical CVE: {saved_cve.id}",
                        description=saved_cve.description[:500],
                        source=f"NVD: {saved_cve.id}",
                        timestamp=datetime.now(),
                        status=AlertStatus.ACTIVE,
                        related_entities={
                            "cve_id": saved_cve.id,
                            "cvss_score": saved_cve.cvss.base_score,
                            "severity": saved_cve.severity.value,
                        },
                        metadata={
                            "affected_vendors": saved_cve.affected_vendors,
                            "affected_products": saved_cve.affected_products,
                            "published_date": saved_cve.published_date.isoformat(),
                        },
                    )
                    alert_repo.save(alert)
                    alerts.append(alert)
                    stats.alerts_generated += 1
                    stats.increment_alert_severity(AlertSeverity.CRITICAL)

                # Generate alert for high severity CVEs (CVSS >= 7.0)
                elif saved_cve.cvss and saved_cve.cvss.base_score >= 7.0:
                    alert = Alert(
                        id=f"alert-cve-{saved_cve.id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        alert_type=AlertType.CVE_HIGH,
                        severity=AlertSeverity.HIGH,
                        title=f"High Severity CVE: {saved_cve.id}",
                        description=saved_cve.description[:500],
                        source=f"NVD: {saved_cve.id}",
                        timestamp=datetime.now(),
                        status=AlertStatus.ACTIVE,
                        related_entities={
                            "cve_id": saved_cve.id,
                            "cvss_score": saved_cve.cvss.base_score,
                            "severity": saved_cve.severity.value,
                        },
                        metadata={
                            "affected_vendors": saved_cve.affected_vendors,
                            "affected_products": saved_cve.affected_products,
                            "published_date": saved_cve.published_date.isoformat(),
                        },
                    )
                    alert_repo.save(alert)
                    alerts.append(alert)
                    stats.alerts_generated += 1
                    stats.increment_alert_severity(AlertSeverity.HIGH)

            except Exception as e:
                stats.add_error(f"Failed to save CVE {cve.id}: {e}")
                continue

        logger.info("💾 Saved %d/%d CVEs to database", stats.cves_saved, stats.cves_scraped)
        logger.info("🚨 Generated %d alerts for high/critical CVEs", len(alerts))

    except Exception as e:
        stats.add_error(f"Failed to scrape NVD: {e}")

    return alerts


def scrape_otx_threats(
    scraper: OTXScraper,
    threat_repo: SQLiteThreatIntelRepository,
    ioc_repo: SQLiteIOCRepository,
    alert_repo: SQLiteAlertRepository,
    stats: ScraperStats,
    hours: int = 24,
) -> list[Alert]:
    """
    Scrape threats and IOCs from AlienVault OTX.

    Args:
        scraper: OTX scraper instance
        threat_repo: Threat intelligence repository
        ioc_repo: IOC repository
        alert_repo: Alert repository
        stats: Statistics tracker
        hours: Number of hours to scrape

    Returns:
        List of generated alerts
    """
    logger.info("📥 Scraping OTX threats from last %d hours...", hours)
    alerts: list[Alert] = []

    try:
        # Scrape threats
        threats = scraper.scrape_recent(hours=hours)
        stats.threats_scraped = len(threats)
        logger.info("✅ Scraped %d threats", len(threats))

        if not threats:
            logger.warning("⚠️  No threats found")
            return alerts

        # Save threats and extract IOCs
        for threat in threats:
            try:
                # Save threat
                saved_threat = threat_repo.save(threat)
                stats.threats_saved += 1

                # Extract IOCs from threat raw_data
                if threat.raw_data and "pulse_id" in threat.raw_data:
                    try:
                        iocs = scraper.scrape_iocs_from_pulse(threat.raw_data["pulse_id"])
                        stats.iocs_scraped += len(iocs)

                        # Save IOCs
                        for ioc in iocs:
                            try:
                                ioc_repo.save(ioc)
                                stats.iocs_saved += 1
                            except Exception as e:
                                stats.add_error(f"Failed to save IOC {ioc.value}: {e}")
                                continue

                    except Exception as e:
                        stats.add_error(
                            f"Failed to scrape IOCs for pulse {threat.metadata['pulse_id']}: {e}"
                        )

                # Generate alert for high/critical threats
                if saved_threat.severity.value in ["critical", "high"]:
                    alert = Alert(
                        id=f"alert-threat-{saved_threat.document_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        alert_type=AlertType.THREAT_DETECTED,
                        severity=(
                            AlertSeverity.CRITICAL
                            if saved_threat.severity.value == "critical"
                            else AlertSeverity.HIGH
                        ),
                        title=f"{saved_threat.severity.value.upper()} Threat: {saved_threat.title[:50]}",
                        description=saved_threat.content[:500],
                        source=f"OTX: {saved_threat.source}",
                        timestamp=datetime.now(),
                        status=AlertStatus.ACTIVE,
                        related_entities={
                            "threat_id": saved_threat.document_id,
                            "threat_type": saved_threat.threat_type.value,
                            "severity": saved_threat.severity.value,
                        },
                        metadata={
                            "tags": saved_threat.tags,
                            "ioc_count": saved_threat.iocs_count,
                            "published_date": saved_threat.published_date.isoformat(),
                        },
                    )
                    alert_repo.save(alert)
                    alerts.append(alert)
                    stats.alerts_generated += 1
                    stats.increment_alert_severity(alert.severity)

            except Exception as e:
                stats.add_error(f"Failed to save threat {threat.document_id}: {e}")
                continue

        logger.info(
            "💾 Saved %d/%d threats to database", stats.threats_saved, stats.threats_scraped
        )
        logger.info("💾 Saved %d/%d IOCs to database", stats.iocs_saved, stats.iocs_scraped)
        logger.info("🚨 Generated %d alerts for high/critical threats", len(alerts))

    except Exception as e:
        stats.add_error(f"Failed to scrape OTX: {e}")

    return alerts


def send_notifications(alerts: list[Alert], stats: ScraperStats) -> None:
    """
    Send notifications via Slack and Email.

    Args:
        alerts: List of alerts to notify
        stats: Statistics tracker
    """
    if not alerts:
        logger.info("ℹ️  No alerts to notify")
        return

    logger.info("📧 Sending notifications for %d alerts...", len(alerts))

    # Try Slack notification
    try:
        if settings.slack_webhook_url:
            slack = SlackNotifier(settings.slack_webhook_url)
            slack.send_summary(alerts)
            logger.info("✅ Slack notification sent")
        else:
            logger.info("ℹ️  Slack webhook not configured, skipping")
    except Exception as e:
        stats.add_error(f"Failed to send Slack notification: {e}")

    # Try Email notification
    try:
        if all(
            [
                settings.smtp_host,
                settings.smtp_port,
                settings.smtp_user,
                settings.smtp_password,
                settings.smtp_from,
                settings.smtp_to,
            ]
        ):
            email = EmailNotifier(
                smtp_host=settings.smtp_host,
                smtp_port=settings.smtp_port,
                smtp_user=settings.smtp_user,
                smtp_password=settings.smtp_password,
                from_email=settings.smtp_from,
                to_emails=settings.smtp_to.split(",") if settings.smtp_to else [],
            )
            email.send_summary(alerts)
            logger.info("✅ Email notification sent")
        else:
            logger.info("ℹ️  SMTP not configured, skipping email")
    except Exception as e:
        stats.add_error(f"Failed to send email notification: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scrape threat intelligence and populate database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape last 7 days of CVEs and 24 hours of threats
  %(prog)s

  # Scrape last 30 days with notifications
  %(prog)s --days 30 --notify

  # Quick test (1 day, no notifications)
  %(prog)s --days 1
        """,
    )

    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to scrape CVEs from NVD (default: 7)",
    )

    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Number of hours to scrape threats from OTX (default: 24)",
    )

    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send notifications via Slack/Email",
    )

    parser.add_argument(
        "--skip-nvd",
        action="store_true",
        help="Skip NVD scraping (only scrape OTX)",
    )

    parser.add_argument(
        "--skip-otx",
        action="store_true",
        help="Skip OTX scraping (only scrape NVD)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    logger.info("🚀 Starting threat intelligence scraping...")
    logger.info("Configuration:")
    logger.info("  - CVE days: %d", args.days)
    logger.info("  - OTX hours: %d", args.hours)
    logger.info("  - Notifications: %s", "enabled" if args.notify else "disabled")
    logger.info("  - NVD API Key: %s", "configured ✅" if settings.nvd_api_key else "missing ❌")
    logger.info("  - OTX API Key: %s", "configured ✅" if settings.otx_api_key else "missing ❌")

    # Initialize statistics
    stats = ScraperStats()

    # Initialize repositories
    logger.info("\n📦 Initializing repositories...")
    cve_repo = SQLiteCVERepository(settings.database_url)
    threat_repo = SQLiteThreatIntelRepository(settings.database_url)
    ioc_repo = SQLiteIOCRepository(settings.database_url)
    alert_repo = SQLiteAlertRepository(settings.database_url)

    # Initialize scrapers
    logger.info("🔧 Initializing scrapers...")
    nvd_scraper = NVDScraper()
    otx_scraper = OTXScraper()

    all_alerts: list[Alert] = []

    # Scrape NVD CVEs
    if not args.skip_nvd:
        logger.info("\n" + "=" * 60)
        nvd_alerts = scrape_nvd_cves(nvd_scraper, cve_repo, alert_repo, stats, days=args.days)
        all_alerts.extend(nvd_alerts)
    else:
        logger.info("\n⏭️  Skipping NVD scraping")

    # Scrape OTX threats
    if not args.skip_otx:
        logger.info("\n" + "=" * 60)
        otx_alerts = scrape_otx_threats(
            otx_scraper, threat_repo, ioc_repo, alert_repo, stats, hours=args.hours
        )
        all_alerts.extend(otx_alerts)
    else:
        logger.info("\n⏭️  Skipping OTX scraping")

    # Send notifications
    if args.notify and all_alerts:
        logger.info("\n" + "=" * 60)
        send_notifications(all_alerts, stats)

    # Print summary
    stats.print_summary()

    # Return exit code
    if stats.errors:
        logger.warning("\n⚠️  Scraping completed with errors")
        return 1

    logger.info("\n✅ Scraping completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
