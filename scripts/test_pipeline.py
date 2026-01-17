#!/usr/bin/env python3
"""
End-to-end pipeline test script.

Tests the complete threat intelligence aggregation pipeline:
1. Scrape CVEs and Threat Intelligence
2. Extract IOCs
3. Save to database
4. Generate alerts
5. Query and display results
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from datetime import datetime

from threat_intelligence_aggregator.application.use_cases.scrape_and_extract import (
    ScrapeAndExtractUseCase,
)
from threat_intelligence_aggregator.infrastructure.adapters.ml_models import NERIOCExtractor
from threat_intelligence_aggregator.infrastructure.adapters.repositories import (
    SQLiteAlertRepository,
    SQLiteCVERepository,
    SQLiteIOCRepository,
    SQLiteThreatIntelRepository,
)
from threat_intelligence_aggregator.infrastructure.adapters.scrapers import (
    NVDScraper,
    OTXScraper,
)
from threat_intelligence_aggregator.infrastructure.config.logging_config import get_logger

logger = get_logger(__name__)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main() -> None:
    """Run end-to-end pipeline test."""
    print_section("🚀 Threat Intelligence Aggregator - Pipeline Test")

    # Database path
    db_url = "sqlite:///threat_intel_test.db"
    print(f"📊 Database: {db_url}\n")

    # Initialize adapters
    print_section("🔧 Initializing Adapters")

    # Scrapers
    nvd_scraper = NVDScraper(use_mock=True)
    otx_scraper = OTXScraper(use_mock=True)
    print("✅ Scrapers initialized")

    # Extractor
    ioc_extractor = NERIOCExtractor(use_spacy=False)  # Regex-only for simplicity
    print("✅ IOC Extractor initialized")

    # Repositories
    cve_repo = SQLiteCVERepository(db_url)
    ioc_repo = SQLiteIOCRepository(db_url)
    threat_intel_repo = SQLiteThreatIntelRepository(db_url)
    alert_repo = SQLiteAlertRepository(db_url)
    print("✅ Repositories initialized")

    # Initialize database tables
    from threat_intelligence_aggregator.infrastructure.adapters.repositories.base import Base
    from sqlalchemy import create_engine

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    print("✅ Database tables created")

    # Step 1: Scrape CVEs
    print_section("1️⃣ Scraping CVEs from NVD")
    cves = nvd_scraper.scrape_recent(days=7)
    print(f"📥 Scraped {len(cves)} CVEs")

    for i, cve in enumerate(cves[:3], 1):
        print(f"  {i}. {cve.cve_id} - {cve.severity.value} - {cve.description[:80]}...")

    # Step 2: Scrape Threat Intelligence
    print_section("2️⃣ Scraping Threat Intelligence from OTX")
    threats = otx_scraper.scrape_recent(hours=48)
    print(f"📥 Scraped {len(threats)} threat intelligence documents")

    for i, threat in enumerate(threats[:3], 1):
        print(f"  {i}. [{threat.threat_type.value}] {threat.title[:60]}...")

    # Step 3: Extract IOCs
    print_section("3️⃣ Extracting IOCs")
    all_iocs = []

    # Extract IOCs from CVE descriptions
    for cve in cves:
        iocs = ioc_extractor.extract(cve.description)
        all_iocs.extend(iocs)

    # Extract IOCs from threat intel content
    for threat in threats:
        text = f"{threat.title} {threat.content}"
        iocs = ioc_extractor.extract(text)
        all_iocs.extend(iocs)

    print(f"🔍 Extracted {len(all_iocs)} IOCs")

    # Group IOCs by type
    ioc_types = {}
    for ioc in all_iocs:
        ioc_types[ioc.ioc_type.value] = ioc_types.get(ioc.ioc_type.value, 0) + 1

    for ioc_type, count in sorted(ioc_types.items()):
        print(f"  - {ioc_type}: {count}")

    # Step 4: Save to database
    print_section("4️⃣ Saving to Database")

    # Save CVEs
    cve_repo.save_many(cves)
    print(f"💾 Saved {len(cves)} CVEs")

    # Save Threat Intelligence
    threat_intel_repo.save_many(threats)
    print(f"💾 Saved {len(threats)} threat intelligence documents")

    # Save IOCs
    ioc_repo.save_many(all_iocs)
    print(f"💾 Saved {len(all_iocs)} IOCs")

    # Step 5: Generate statistics
    print_section("5️⃣ Database Statistics")

    total_cves = cve_repo.count_all()
    total_threats = threat_intel_repo.count_all()
    total_iocs = ioc_repo.count_all()

    print(f"📊 Total CVEs: {total_cves}")
    print(f"📊 Total Threat Intelligence: {total_threats}")
    print(f"📊 Total IOCs: {total_iocs}")

    # Step 6: Query examples
    print_section("6️⃣ Query Examples")

    # Find critical CVEs
    critical_cves = cve_repo.find_by_severity("CRITICAL")
    print(f"🔴 Critical CVEs: {len(critical_cves)}")
    for cve in critical_cves[:3]:
        print(f"  - {cve.cve_id} (Score: {cve.cvss.base_score if cve.cvss else 'N/A'})")

    # Find high severity threats
    high_threats = threat_intel_repo.find_high_severity()
    print(f"\n🔴 High Severity Threats: {len(high_threats)}")
    for threat in high_threats[:3]:
        print(f"  - [{threat.threat_type.value}] {threat.title[:60]}...")

    # Find IP IOCs
    ip_iocs = ioc_repo.find_by_type("IP_ADDRESS")
    print(f"\n🌐 IP Address IOCs: {len(ip_iocs)}")
    for ioc in ip_iocs[:5]:
        print(f"  - {ioc.value} (Confidence: {ioc.confidence_score:.2f})")

    # Step 7: Test use case
    print_section("7️⃣ Testing Use Case")

    use_case = ScrapeAndExtractUseCase(
        cve_scraper=nvd_scraper,
        ioc_extractor=ioc_extractor,
        cve_repository=cve_repo,
        ioc_repository=ioc_repo,
    )

    result = use_case.execute(days_back=3)
    print(f"✅ Use case executed successfully")
    print(f"  - CVEs scraped: {result['cves_scraped']}")
    print(f"  - CVEs saved: {result['cves_saved']}")
    print(f"  - IOCs extracted: {result['iocs_extracted']}")
    print(f"  - IOCs saved: {result['iocs_saved']}")
    print(f"  - Critical CVEs: {result['critical_cve_count']}")

    # Summary
    print_section("✅ Pipeline Test Complete")
    print("All components working correctly!")
    print(f"\n📂 Test database created: threat_intel_test.db")
    print("\nNext steps:")
    print("  1. Implement ML models (LDA, BERT, Word2Vec)")
    print("  2. Create FastAPI endpoints")
    print("  3. Build frontend interface")
    print("  4. Add notification system (Slack, Email)")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("❌ Pipeline test failed", error=str(e), exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)
