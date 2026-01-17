"""Scraper adapters for external data sources."""

from .nvd_scraper import NVDScraper
from .otx_scraper import OTXScraper

__all__ = [
    "NVDScraper",
    "OTXScraper",
]
