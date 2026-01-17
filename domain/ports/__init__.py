"""Domain ports - Abstract interfaces for infrastructure adapters."""

from .extractors import IOCExtractorPort
from .modelers import SeverityClassifierPort, SimilaritySearchPort, TopicModelerPort
from .notifiers import EmailNotifierPort, NotificationPort, SlackNotifierPort
from .repositories import (
    AlertRepository,
    CVERepository,
    IOCRepository,
    ThreatIntelRepository,
    TopicRepository,
)
from .scrapers import CVEScraperPort, ThreatFeedScraperPort

__all__ = [
    # Repositories
    "CVERepository",
    "IOCRepository",
    "ThreatIntelRepository",
    "TopicRepository",
    "AlertRepository",
    # Scrapers
    "CVEScraperPort",
    "ThreatFeedScraperPort",
    # Extractors
    "IOCExtractorPort",
    # Modelers
    "TopicModelerPort",
    "SeverityClassifierPort",
    "SimilaritySearchPort",
    # Notifiers
    "NotificationPort",
    "SlackNotifierPort",
    "EmailNotifierPort",
]
