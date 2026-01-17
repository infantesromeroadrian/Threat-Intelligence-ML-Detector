"""
Topic domain entity for LDA topic modeling.

Pure Python domain entity with no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class TopicWord:
    """Word in a topic with its probability weight."""

    word: str
    probability: float  # 0.0-1.0

    def __post_init__(self) -> None:
        """Validate topic word."""
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"Probability must be between 0.0 and 1.0, got {self.probability}")


@dataclass
class Topic:
    """
    Topic entity from LDA topic modeling.

    Represents a discovered topic in threat intelligence documents.
    """

    topic_id: str  # e.g., "topic_0", "topic_1"
    topic_number: int  # 0-indexed topic number
    keywords: list[TopicWord]  # Top N words with probabilities
    label: Optional[str] = None  # Human-assigned label (e.g., "Ransomware Attacks")
    coherence_score: Optional[float] = None  # Topic coherence metric
    document_count: int = 0  # Number of documents associated with this topic
    discovery_date: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    related_cves: list[str] = field(default_factory=list)
    related_iocs: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate topic data."""
        if not self.topic_id:
            raise ValueError("Topic ID cannot be empty")
        if self.topic_number < 0:
            raise ValueError(f"Topic number must be >= 0, got {self.topic_number}")
        if not self.keywords:
            raise ValueError("Topic must have at least one keyword")

    @property
    def top_keywords(self) -> list[str]:
        """Get top keyword strings (without probabilities)."""
        return [kw.word for kw in self.keywords]

    @property
    def is_labeled(self) -> bool:
        """Check if topic has been manually labeled."""
        return self.label is not None

    @property
    def has_coherence_score(self) -> bool:
        """Check if topic has coherence score."""
        return self.coherence_score is not None

    @property
    def is_significant(self) -> bool:
        """
        Check if topic is significant.

        A topic is considered significant if it has:
        - At least 5 documents
        - Coherence score > 0.4 (if available)
        """
        has_enough_docs = self.document_count >= 5
        has_good_coherence = self.coherence_score is None or self.coherence_score > 0.4
        return has_enough_docs and has_good_coherence

    def get_keywords_string(self, top_n: int = 10) -> str:
        """
        Get top N keywords as comma-separated string.

        Args:
            top_n: Number of top keywords to include

        Returns:
            Comma-separated keyword string
        """
        keywords = self.top_keywords[:top_n]
        return ", ".join(keywords)

    def to_dict(self) -> dict[str, object]:
        """Convert topic to dictionary."""
        return {
            "topic_id": self.topic_id,
            "topic_number": self.topic_number,
            "keywords": [{"word": kw.word, "probability": kw.probability} for kw in self.keywords],
            "label": self.label,
            "coherence_score": self.coherence_score,
            "document_count": self.document_count,
            "discovery_date": self.discovery_date.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "related_cves": self.related_cves,
            "related_iocs": self.related_iocs,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        """String representation."""
        label_str = f"'{self.label}'" if self.label else f"Topic {self.topic_number}"
        keywords_str = self.get_keywords_string(top_n=5)
        return f"{label_str}: {keywords_str}"
