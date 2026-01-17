"""
Topic API models (DTOs) for topic modeling.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TopicKeywordResponse(BaseModel):
    """Topic keyword with probability."""

    word: str = Field(description="Keyword")
    probability: float = Field(ge=0.0, le=1.0, description="Probability weight")


class TopicResponse(BaseModel):
    """Topic response model."""

    topic_id: str = Field(description="Topic identifier")
    topic_number: int = Field(description="Topic number (0-indexed)")
    keywords: list[TopicKeywordResponse] = Field(description="Top keywords with probabilities")
    label: str | None = Field(default=None, description="Human-assigned label")
    coherence_score: float | None = Field(default=None, description="Topic coherence metric")
    document_count: int = Field(description="Number of associated documents")
    discovery_date: str = Field(description="Discovery date (ISO format)")
    last_updated: str = Field(description="Last update date (ISO format)")
    related_cves: list[str] = Field(default_factory=list, description="Related CVE IDs")
    related_iocs: list[str] = Field(default_factory=list, description="Related IOC IDs")
    is_significant: bool = Field(description="Whether topic is statistically significant")

    class Config:
        json_schema_extra = {
            "example": {
                "topic_id": "topic_0",
                "topic_number": 0,
                "keywords": [
                    {"word": "ransomware", "probability": 0.08},
                    {"word": "encryption", "probability": 0.06},
                    {"word": "malware", "probability": 0.05},
                ],
                "label": "Ransomware Attacks",
                "coherence_score": 0.65,
                "document_count": 25,
                "discovery_date": "2024-01-15T10:00:00Z",
                "last_updated": "2024-01-15T10:00:00Z",
                "related_cves": ["CVE-2024-1234"],
                "related_iocs": [],
                "is_significant": True,
            }
        }


class TopicDiscoveryRequest(BaseModel):
    """Topic discovery request."""

    num_topics: int = Field(default=10, ge=2, le=50, description="Number of topics to discover")
    min_documents: int = Field(
        default=5, ge=1, description="Minimum documents required for training"
    )
    hours_back: int = Field(
        default=168, ge=1, description="Hours to look back for documents (default: 7 days)"
    )


class TopicUpdateRequest(BaseModel):
    """Topic update request."""

    label: str = Field(min_length=1, max_length=200, description="New topic label")


class TopicStatsResponse(BaseModel):
    """Topic statistics."""

    total_topics: int
    labeled_topics: int
    significant_topics: int
    avg_coherence_score: float
    avg_documents_per_topic: float
