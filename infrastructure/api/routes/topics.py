"""
Topic API routes.

Endpoints for topic modeling and discovery.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ....domain.entities import Topic
from ...adapters.repositories import SQLiteTopicRepository
from ...config.logging_config import get_logger
from ...config.settings import settings
from ..models import (
    PaginatedResponse,
    PaginationParams,
    SuccessResponse,
    TopicKeywordResponse,
    TopicResponse,
    TopicStatsResponse,
    TopicUpdateRequest,
)

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# Dependency Injection
# =============================================================================


def get_topic_repository() -> SQLiteTopicRepository:
    """Get Topic repository instance."""
    return SQLiteTopicRepository(db_url=settings.database_url)


TopicRepositoryDep = Annotated[SQLiteTopicRepository, Depends(get_topic_repository)]


# =============================================================================
# Helper Functions
# =============================================================================


def topic_to_response(topic: Topic) -> TopicResponse:
    """Convert Topic domain entity to API response model."""
    keywords = [
        TopicKeywordResponse(word=kw.word, probability=kw.probability) for kw in topic.keywords
    ]

    return TopicResponse(
        topic_id=topic.topic_id,
        topic_number=topic.topic_number,
        keywords=keywords,
        label=topic.label,
        coherence_score=topic.coherence_score,
        document_count=topic.document_count,
        discovery_date=topic.discovery_date.isoformat(),
        last_updated=topic.last_updated.isoformat(),
        related_cves=topic.related_cves,
        related_iocs=topic.related_iocs,
        is_significant=topic.is_significant,
    )


# =============================================================================
# Routes
# =============================================================================


@router.get("/", response_model=PaginatedResponse[TopicResponse])
async def list_topics(
    repo: TopicRepositoryDep,
    pagination: Annotated[PaginationParams, Depends()],
) -> PaginatedResponse[TopicResponse]:
    """List all discovered topics."""
    logger.info("📋 Listing topics", source="TopicsRoutes")

    topics = repo.find_all()

    # Pagination
    total = len(topics)
    paginated_topics = topics[pagination.skip : pagination.skip + pagination.limit]

    # Convert to response models
    items = [topic_to_response(t) for t in paginated_topics]

    return PaginatedResponse.create(
        items=items, total=total, skip=pagination.skip, limit=pagination.limit
    )


@router.get("/stats", response_model=TopicStatsResponse)
async def get_topic_stats(repo: TopicRepositoryDep) -> TopicStatsResponse:
    """Get topic statistics."""
    logger.info("📊 Getting topic statistics", source="TopicsRoutes")

    total_topics = repo.count_all()
    labeled_topics = len(repo.find_labeled())
    significant_topics = len(repo.find_significant())

    all_topics = repo.find_all()
    avg_coherence = (
        sum(t.coherence_score for t in all_topics if t.coherence_score) / len(all_topics)
        if all_topics
        else 0.0
    )
    avg_documents = (
        sum(t.document_count for t in all_topics) / len(all_topics) if all_topics else 0.0
    )

    return TopicStatsResponse(
        total_topics=total_topics,
        labeled_topics=labeled_topics,
        significant_topics=significant_topics,
        avg_coherence_score=round(avg_coherence, 3),
        avg_documents_per_topic=round(avg_documents, 1),
    )


@router.get("/significant", response_model=list[TopicResponse])
async def get_significant_topics(repo: TopicRepositoryDep) -> list[TopicResponse]:
    """Get statistically significant topics."""
    logger.info("⭐ Getting significant topics", source="TopicsRoutes")

    topics = repo.find_significant()
    return [topic_to_response(t) for t in topics]


@router.get("/{topic_id}", response_model=TopicResponse)
async def get_topic_by_id(topic_id: str, repo: TopicRepositoryDep) -> TopicResponse:
    """Get a specific topic by ID."""
    logger.info("🔍 Getting topic by ID", source="TopicsRoutes", topic_id=topic_id)

    topic = repo.find_by_id(topic_id)

    if not topic:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Topic {topic_id} not found",
        )

    return topic_to_response(topic)


@router.put("/{topic_id}/label", response_model=SuccessResponse)
async def update_topic_label(
    topic_id: str, request: TopicUpdateRequest, repo: TopicRepositoryDep
) -> SuccessResponse:
    """Update a topic's human-assigned label."""
    logger.info("🏷️ Updating topic label", source="TopicsRoutes", topic_id=topic_id)

    # Check if topic exists
    topic = repo.find_by_id(topic_id)
    if not topic:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Topic {topic_id} not found",
        )

    # Update label
    repo.update_label(topic_id, request.label)

    return SuccessResponse(
        message=f"Topic {topic_id} label updated",
        data={"topic_id": topic_id, "label": request.label},
    )
