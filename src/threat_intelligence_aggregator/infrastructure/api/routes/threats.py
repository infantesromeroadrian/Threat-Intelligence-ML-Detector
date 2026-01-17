"""
Threat Intelligence API routes.

Endpoints for managing threat intelligence documents.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ....domain.entities import ThreatIntel
from ...adapters.repositories import SQLiteThreatIntelRepository
from ...config.logging_config import get_logger
from ...config.settings import settings
from ..models import (
    PaginatedResponse,
    PaginationParams,
    ThreatIntelFilterParams,
    ThreatIntelResponse,
    ThreatIntelStatsResponse,
)

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# Dependency Injection
# =============================================================================


def get_threat_intel_repository() -> SQLiteThreatIntelRepository:
    """Get ThreatIntel repository instance."""
    return SQLiteThreatIntelRepository(db_url=settings.database_url)


ThreatIntelRepositoryDep = Annotated[
    SQLiteThreatIntelRepository, Depends(get_threat_intel_repository)
]


# =============================================================================
# Helper Functions
# =============================================================================


def threat_intel_to_response(threat: ThreatIntel) -> ThreatIntelResponse:
    """Convert ThreatIntel domain entity to API response model."""
    return ThreatIntelResponse(
        document_id=threat.document_id,
        title=threat.title,
        content=threat.content,
        threat_type=threat.threat_type.value,
        severity=threat.severity.value,
        source=threat.source,
        published_date=threat.published_date.isoformat(),
        collected_at=threat.collected_at.isoformat(),
        author=threat.author,
        tags=threat.tags,
        related_cves=threat.related_cves,
        iocs_count=threat.iocs_count,
        url=threat.url,
        tlp=threat.tlp,
        confidence_score=threat.confidence_score,
        word_count=threat.word_count,
    )


# =============================================================================
# Routes
# =============================================================================


@router.get("/", response_model=PaginatedResponse[ThreatIntelResponse])
async def list_threats(
    repo: ThreatIntelRepositoryDep,
    pagination: Annotated[PaginationParams, Depends()],
    filters: Annotated[ThreatIntelFilterParams, Depends()],
) -> PaginatedResponse[ThreatIntelResponse]:
    """List threat intelligence documents with filtering and pagination."""
    logger.info("📋 Listing threat intelligence", source="ThreatsRoutes")

    # Apply filters
    threats: list[ThreatIntel] = []

    if filters.threat_type:
        threats = repo.find_by_threat_type(filters.threat_type)
    elif filters.severity:
        threats = repo.find_by_severity(filters.severity)
    elif filters.source:
        threats = repo.find_by_source(filters.source)
    elif filters.keyword:
        threats = repo.search_by_keywords([filters.keyword])
    else:
        threats = repo.find_recent(limit=1000)

    # Additional filtering
    if filters.min_confidence:
        threats = [
            t
            for t in threats
            if t.confidence_score and t.confidence_score >= filters.min_confidence
        ]

    if filters.tag:
        threats = [t for t in threats if filters.tag.lower() in [tag.lower() for tag in t.tags]]

    # Pagination
    total = len(threats)
    paginated_threats = threats[pagination.skip : pagination.skip + pagination.limit]

    # Convert to response models
    items = [threat_intel_to_response(t) for t in paginated_threats]

    return PaginatedResponse.create(
        items=items, total=total, skip=pagination.skip, limit=pagination.limit
    )


@router.get("/stats", response_model=ThreatIntelStatsResponse)
async def get_threat_intel_stats(repo: ThreatIntelRepositoryDep) -> ThreatIntelStatsResponse:
    """Get threat intelligence statistics."""
    logger.info("📊 Getting threat intelligence statistics", source="ThreatsRoutes")

    total_documents = repo.count_all()

    # Get all threats for grouping (simplified)
    all_threats = repo.find_recent(limit=10000)

    # Group by threat type
    by_threat_type: dict[str, int] = {}
    for threat in all_threats:
        threat_type = threat.threat_type.value
        by_threat_type[threat_type] = by_threat_type.get(threat_type, 0) + 1

    # Group by severity
    by_severity: dict[str, int] = {}
    for threat in all_threats:
        severity = threat.severity.value
        by_severity[severity] = by_severity.get(severity, 0) + 1

    # Group by source
    by_source: dict[str, int] = {}
    for threat in all_threats:
        by_source[threat.source] = by_source.get(threat.source, 0) + 1

    # Recent counts
    now = datetime.utcnow()
    recent_24h = sum(1 for t in all_threats if (now - t.published_date).days == 0)
    recent_7d = sum(1 for t in all_threats if (now - t.published_date).days <= 7)

    # Average IOCs
    avg_iocs = sum(t.iocs_count for t in all_threats) / len(all_threats) if all_threats else 0.0

    return ThreatIntelStatsResponse(
        total_documents=total_documents,
        by_threat_type=by_threat_type,
        by_severity=by_severity,
        by_source=by_source,
        recent_24h=recent_24h,
        recent_7d=recent_7d,
        avg_iocs_per_document=round(avg_iocs, 2),
    )


@router.get("/recent", response_model=list[ThreatIntelResponse])
async def get_recent_threats(
    repo: ThreatIntelRepositoryDep,
    limit: int = Query(default=50, ge=1, le=500, description="Number of recent threats to return"),
) -> list[ThreatIntelResponse]:
    """Get most recent threat intelligence documents."""
    logger.info("🕒 Getting recent threats", source="ThreatsRoutes", limit=limit)

    threats = repo.find_recent(limit=limit)
    return [threat_intel_to_response(t) for t in threats]


@router.get("/high-severity", response_model=list[ThreatIntelResponse])
async def get_high_severity_threats(repo: ThreatIntelRepositoryDep) -> list[ThreatIntelResponse]:
    """Get high and critical severity threats."""
    logger.info("🔴 Getting high severity threats", source="ThreatsRoutes")

    threats = repo.find_high_severity()
    return [threat_intel_to_response(t) for t in threats]


@router.get("/{document_id}", response_model=ThreatIntelResponse)
async def get_threat_by_id(document_id: str, repo: ThreatIntelRepositoryDep) -> ThreatIntelResponse:
    """Get a specific threat intelligence document by ID."""
    logger.info("🔍 Getting threat by ID", source="ThreatsRoutes", document_id=document_id)

    threat = repo.find_by_id(document_id)

    if not threat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Threat intelligence document {document_id} not found",
        )

    return threat_intel_to_response(threat)
