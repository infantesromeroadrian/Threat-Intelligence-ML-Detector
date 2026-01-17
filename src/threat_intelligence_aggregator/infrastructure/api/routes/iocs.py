"""
IOC (Indicator of Compromise) API routes.

Endpoints for managing IOCs extracted from threat intelligence.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ....domain.entities import IOC, IOCConfidence, IOCType
from ...adapters.repositories import SQLiteIOCRepository
from ...config.logging_config import get_logger
from ...config.settings import settings
from ..models import (
    IOCCreateRequest,
    IOCFilterParams,
    IOCResponse,
    IOCStatsResponse,
    PaginatedResponse,
    PaginationParams,
    SuccessResponse,
)

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# Dependency Injection
# =============================================================================


def get_ioc_repository() -> SQLiteIOCRepository:
    """Get IOC repository instance."""
    return SQLiteIOCRepository(db_url=settings.database_url)


IOCRepositoryDep = Annotated[SQLiteIOCRepository, Depends(get_ioc_repository)]


# =============================================================================
# Helper Functions
# =============================================================================


def ioc_to_response(ioc: IOC) -> IOCResponse:
    """Convert IOC domain entity to API response model."""
    # Generate ID from value + type
    ioc_id = f"{ioc.ioc_type.value}_{hash(ioc.value) % 100000}"

    return IOCResponse(
        ioc_id=ioc_id,
        ioc_type=ioc.ioc_type.value,
        value=ioc.value,
        source=ioc.source_document_id,
        first_seen=ioc.first_seen.isoformat() if ioc.first_seen else ioc.extracted_at.isoformat(),
        last_seen=ioc.last_seen.isoformat() if ioc.last_seen else ioc.extracted_at.isoformat(),
        confidence_score=ioc.reputation_score or 0.5,
        threat_level="HIGH" if ioc.is_malicious else "MEDIUM",
        is_active=True,  # Default to active
        tags=ioc.tags,
        related_cves=ioc.related_cves,
        context=ioc.context,
    )


# =============================================================================
# Routes
# =============================================================================


@router.get("/", response_model=PaginatedResponse[IOCResponse])
async def list_iocs(
    repo: IOCRepositoryDep,
    pagination: Annotated[PaginationParams, Depends()],
    filters: Annotated[IOCFilterParams, Depends()],
) -> PaginatedResponse[IOCResponse]:
    """
    List IOCs with optional filtering and pagination.

    Supports filtering by type, threat level, confidence, active status, tag, and source.
    """
    logger.info("📋 Listing IOCs", source="IOCRoutes", skip=pagination.skip, limit=pagination.limit)

    # Apply filters
    iocs: list[IOC] = []

    if filters.ioc_type:
        iocs = repo.find_by_type(filters.ioc_type)
    elif filters.source:
        iocs = repo.find_by_source(filters.source)
    elif filters.min_confidence:
        iocs = repo.find_by_confidence(str(filters.min_confidence))
    else:
        iocs = repo.find_recent(limit=1000)

    # Additional filtering
    if filters.threat_level:
        # Filter by maliciousness (simplified)
        if filters.threat_level in ("CRITICAL", "HIGH"):
            iocs = [ioc for ioc in iocs if ioc.is_malicious]

    if filters.is_active is not None:
        # All IOCs considered active for now (simplification)
        pass

    if filters.tag:
        iocs = [ioc for ioc in iocs if filters.tag.lower() in [t.lower() for t in ioc.tags]]

    if filters.search:
        search_lower = filters.search.lower()
        iocs = [
            ioc
            for ioc in iocs
            if search_lower in ioc.value.lower()
            or (ioc.context and search_lower in ioc.context.lower())
        ]

    # Pagination
    total = len(iocs)
    paginated_iocs = iocs[pagination.skip : pagination.skip + pagination.limit]

    # Convert to response models
    items = [ioc_to_response(ioc) for ioc in paginated_iocs]

    return PaginatedResponse.create(
        items=items, total=total, skip=pagination.skip, limit=pagination.limit
    )


@router.get("/stats", response_model=IOCStatsResponse)
async def get_ioc_stats(repo: IOCRepositoryDep) -> IOCStatsResponse:
    """Get IOC statistics."""
    logger.info("📊 Getting IOC statistics", source="IOCRoutes")

    # Get all IOCs (consider limiting for large datasets)
    all_iocs = repo.find_recent(limit=10000)

    total_iocs = len(all_iocs)

    # Count by type
    by_type: dict[str, int] = {}
    for ioc in all_iocs:
        ioc_type = ioc.ioc_type.value
        by_type[ioc_type] = by_type.get(ioc_type, 0) + 1

    # Count by threat level (based on maliciousness)
    by_threat_level: dict[str, int] = {
        "HIGH": sum(1 for ioc in all_iocs if ioc.is_malicious),
        "MEDIUM": sum(1 for ioc in all_iocs if not ioc.is_malicious),
    }

    # Active count (all IOCs considered active)
    active_count = total_iocs

    # Recent 24h
    now = datetime.utcnow()
    cutoff = now - timedelta(hours=24)
    recent_24h = sum(1 for ioc in all_iocs if ioc.extracted_at >= cutoff)

    return IOCStatsResponse(
        total_iocs=total_iocs,
        by_type=by_type,
        by_threat_level=by_threat_level,
        active_count=active_count,
        recent_24h=recent_24h,
    )


@router.get("/recent", response_model=list[IOCResponse])
async def get_recent_iocs(
    repo: IOCRepositoryDep,
    limit: int = Query(default=50, ge=1, le=500, description="Number of recent IOCs to return"),
) -> list[IOCResponse]:
    """Get most recent IOCs."""
    logger.info("🕒 Getting recent IOCs", source="IOCRoutes", limit=limit)

    iocs = repo.find_recent(limit=limit)
    return [ioc_to_response(ioc) for ioc in iocs]


@router.get("/active", response_model=list[IOCResponse])
async def get_active_iocs(repo: IOCRepositoryDep) -> list[IOCResponse]:
    """Get active IOCs."""
    logger.info("✅ Getting active IOCs", source="IOCRoutes")

    iocs = repo.find_recent(limit=10000)
    # All IOCs considered active (simplification)
    return [ioc_to_response(ioc) for ioc in iocs]


@router.get("/type/{ioc_type}", response_model=list[IOCResponse])
async def get_iocs_by_type(ioc_type: str, repo: IOCRepositoryDep) -> list[IOCResponse]:
    """Get IOCs by type."""
    logger.info("🔍 Getting IOCs by type", source="IOCRoutes", ioc_type=ioc_type)

    iocs = repo.find_by_type(ioc_type)
    return [ioc_to_response(ioc) for ioc in iocs]


@router.get("/{value}", response_model=IOCResponse)
async def get_ioc_by_value(value: str, repo: IOCRepositoryDep) -> IOCResponse:
    """
    Get a specific IOC by its value.

    Returns 404 if IOC not found.
    """
    logger.info("🔍 Getting IOC by value", source="IOCRoutes", value=value)

    ioc = repo.find_by_value(value)

    if not ioc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"IOC with value '{value}' not found",
        )

    return ioc_to_response(ioc)


@router.post("/", response_model=SuccessResponse, status_code=status.HTTP_201_CREATED)
async def create_ioc(request: IOCCreateRequest, repo: IOCRepositoryDep) -> SuccessResponse:
    """
    Create a new IOC (manual entry).

    Useful for adding IOCs from manual analysis.
    """
    logger.info("➕ Creating IOC", source="IOCRoutes", value=request.value)

    # Check if IOC already exists
    existing = repo.find_by_value(request.value)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"IOC with value '{request.value}' already exists",
        )

    # Map strings to enums
    try:
        ioc_type = IOCType(request.ioc_type)
        confidence = IOCConfidence.HIGH if request.confidence_score >= 0.8 else IOCConfidence.MEDIUM
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Create IOC entity
    now = datetime.utcnow()
    ioc = IOC(
        value=request.value,
        ioc_type=ioc_type,
        confidence=confidence,
        source_document_id=request.source,
        extracted_at=now,
        context=request.context or "",
        tags=request.tags,
        related_cves=request.related_cves,
        first_seen=now,
        last_seen=now,
        reputation_score=request.confidence_score,
    )

    # Save to repository
    repo.save(ioc)

    return SuccessResponse(
        message=f"IOC '{request.value}' created successfully",
        data={"value": request.value, "type": request.ioc_type},
    )
