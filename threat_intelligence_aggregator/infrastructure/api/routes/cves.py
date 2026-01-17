"""
CVE API routes.

Endpoints for managing CVEs (Common Vulnerabilities and Exposures).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ....domain.entities import CVE, CVSS, CVESeverity
from ...adapters.repositories import SQLiteCVERepository
from ...config.logging_config import get_logger
from ...config.settings import settings
from ..models import (
    CVECreateRequest,
    CVEFilterParams,
    CVEResponse,
    CVESummaryResponse,
    CVSSResponse,
    PaginatedResponse,
    PaginationParams,
    SuccessResponse,
)

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# Dependency Injection
# =============================================================================


def get_cve_repository() -> SQLiteCVERepository:
    """Get CVE repository instance."""
    return SQLiteCVERepository(db_url=settings.database_url)


CVERepositoryDep = Annotated[SQLiteCVERepository, Depends(get_cve_repository)]


# =============================================================================
# Helper Functions
# =============================================================================


def cve_to_response(cve: CVE) -> CVEResponse:
    """Convert CVE domain entity to API response model."""
    cvss_response = None
    if cve.cvss:
        cvss_response = CVSSResponse(
            version=cve.cvss.version,
            base_score=cve.cvss.base_score,
            exploitability_score=cve.cvss.exploitability_score,
            impact_score=cve.cvss.impact_score,
            vector_string=cve.cvss.vector_string,
        )

    return CVEResponse(
        cve_id=cve.cve_id,
        description=cve.description,
        published_date=cve.published_date.isoformat(),
        last_modified_date=cve.last_modified_date.isoformat(),
        severity=cve.severity.value,
        cvss=cvss_response,
        cwe_ids=cve.cwe_ids,
        references=cve.references,
        affected_vendors=cve.affected_vendors,
        affected_products=cve.affected_products,
        source=cve.source,
    )


# =============================================================================
# Routes
# =============================================================================


@router.get("/", response_model=PaginatedResponse[CVEResponse])
async def list_cves(
    repo: CVERepositoryDep,
    pagination: Annotated[PaginationParams, Depends()],
    filters: Annotated[CVEFilterParams, Depends()],
) -> PaginatedResponse[CVEResponse]:
    """
    List CVEs with optional filtering and pagination.

    Supports filtering by severity, keyword, vendor, product, CWE ID, CVSS score, and date range.
    """
    logger.info("📋 Listing CVEs", source="CVERoutes", skip=pagination.skip, limit=pagination.limit)

    # Apply filters
    cves: list[CVE] = []

    if filters.severity:
        cves = repo.find_by_severity(filters.severity)
    elif filters.start_date and filters.end_date:
        start = datetime.fromisoformat(filters.start_date)
        end = datetime.fromisoformat(filters.end_date)
        cves = repo.find_by_date_range(start, end)
    elif filters.keyword:
        cves = repo.search_by_keyword(filters.keyword)
    else:
        cves = repo.find_recent(limit=1000)  # Get all recent

    # Additional filtering
    if filters.min_cvss:
        cves = [cve for cve in cves if cve.cvss and cve.cvss.base_score >= filters.min_cvss]

    if filters.vendor:
        cves = [
            cve
            for cve in cves
            if any(filters.vendor.lower() in v.lower() for v in cve.affected_vendors)
        ]

    if filters.product:
        cves = [
            cve
            for cve in cves
            if any(filters.product.lower() in p.lower() for p in cve.affected_products)
        ]

    if filters.cwe_id:
        cves = [cve for cve in cves if filters.cwe_id in cve.cwe_ids]

    # Pagination
    total = len(cves)
    paginated_cves = cves[pagination.skip : pagination.skip + pagination.limit]

    # Convert to response models
    items = [cve_to_response(cve) for cve in paginated_cves]

    return PaginatedResponse.create(
        items=items, total=total, skip=pagination.skip, limit=pagination.limit
    )


@router.get("/stats", response_model=CVESummaryResponse)
async def get_cve_stats(repo: CVERepositoryDep) -> CVESummaryResponse:
    """Get CVE statistics."""
    logger.info("📊 Getting CVE statistics", source="CVERoutes")

    total_cves = repo.count_all()
    critical_count = repo.count_by_severity(CVESeverity.CRITICAL.value)
    high_count = repo.count_by_severity(CVESeverity.HIGH.value)
    medium_count = repo.count_by_severity(CVESeverity.MEDIUM.value)
    low_count = repo.count_by_severity(CVESeverity.LOW.value)

    # Recent counts
    now = datetime.utcnow()
    recent_24h_cves = repo.find_by_date_range(now - timedelta(hours=24), now)
    recent_7d_cves = repo.find_by_date_range(now - timedelta(days=7), now)

    return CVESummaryResponse(
        total_cves=total_cves,
        critical_count=critical_count,
        high_count=high_count,
        medium_count=medium_count,
        low_count=low_count,
        recent_24h=len(recent_24h_cves),
        recent_7d=len(recent_7d_cves),
    )


@router.get("/recent", response_model=list[CVEResponse])
async def get_recent_cves(
    repo: CVERepositoryDep,
    limit: int = Query(default=50, ge=1, le=500, description="Number of recent CVEs to return"),
) -> list[CVEResponse]:
    """Get most recent CVEs."""
    logger.info("🕒 Getting recent CVEs", source="CVERoutes", limit=limit)

    cves = repo.find_recent(limit=limit)
    return [cve_to_response(cve) for cve in cves]


@router.get("/critical", response_model=list[CVEResponse])
async def get_critical_cves(repo: CVERepositoryDep) -> list[CVEResponse]:
    """Get critical severity CVEs."""
    logger.info("🔴 Getting critical CVEs", source="CVERoutes")

    cves = repo.find_by_severity(CVESeverity.CRITICAL.value)
    return [cve_to_response(cve) for cve in cves]


@router.get("/{cve_id}", response_model=CVEResponse)
async def get_cve_by_id(cve_id: str, repo: CVERepositoryDep) -> CVEResponse:
    """
    Get a specific CVE by its ID.

    Returns 404 if CVE not found.
    """
    logger.info("🔍 Getting CVE by ID", source="CVERoutes", cve_id=cve_id)

    cve = repo.find_by_id(cve_id)

    if not cve:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CVE {cve_id} not found",
        )

    return cve_to_response(cve)


@router.post("/", response_model=SuccessResponse, status_code=status.HTTP_201_CREATED)
async def create_cve(request: CVECreateRequest, repo: CVERepositoryDep) -> SuccessResponse:
    """
    Create a new CVE (manual entry).

    Useful for adding CVEs from sources not yet automated.
    """
    logger.info("➕ Creating CVE", source="CVERoutes", cve_id=request.cve_id)

    # Check if CVE already exists
    existing = repo.find_by_id(request.cve_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"CVE {request.cve_id} already exists",
        )

    # Map severity string to enum
    try:
        severity = CVESeverity(request.severity)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid severity: {request.severity}",
        )

    # Create CVSS if provided
    cvss = None
    if request.cvss_base_score is not None:
        cvss = CVSS(
            version="3.1",
            base_score=request.cvss_base_score,
            vector_string="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N",  # Default vector
            exploitability_score=None,
            impact_score=None,
        )

    # Create CVE entity
    now = datetime.utcnow()
    cve = CVE(
        cve_id=request.cve_id,
        description=request.description,
        published_date=now,
        last_modified_date=now,
        severity=severity,
        cvss=cvss,
        cwe_ids=request.cwe_ids,
        references=request.references,
        affected_vendors=request.affected_vendors,
        affected_products=request.affected_products,
        source="Manual Entry",
        raw_data=None,
    )

    # Save to repository
    repo.save(cve)

    return SuccessResponse(
        message=f"CVE {request.cve_id} created successfully",
        data={"cve_id": request.cve_id},
    )


@router.delete("/{cve_id}", response_model=SuccessResponse)
async def delete_cve(cve_id: str, repo: CVERepositoryDep) -> SuccessResponse:
    """
    Delete a CVE by its ID.

    Returns 404 if CVE not found.
    """
    logger.info("🗑️ Deleting CVE", source="CVERoutes", cve_id=cve_id)

    deleted = repo.delete_by_id(cve_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CVE {cve_id} not found",
        )

    return SuccessResponse(
        message=f"CVE {cve_id} deleted successfully",
        data={"cve_id": cve_id},
    )
