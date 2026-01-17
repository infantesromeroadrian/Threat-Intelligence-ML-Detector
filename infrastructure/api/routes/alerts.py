"""
Alert API routes.

Endpoints for managing security alerts.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ....domain.entities import Alert
from ...adapters.repositories import SQLiteAlertRepository
from ...config.logging_config import get_logger
from ...config.settings import settings
from ..models import (
    AlertAcknowledgeRequest,
    AlertFalsePositiveRequest,
    AlertFilterParams,
    AlertResponse,
    AlertResolveRequest,
    AlertStatsResponse,
    PaginatedResponse,
    PaginationParams,
    SuccessResponse,
)

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# Dependency Injection
# =============================================================================


def get_alert_repository() -> SQLiteAlertRepository:
    """Get Alert repository instance."""
    return SQLiteAlertRepository(db_url=settings.database_url)


AlertRepositoryDep = Annotated[SQLiteAlertRepository, Depends(get_alert_repository)]


# =============================================================================
# Helper Functions
# =============================================================================


def alert_to_response(alert: Alert) -> AlertResponse:
    """Convert Alert domain entity to API response model."""
    return AlertResponse(
        alert_id=alert.alert_id,
        alert_type=alert.alert_type.value,
        severity=alert.severity.value,
        title=alert.title,
        description=alert.description,
        created_at=alert.created_at.isoformat(),
        status=alert.status.value,
        source_entity_type=alert.source_entity_type,
        source_entity_id=alert.source_entity_id,
        related_cves=alert.related_cves,
        related_iocs=alert.related_iocs,
        related_topics=alert.related_topics,
        tags=alert.tags,
        actionable_items=alert.actionable_items,
        confidence_score=alert.confidence_score,
        acknowledged_at=alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
        acknowledged_by=alert.acknowledged_by,
        resolved_at=alert.resolved_at.isoformat() if alert.resolved_at else None,
        resolved_by=alert.resolved_by,
        resolution_notes=alert.resolution_notes,
        notification_sent=alert.notification_sent,
        age_hours=alert.age_hours,
        is_active=alert.is_active,
        is_critical=alert.is_critical,
    )


# =============================================================================
# Routes
# =============================================================================


@router.get("/", response_model=PaginatedResponse[AlertResponse])
async def list_alerts(
    repo: AlertRepositoryDep,
    pagination: Annotated[PaginationParams, Depends()],
    filters: Annotated[AlertFilterParams, Depends()],
) -> PaginatedResponse[AlertResponse]:
    """List alerts with filtering and pagination."""
    logger.info("📋 Listing alerts", source="AlertsRoutes")

    # Apply filters
    alerts: list[Alert] = []

    if filters.status:
        alerts = repo.find_by_status(filters.status)
    elif filters.severity:
        alerts = repo.find_by_severity(filters.severity)
    elif filters.alert_type:
        alerts = repo.find_by_type(filters.alert_type)
    elif filters.is_critical:
        alerts = repo.find_critical()
    elif filters.is_active is not None and filters.is_active:
        alerts = repo.find_active()
    else:
        alerts = repo.find_recent(limit=1000)

    # Additional filtering
    if filters.min_confidence:
        alerts = [a for a in alerts if a.confidence_score >= filters.min_confidence]

    # Pagination
    total = len(alerts)
    paginated_alerts = alerts[pagination.skip : pagination.skip + pagination.limit]

    # Convert to response models
    items = [alert_to_response(a) for a in paginated_alerts]

    return PaginatedResponse.create(
        items=items, total=total, skip=pagination.skip, limit=pagination.limit
    )


@router.get("/stats", response_model=AlertStatsResponse)
async def get_alert_stats(repo: AlertRepositoryDep) -> AlertStatsResponse:
    """Get alert statistics."""
    logger.info("📊 Getting alert statistics", source="AlertsRoutes")

    total_alerts = repo.count_all()

    # Get all alerts for grouping
    all_alerts = repo.find_recent(limit=10000)

    # Group by status
    by_status: dict[str, int] = {}
    for alert in all_alerts:
        by_status[alert.status.value] = by_status.get(alert.status.value, 0) + 1

    # Group by severity
    by_severity: dict[str, int] = {}
    for alert in all_alerts:
        by_severity[alert.severity.value] = by_severity.get(alert.severity.value, 0) + 1

    # Group by type
    by_type: dict[str, int] = {}
    for alert in all_alerts:
        by_type[alert.alert_type.value] = by_type.get(alert.alert_type.value, 0) + 1

    active_count = len(repo.find_active())
    critical_count = len(repo.find_critical())

    # Recent 24h
    recent_24h = sum(1 for a in all_alerts if a.age_hours <= 24)

    # Average resolution time (simplified)
    resolved_alerts = [a for a in all_alerts if a.resolved_at]
    avg_resolution_time = (
        sum(a.age_hours for a in resolved_alerts) / len(resolved_alerts) if resolved_alerts else 0.0
    )

    return AlertStatsResponse(
        total_alerts=total_alerts,
        by_status=by_status,
        by_severity=by_severity,
        by_type=by_type,
        active_count=active_count,
        critical_count=critical_count,
        recent_24h=recent_24h,
        avg_resolution_time_hours=round(avg_resolution_time, 2),
    )


@router.get("/active", response_model=list[AlertResponse])
async def get_active_alerts(repo: AlertRepositoryDep) -> list[AlertResponse]:
    """Get active alerts."""
    logger.info("✅ Getting active alerts", source="AlertsRoutes")

    alerts = repo.find_active()
    return [alert_to_response(a) for a in alerts]


@router.get("/critical", response_model=list[AlertResponse])
async def get_critical_alerts(repo: AlertRepositoryDep) -> list[AlertResponse]:
    """Get critical severity alerts."""
    logger.info("🔴 Getting critical alerts", source="AlertsRoutes")

    alerts = repo.find_critical()
    return [alert_to_response(a) for a in alerts]


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert_by_id(alert_id: str, repo: AlertRepositoryDep) -> AlertResponse:
    """Get a specific alert by ID."""
    logger.info("🔍 Getting alert by ID", source="AlertsRoutes", alert_id=alert_id)

    alert = repo.find_by_id(alert_id)

    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found",
        )

    return alert_to_response(alert)


@router.post("/{alert_id}/acknowledge", response_model=SuccessResponse)
async def acknowledge_alert(
    alert_id: str, request: AlertAcknowledgeRequest, repo: AlertRepositoryDep
) -> SuccessResponse:
    """Acknowledge an alert."""
    logger.info("✅ Acknowledging alert", source="AlertsRoutes", alert_id=alert_id)

    alert = repo.find_by_id(alert_id)
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found",
        )

    alert.acknowledge(request.acknowledged_by)
    repo.save(alert)

    return SuccessResponse(
        message=f"Alert {alert_id} acknowledged",
        data={"alert_id": alert_id, "acknowledged_by": request.acknowledged_by},
    )


@router.post("/{alert_id}/resolve", response_model=SuccessResponse)
async def resolve_alert(
    alert_id: str, request: AlertResolveRequest, repo: AlertRepositoryDep
) -> SuccessResponse:
    """Resolve an alert."""
    logger.info("✅ Resolving alert", source="AlertsRoutes", alert_id=alert_id)

    alert = repo.find_by_id(alert_id)
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found",
        )

    alert.resolve(request.resolved_by, request.resolution_notes)
    repo.save(alert)

    return SuccessResponse(
        message=f"Alert {alert_id} resolved",
        data={"alert_id": alert_id, "resolved_by": request.resolved_by},
    )


@router.post("/{alert_id}/false-positive", response_model=SuccessResponse)
async def mark_false_positive(
    alert_id: str, request: AlertFalsePositiveRequest, repo: AlertRepositoryDep
) -> SuccessResponse:
    """Mark an alert as false positive."""
    logger.info("❌ Marking alert as false positive", source="AlertsRoutes", alert_id=alert_id)

    alert = repo.find_by_id(alert_id)
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found",
        )

    alert.mark_false_positive(request.marked_by, request.notes)
    repo.save(alert)

    return SuccessResponse(
        message=f"Alert {alert_id} marked as false positive",
        data={"alert_id": alert_id, "marked_by": request.marked_by},
    )
