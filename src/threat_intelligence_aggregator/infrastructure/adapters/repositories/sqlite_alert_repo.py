"""
SQLite Alert Repository implementation.

Implements AlertRepository port using SQLAlchemy + SQLite.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, String, Text
from sqlalchemy.orm import Session

from ....domain.entities import Alert, AlertSeverity, AlertStatus, AlertType
from ....infrastructure.config.logging_config import get_logger
from .base import Base, BaseRepository

logger = get_logger(__name__)


# SQLAlchemy ORM Model
class AlertModel(Base):
    """SQLAlchemy model for Alert."""

    __tablename__ = "alerts"

    alert_id = Column(String(50), primary_key=True, index=True)
    alert_type = Column(String(30), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)

    source_entity_type = Column(String(50), nullable=False)
    source_entity_id = Column(String(50), nullable=False)

    # Arrays stored as JSON
    related_cves = Column(JSON, nullable=False, default=list)
    related_iocs = Column(JSON, nullable=False, default=list)
    related_topics = Column(JSON, nullable=False, default=list)
    tags = Column(JSON, nullable=False, default=list)
    actionable_items = Column(JSON, nullable=False, default=list)

    confidence_score = Column(Float, nullable=False, default=0.0)

    # Workflow tracking
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(200), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(String(200), nullable=True)
    resolution_notes = Column(Text, nullable=False, default="")
    notification_sent = Column(Boolean, nullable=False, default=False)

    extra_metadata = Column(JSON, nullable=True)


class SQLiteAlertRepository(BaseRepository):
    """SQLite implementation of AlertRepository."""

    def save(self, alert: Alert) -> None:
        """Save an alert to the repository."""
        with self.get_session() as session:
            self._save_alert(session, alert)
            session.commit()

        logger.info(
            "💾 Saved Alert",
            source="SQLiteAlertRepository",
            alert_id=alert.alert_id,
            severity=alert.severity.value,
        )

    def save_many(self, alerts: list[Alert]) -> None:
        """Save multiple alerts to the repository."""
        with self.get_session() as session:
            for alert in alerts:
                self._save_alert(session, alert)
            session.commit()

        logger.info(
            f"💾 Saved {len(alerts)} Alerts",
            source="SQLiteAlertRepository",
            count=len(alerts),
        )

    def find_by_id(self, alert_id: str) -> Alert | None:
        """Find an alert by its ID."""
        with self.get_session() as session:
            model = session.query(AlertModel).filter(AlertModel.alert_id == alert_id).first()
            return self._to_entity(model) if model else None

    def find_by_status(self, status: str) -> list[Alert]:
        """Find alerts by status."""
        with self.get_session() as session:
            models = session.query(AlertModel).filter(AlertModel.status == status).all()
            return [self._to_entity(m) for m in models]

    def find_by_severity(self, severity: str) -> list[Alert]:
        """Find alerts by severity level."""
        with self.get_session() as session:
            models = session.query(AlertModel).filter(AlertModel.severity == severity).all()
            return [self._to_entity(m) for m in models]

    def find_by_type(self, alert_type: str) -> list[Alert]:
        """Find alerts by type."""
        with self.get_session() as session:
            models = session.query(AlertModel).filter(AlertModel.alert_type == alert_type).all()
            return [self._to_entity(m) for m in models]

    def find_active(self) -> list[Alert]:
        """Find all active alerts (not resolved/ignored)."""
        with self.get_session() as session:
            models = (
                session.query(AlertModel)
                .filter(
                    AlertModel.status.notin_(
                        [
                            AlertStatus.RESOLVED.value,
                            AlertStatus.FALSE_POSITIVE.value,
                            AlertStatus.IGNORED.value,
                        ]
                    )
                )
                .all()
            )
            return [self._to_entity(m) for m in models]

    def find_critical(self) -> list[Alert]:
        """Find critical severity alerts."""
        with self.get_session() as session:
            models = (
                session.query(AlertModel)
                .filter(AlertModel.severity == AlertSeverity.CRITICAL.value)
                .all()
            )
            return [self._to_entity(m) for m in models]

    def find_high_severity(self) -> list[Alert]:
        """Find high and critical severity alerts."""
        with self.get_session() as session:
            models = (
                session.query(AlertModel)
                .filter(
                    AlertModel.severity.in_(
                        [AlertSeverity.CRITICAL.value, AlertSeverity.HIGH.value]
                    )
                )
                .all()
            )
            return [self._to_entity(m) for m in models]

    def find_unacknowledged(self) -> list[Alert]:
        """Find alerts that haven't been acknowledged yet."""
        with self.get_session() as session:
            models = (
                session.query(AlertModel).filter(AlertModel.status == AlertStatus.NEW.value).all()
            )
            return [self._to_entity(m) for m in models]

    def find_by_date_range(self, start_date: datetime, end_date: datetime) -> list[Alert]:
        """Find alerts created within a date range."""
        with self.get_session() as session:
            models = (
                session.query(AlertModel)
                .filter(
                    AlertModel.created_at >= start_date,
                    AlertModel.created_at <= end_date,
                )
                .all()
            )
            return [self._to_entity(m) for m in models]

    def find_recent(self, limit: int = 100) -> list[Alert]:
        """Find the most recent alerts."""
        with self.get_session() as session:
            models = (
                session.query(AlertModel).order_by(AlertModel.created_at.desc()).limit(limit).all()
            )
            return [self._to_entity(m) for m in models]

    def find_by_source_entity(self, source_entity_type: str, source_entity_id: str) -> list[Alert]:
        """
        Find alerts by source entity.

        Args:
            source_entity_type: Type of source entity (e.g., "CVE", "ThreatIntel")
            source_entity_id: ID of source entity

        Returns:
            List of alerts related to the source entity
        """
        with self.get_session() as session:
            models = (
                session.query(AlertModel)
                .filter(
                    AlertModel.source_entity_type == source_entity_type,
                    AlertModel.source_entity_id == source_entity_id,
                )
                .all()
            )
            return [self._to_entity(m) for m in models]

    def update_status(self, alert_id: str, status: AlertStatus) -> None:
        """
        Update alert status.

        Args:
            alert_id: Alert ID
            status: New status
        """
        with self.get_session() as session:
            session.query(AlertModel).filter(AlertModel.alert_id == alert_id).update(
                {"status": status.value}
            )
            session.commit()

        logger.info(
            "🔄 Updated alert status",
            source="SQLiteAlertRepository",
            alert_id=alert_id,
            status=status.value,
        )

    def mark_notification_sent(self, alert_id: str) -> None:
        """
        Mark alert as notified.

        Args:
            alert_id: Alert ID
        """
        with self.get_session() as session:
            session.query(AlertModel).filter(AlertModel.alert_id == alert_id).update(
                {"notification_sent": True}
            )
            session.commit()

        logger.info(
            "📧 Marked alert as notified",
            source="SQLiteAlertRepository",
            alert_id=alert_id,
        )

    def count_all(self) -> int:
        """Count total number of alerts."""
        with self.get_session() as session:
            return session.query(AlertModel).count()

    def count_by_status(self, status: str) -> int:
        """Count alerts by status."""
        with self.get_session() as session:
            return session.query(AlertModel).filter(AlertModel.status == status).count()

    def count_by_severity(self, severity: str) -> int:
        """Count alerts by severity."""
        with self.get_session() as session:
            return session.query(AlertModel).filter(AlertModel.severity == severity).count()

    def delete_by_id(self, alert_id: str) -> bool:
        """Delete an alert by its ID."""
        with self.get_session() as session:
            deleted = session.query(AlertModel).filter(AlertModel.alert_id == alert_id).delete()
            session.commit()

        if deleted:
            logger.info(
                "🗑️ Deleted Alert",
                source="SQLiteAlertRepository",
                alert_id=alert_id,
            )

        return deleted > 0

    # Private helper methods

    def _save_alert(self, session: Session, alert: Alert) -> None:
        """Internal method to save alert to session."""
        model = AlertModel(
            alert_id=alert.alert_id,
            alert_type=alert.alert_type.value,
            severity=alert.severity.value,
            title=alert.title,
            description=alert.description,
            created_at=alert.created_at,
            status=alert.status.value,
            source_entity_type=alert.source_entity_type,
            source_entity_id=alert.source_entity_id,
            related_cves=alert.related_cves,
            related_iocs=alert.related_iocs,
            related_topics=alert.related_topics,
            tags=alert.tags,
            actionable_items=alert.actionable_items,
            confidence_score=alert.confidence_score,
            acknowledged_at=alert.acknowledged_at,
            acknowledged_by=alert.acknowledged_by,
            resolved_at=alert.resolved_at,
            resolved_by=alert.resolved_by,
            resolution_notes=alert.resolution_notes,
            notification_sent=alert.notification_sent,
            extra_metadata=alert.metadata,
        )

        session.merge(model)  # Insert or update

    def _to_entity(self, model: AlertModel) -> Alert:
        """Convert SQLAlchemy model to domain entity."""
        return Alert(
            alert_id=model.alert_id,
            alert_type=AlertType(model.alert_type),
            severity=AlertSeverity(model.severity),
            title=model.title,
            description=model.description,
            created_at=model.created_at,
            status=AlertStatus(model.status),
            source_entity_type=model.source_entity_type,
            source_entity_id=model.source_entity_id,
            related_cves=model.related_cves or [],
            related_iocs=model.related_iocs or [],
            related_topics=model.related_topics or [],
            tags=model.tags or [],
            actionable_items=model.actionable_items or [],
            confidence_score=model.confidence_score,
            acknowledged_at=model.acknowledged_at,
            acknowledged_by=model.acknowledged_by,
            resolved_at=model.resolved_at,
            resolved_by=model.resolved_by,
            resolution_notes=model.resolution_notes,
            notification_sent=model.notification_sent,
            metadata=model.extra_metadata or {},
        )
