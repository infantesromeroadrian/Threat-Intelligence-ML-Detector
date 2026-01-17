"""
SQLite IOC Repository implementation.

Implements IOCRepository port using SQLAlchemy + SQLite.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, JSON, String, Text
from sqlalchemy.orm import Session

from ....domain.entities import IOC, IOCConfidence, IOCType
from ....infrastructure.config.logging_config import get_logger
from .base import Base, BaseRepository

logger = get_logger(__name__)


# SQLAlchemy ORM Model
class IOCModel(Base):
    """SQLAlchemy model for IOC."""

    __tablename__ = "iocs"

    # Composite primary key (value + type)
    value = Column(String(500), primary_key=True, index=True)
    ioc_type = Column(String(50), primary_key=True, index=True)

    confidence = Column(String(20), nullable=False, index=True)
    source_document_id = Column(String(100), nullable=False, index=True)
    extracted_at = Column(DateTime, nullable=False, index=True)
    context = Column(Text, default="")

    # Arrays stored as JSON
    tags = Column(JSON, nullable=False, default=list)
    related_cves = Column(JSON, nullable=False, default=list)

    first_seen = Column(DateTime, nullable=True)
    last_seen = Column(DateTime, nullable=True)
    reputation_score = Column(Float, nullable=True)


class SQLiteIOCRepository(BaseRepository):
    """SQLite implementation of IOCRepository."""

    def save(self, ioc: IOC) -> None:
        """Save an IOC to the repository."""
        with self.get_session() as session:
            self._save_ioc(session, ioc)
            session.commit()

        logger.info(
            f"💾 Saved IOC {ioc.ioc_type.value}: {ioc.value[:50]}",
            source="SQLiteIOCRepository",
        )

    def save_many(self, iocs: list[IOC]) -> None:
        """Save multiple IOCs to the repository."""
        with self.get_session() as session:
            for ioc in iocs:
                self._save_ioc(session, ioc)
            session.commit()

        logger.info(
            f"💾 Saved {len(iocs)} IOCs",
            source="SQLiteIOCRepository",
            count=len(iocs),
        )

    def find_by_value(self, value: str) -> IOC | None:
        """Find an IOC by its value."""
        with self.get_session() as session:
            model = session.query(IOCModel).filter(IOCModel.value == value).first()
            return self._to_entity(model) if model else None

    def find_by_type(self, ioc_type: str) -> list[IOC]:
        """Find IOCs by type."""
        with self.get_session() as session:
            models = session.query(IOCModel).filter(IOCModel.ioc_type == ioc_type).all()
            return [self._to_entity(m) for m in models]

    def find_by_source(self, source_document_id: str) -> list[IOC]:
        """Find IOCs extracted from a specific document."""
        with self.get_session() as session:
            models = (
                session.query(IOCModel)
                .filter(IOCModel.source_document_id == source_document_id)
                .all()
            )
            return [self._to_entity(m) for m in models]

    def find_by_confidence(self, min_confidence: str) -> list[IOC]:
        """Find IOCs with minimum confidence level."""
        # Map confidence levels
        confidence_order = ["HIGH", "MEDIUM", "LOW", "UNKNOWN"]
        try:
            min_idx = confidence_order.index(min_confidence.upper())
            valid_confidences = confidence_order[: min_idx + 1]
        except ValueError:
            valid_confidences = ["HIGH"]

        with self.get_session() as session:
            models = (
                session.query(IOCModel).filter(IOCModel.confidence.in_(valid_confidences)).all()
            )
            return [self._to_entity(m) for m in models]

    def find_recent(self, limit: int = 100) -> list[IOC]:
        """Find the most recently extracted IOCs."""
        with self.get_session() as session:
            models = (
                session.query(IOCModel).order_by(IOCModel.extracted_at.desc()).limit(limit).all()
            )
            return [self._to_entity(m) for m in models]

    def count(self) -> int:
        """Count total IOCs in repository."""
        with self.get_session() as session:
            return session.query(IOCModel).count()

    def delete(self, value: str, ioc_type: str) -> None:
        """Delete an IOC from the repository."""
        with self.get_session() as session:
            session.query(IOCModel).filter(
                IOCModel.value == value, IOCModel.ioc_type == ioc_type
            ).delete()
            session.commit()

    # =========================================================================
    # Private Methods
    # =========================================================================

    @staticmethod
    def _save_ioc(session: Session, ioc: IOC) -> None:
        """Save IOC in a session (without committing)."""
        # Check if exists
        existing = (
            session.query(IOCModel)
            .filter(IOCModel.value == ioc.value, IOCModel.ioc_type == ioc.ioc_type.value)
            .first()
        )

        if existing:
            # Update
            existing.confidence = ioc.confidence.value
            existing.context = ioc.context
            existing.tags = ioc.tags
            existing.related_cves = ioc.related_cves
            existing.last_seen = datetime.utcnow()
            if ioc.reputation_score is not None:
                existing.reputation_score = ioc.reputation_score
        else:
            # Insert
            model = IOCModel(
                value=ioc.value,
                ioc_type=ioc.ioc_type.value,
                confidence=ioc.confidence.value,
                source_document_id=ioc.source_document_id,
                extracted_at=ioc.extracted_at,
                context=ioc.context,
                tags=ioc.tags,
                related_cves=ioc.related_cves,
                first_seen=ioc.first_seen or datetime.utcnow(),
                last_seen=ioc.last_seen or datetime.utcnow(),
                reputation_score=ioc.reputation_score,
            )
            session.add(model)

    @staticmethod
    def _to_entity(model: IOCModel) -> IOC:
        """Convert SQLAlchemy model to domain entity."""
        return IOC(
            value=model.value,
            ioc_type=IOCType(model.ioc_type),
            confidence=IOCConfidence(model.confidence),
            source_document_id=model.source_document_id,
            extracted_at=model.extracted_at,
            context=model.context,
            tags=model.tags or [],
            related_cves=model.related_cves or [],
            first_seen=model.first_seen,
            last_seen=model.last_seen,
            reputation_score=model.reputation_score,
        )
