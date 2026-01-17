"""
SQLite ThreatIntel Repository implementation.

Implements ThreatIntelRepository port using SQLAlchemy + SQLite.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Session

from ....domain.entities import ThreatIntel, ThreatSeverity, ThreatType
from ....infrastructure.config.logging_config import get_logger
from .base import Base, BaseRepository

logger = get_logger(__name__)


# SQLAlchemy ORM Model
class ThreatIntelModel(Base):
    """SQLAlchemy model for ThreatIntel."""

    __tablename__ = "threat_intel"

    document_id = Column(String(50), primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    threat_type = Column(String(30), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    source = Column(String(100), nullable=False)
    published_date = Column(DateTime, nullable=False, index=True)
    collected_at = Column(DateTime, nullable=False, index=True)
    author = Column(String(200), nullable=True)

    # Arrays stored as JSON
    tags = Column(JSON, nullable=False, default=list)
    related_cves = Column(JSON, nullable=False, default=list)

    iocs_count = Column(Integer, nullable=False, default=0)
    url = Column(String(500), nullable=True)
    tlp = Column(String(10), nullable=False, default="WHITE")
    confidence_score = Column(Float, nullable=True)
    raw_data = Column(JSON, nullable=True)


class SQLiteThreatIntelRepository(BaseRepository):
    """SQLite implementation of ThreatIntelRepository."""

    def save(self, threat_intel: ThreatIntel) -> None:
        """Save a threat intelligence document to the repository."""
        with self.get_session() as session:
            self._save_threat_intel(session, threat_intel)
            session.commit()

        logger.info(
            "💾 Saved ThreatIntel document",
            source="SQLiteThreatIntelRepository",
            document_id=threat_intel.document_id,
            threat_type=threat_intel.threat_type.value,
        )

    def save_many(self, threat_intels: list[ThreatIntel]) -> None:
        """Save multiple threat intelligence documents to the repository."""
        with self.get_session() as session:
            for threat_intel in threat_intels:
                self._save_threat_intel(session, threat_intel)
            session.commit()

        logger.info(
            f"💾 Saved {len(threat_intels)} ThreatIntel documents",
            source="SQLiteThreatIntelRepository",
            count=len(threat_intels),
        )

    def find_by_id(self, document_id: str) -> ThreatIntel | None:
        """Find a threat intelligence document by its ID."""
        with self.get_session() as session:
            model = (
                session.query(ThreatIntelModel)
                .filter(ThreatIntelModel.document_id == document_id)
                .first()
            )
            return self._to_entity(model) if model else None

    def find_by_threat_type(self, threat_type: str) -> list[ThreatIntel]:
        """Find threat intelligence documents by threat type."""
        with self.get_session() as session:
            models = (
                session.query(ThreatIntelModel)
                .filter(ThreatIntelModel.threat_type == threat_type)
                .all()
            )
            return [self._to_entity(m) for m in models]

    def find_by_severity(self, severity: str) -> list[ThreatIntel]:
        """Find threat intelligence documents by severity level."""
        with self.get_session() as session:
            models = (
                session.query(ThreatIntelModel).filter(ThreatIntelModel.severity == severity).all()
            )
            return [self._to_entity(m) for m in models]

    def find_by_source(self, source: str) -> list[ThreatIntel]:
        """Find threat intelligence documents by source."""
        with self.get_session() as session:
            models = session.query(ThreatIntelModel).filter(ThreatIntelModel.source == source).all()
            return [self._to_entity(m) for m in models]

    def find_by_date_range(self, start_date: datetime, end_date: datetime) -> list[ThreatIntel]:
        """Find threat intelligence documents published within a date range."""
        with self.get_session() as session:
            models = (
                session.query(ThreatIntelModel)
                .filter(
                    ThreatIntelModel.published_date >= start_date,
                    ThreatIntelModel.published_date <= end_date,
                )
                .all()
            )
            return [self._to_entity(m) for m in models]

    def find_recent(self, limit: int = 100) -> list[ThreatIntel]:
        """Find the most recent threat intelligence documents."""
        with self.get_session() as session:
            models = (
                session.query(ThreatIntelModel)
                .order_by(ThreatIntelModel.published_date.desc())
                .limit(limit)
                .all()
            )
            return [self._to_entity(m) for m in models]

    def find_high_severity(self) -> list[ThreatIntel]:
        """Find high and critical severity threat intelligence documents."""
        with self.get_session() as session:
            models = (
                session.query(ThreatIntelModel)
                .filter(
                    ThreatIntelModel.severity.in_(
                        [ThreatSeverity.CRITICAL.value, ThreatSeverity.HIGH.value]
                    )
                )
                .all()
            )
            return [self._to_entity(m) for m in models]

    def search_by_keywords(self, keywords: list[str]) -> list[ThreatIntel]:
        """
        Search threat intelligence documents by keywords in title or content.

        Args:
            keywords: List of keywords to search for

        Returns:
            List of matching threat intelligence documents
        """
        with self.get_session() as session:
            # Build OR conditions for each keyword
            conditions = []
            for keyword in keywords:
                keyword_lower = f"%{keyword.lower()}%"
                conditions.append(ThreatIntelModel.title.ilike(keyword_lower))
                conditions.append(ThreatIntelModel.content.ilike(keyword_lower))

            if not conditions:
                return []

            from sqlalchemy import or_

            models = session.query(ThreatIntelModel).filter(or_(*conditions)).all()
            return [self._to_entity(m) for m in models]

    def count_all(self) -> int:
        """Count total number of threat intelligence documents."""
        with self.get_session() as session:
            return session.query(ThreatIntelModel).count()

    def count_by_threat_type(self, threat_type: str) -> int:
        """Count threat intelligence documents by threat type."""
        with self.get_session() as session:
            return (
                session.query(ThreatIntelModel)
                .filter(ThreatIntelModel.threat_type == threat_type)
                .count()
            )

    def delete_by_id(self, document_id: str) -> bool:
        """Delete a threat intelligence document by its ID."""
        with self.get_session() as session:
            deleted = (
                session.query(ThreatIntelModel)
                .filter(ThreatIntelModel.document_id == document_id)
                .delete()
            )
            session.commit()

        if deleted:
            logger.info(
                "🗑️ Deleted ThreatIntel document",
                source="SQLiteThreatIntelRepository",
                document_id=document_id,
            )

        return deleted > 0

    # Private helper methods

    def _save_threat_intel(self, session: Session, threat_intel: ThreatIntel) -> None:
        """Internal method to save threat intel to session."""
        model = ThreatIntelModel(
            document_id=threat_intel.document_id,
            title=threat_intel.title,
            content=threat_intel.content,
            threat_type=threat_intel.threat_type.value,
            severity=threat_intel.severity.value,
            source=threat_intel.source,
            published_date=threat_intel.published_date,
            collected_at=threat_intel.collected_at,
            author=threat_intel.author,
            tags=threat_intel.tags,
            related_cves=threat_intel.related_cves,
            iocs_count=threat_intel.iocs_count,
            url=threat_intel.url,
            tlp=threat_intel.tlp,
            confidence_score=threat_intel.confidence_score,
            raw_data=threat_intel.raw_data,
        )

        session.merge(model)  # Insert or update

    def _to_entity(self, model: ThreatIntelModel) -> ThreatIntel:
        """Convert SQLAlchemy model to domain entity."""
        return ThreatIntel(
            document_id=model.document_id,
            title=model.title,
            content=model.content,
            threat_type=ThreatType(model.threat_type),
            severity=ThreatSeverity(model.severity),
            source=model.source,
            published_date=model.published_date,
            collected_at=model.collected_at,
            author=model.author,
            tags=model.tags or [],
            related_cves=model.related_cves or [],
            iocs_count=model.iocs_count,
            url=model.url,
            tlp=model.tlp,
            confidence_score=model.confidence_score,
            raw_data=model.raw_data,
        )
