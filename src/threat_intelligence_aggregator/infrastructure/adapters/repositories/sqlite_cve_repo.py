"""
SQLite CVE Repository implementation.

Implements CVERepository port using SQLAlchemy + SQLite.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Session

from ....domain.entities import CVE, CVSS, CVESeverity
from ....infrastructure.config.logging_config import get_logger
from .base import Base, BaseRepository

logger = get_logger(__name__)


# SQLAlchemy ORM Model
class CVEModel(Base):
    """SQLAlchemy model for CVE."""

    __tablename__ = "cves"

    cve_id = Column(String(20), primary_key=True, index=True)
    description = Column(Text, nullable=False)
    published_date = Column(DateTime, nullable=False, index=True)
    last_modified_date = Column(DateTime, nullable=False)
    severity = Column(String(20), nullable=False, index=True)

    # CVSS (stored as JSON)
    cvss_data = Column(JSON, nullable=True)

    # Arrays stored as JSON
    cwe_ids = Column(JSON, nullable=False, default=list)
    references = Column(JSON, nullable=False, default=list)
    affected_vendors = Column(JSON, nullable=False, default=list)
    affected_products = Column(JSON, nullable=False, default=list)

    source = Column(String(50), nullable=False, default="NVD")
    raw_data = Column(JSON, nullable=True)


class SQLiteCVERepository(BaseRepository):
    """SQLite implementation of CVERepository."""

    def save(self, cve: CVE) -> None:
        """Save a CVE to the repository."""
        with self.get_session() as session:
            self._save_cve(session, cve)
            session.commit()

        logger.info(
            f"💾 Saved CVE {cve.cve_id}",
            source="SQLiteCVERepository",
            cve_id=cve.cve_id,
        )

    def save_many(self, cves: list[CVE]) -> None:
        """Save multiple CVEs to the repository."""
        with self.get_session() as session:
            for cve in cves:
                self._save_cve(session, cve)
            session.commit()

        logger.info(
            f"💾 Saved {len(cves)} CVEs",
            source="SQLiteCVERepository",
            count=len(cves),
        )

    def find_by_id(self, cve_id: str) -> CVE | None:
        """Find a CVE by its ID."""
        with self.get_session() as session:
            model = session.query(CVEModel).filter(CVEModel.cve_id == cve_id).first()
            return self._to_entity(model) if model else None

    def find_by_severity(self, severity: str) -> list[CVE]:
        """Find CVEs by severity level."""
        with self.get_session() as session:
            models = session.query(CVEModel).filter(CVEModel.severity == severity).all()
            return [self._to_entity(m) for m in models]

    def find_by_date_range(self, start_date: datetime, end_date: datetime) -> list[CVE]:
        """Find CVEs published within a date range."""
        with self.get_session() as session:
            models = (
                session.query(CVEModel)
                .filter(CVEModel.published_date >= start_date, CVEModel.published_date <= end_date)
                .all()
            )
            return [self._to_entity(m) for m in models]

    def find_recent(self, limit: int = 100) -> list[CVE]:
        """Find the most recent CVEs."""
        with self.get_session() as session:
            models = (
                session.query(CVEModel).order_by(CVEModel.published_date.desc()).limit(limit).all()
            )
            return [self._to_entity(m) for m in models]

    def exists(self, cve_id: str) -> bool:
        """Check if a CVE exists in the repository."""
        with self.get_session() as session:
            return session.query(CVEModel).filter(CVEModel.cve_id == cve_id).count() > 0

    def count(self) -> int:
        """Count total CVEs in repository."""
        with self.get_session() as session:
            return session.query(CVEModel).count()

    def count_all(self) -> int:
        """Count all CVEs (alias for count)."""
        return self.count()

    def count_by_severity(self, severity: str) -> int:
        """Count CVEs by severity level."""
        with self.get_session() as session:
            return session.query(CVEModel).filter(CVEModel.severity == severity).count()

    def search_by_keyword(self, keyword: str) -> list[CVE]:
        """
        Search CVEs by keyword in description.

        Args:
            keyword: Keyword to search for

        Returns:
            List of matching CVEs
        """
        with self.get_session() as session:
            keyword_pattern = f"%{keyword.lower()}%"
            models = (
                session.query(CVEModel).filter(CVEModel.description.ilike(keyword_pattern)).all()
            )
            return [self._to_entity(m) for m in models]

    def delete(self, cve_id: str) -> None:
        """Delete a CVE from the repository."""
        with self.get_session() as session:
            session.query(CVEModel).filter(CVEModel.cve_id == cve_id).delete()
            session.commit()

    def delete_by_id(self, cve_id: str) -> bool:
        """
        Delete a CVE by ID (returns success status).

        Args:
            cve_id: CVE identifier

        Returns:
            True if CVE was deleted, False if not found
        """
        with self.get_session() as session:
            deleted = session.query(CVEModel).filter(CVEModel.cve_id == cve_id).delete()
            session.commit()
            return deleted > 0

    # =========================================================================
    # Private Methods
    # =========================================================================

    @staticmethod
    def _save_cve(session: Session, cve: CVE) -> None:
        """Save CVE in a session (without committing)."""
        # Check if exists
        existing = session.query(CVEModel).filter(CVEModel.cve_id == cve.cve_id).first()

        if existing:
            # Update
            existing.description = cve.description
            existing.last_modified_date = cve.last_modified_date
            existing.severity = cve.severity.value
            existing.cvss_data = cve.cvss.__dict__ if cve.cvss else None
            existing.cwe_ids = cve.cwe_ids
            existing.references = cve.references
            existing.affected_vendors = cve.affected_vendors
            existing.affected_products = cve.affected_products
            existing.source = cve.source
        else:
            # Insert
            model = CVEModel(
                cve_id=cve.cve_id,
                description=cve.description,
                published_date=cve.published_date,
                last_modified_date=cve.last_modified_date,
                severity=cve.severity.value,
                cvss_data=cve.cvss.__dict__ if cve.cvss else None,
                cwe_ids=cve.cwe_ids,
                references=cve.references,
                affected_vendors=cve.affected_vendors,
                affected_products=cve.affected_products,
                source=cve.source,
                raw_data=cve.raw_data,
            )
            session.add(model)

    @staticmethod
    def _to_entity(model: CVEModel) -> CVE:
        """Convert SQLAlchemy model to domain entity."""
        # Reconstruct CVSS if exists
        cvss = None
        if model.cvss_data:
            cvss = CVSS(**model.cvss_data)

        return CVE(
            cve_id=model.cve_id,
            description=model.description,
            published_date=model.published_date,
            last_modified_date=model.last_modified_date,
            severity=CVESeverity(model.severity),
            cvss=cvss,
            cwe_ids=model.cwe_ids or [],
            references=model.references or [],
            affected_vendors=model.affected_vendors or [],
            affected_products=model.affected_products or [],
            source=model.source,
            raw_data=model.raw_data,
        )
