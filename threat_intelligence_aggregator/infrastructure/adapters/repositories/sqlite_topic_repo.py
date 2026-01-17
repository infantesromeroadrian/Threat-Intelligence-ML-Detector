"""
SQLite Topic Repository implementation.

Implements TopicRepository port using SQLAlchemy + SQLite.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String
from sqlalchemy.orm import Session

from ....domain.entities import Topic, TopicWord
from ....infrastructure.config.logging_config import get_logger
from .base import Base, BaseRepository

logger = get_logger(__name__)


# SQLAlchemy ORM Model
class TopicModel(Base):
    """SQLAlchemy model for Topic."""

    __tablename__ = "topics"

    topic_id = Column(String(50), primary_key=True, index=True)
    topic_number = Column(Integer, nullable=False, index=True)
    label = Column(String(200), nullable=True)
    coherence_score = Column(Float, nullable=True)
    document_count = Column(Integer, nullable=False, default=0)
    discovery_date = Column(DateTime, nullable=False, index=True)
    last_updated = Column(DateTime, nullable=False)

    # Arrays stored as JSON
    keywords = Column(JSON, nullable=False)  # List of {word, probability}
    related_cves = Column(JSON, nullable=False, default=list)
    related_iocs = Column(JSON, nullable=False, default=list)

    extra_metadata = Column(JSON, nullable=True)


class SQLiteTopicRepository(BaseRepository):
    """SQLite implementation of TopicRepository."""

    def save(self, topic: Topic) -> None:
        """Save a topic to the repository."""
        with self.get_session() as session:
            self._save_topic(session, topic)
            session.commit()

        logger.info(
            "💾 Saved Topic",
            source="SQLiteTopicRepository",
            topic_id=topic.topic_id,
            topic_number=topic.topic_number,
        )

    def save_many(self, topics: list[Topic]) -> None:
        """Save multiple topics to the repository."""
        with self.get_session() as session:
            for topic in topics:
                self._save_topic(session, topic)
            session.commit()

        logger.info(
            f"💾 Saved {len(topics)} Topics",
            source="SQLiteTopicRepository",
            count=len(topics),
        )

    def find_by_id(self, topic_id: str) -> Topic | None:
        """Find a topic by its ID."""
        with self.get_session() as session:
            model = session.query(TopicModel).filter(TopicModel.topic_id == topic_id).first()
            return self._to_entity(model) if model else None

    def find_by_topic_number(self, topic_number: int) -> Topic | None:
        """Find a topic by its number."""
        with self.get_session() as session:
            model = (
                session.query(TopicModel).filter(TopicModel.topic_number == topic_number).first()
            )
            return self._to_entity(model) if model else None

    def find_all(self) -> list[Topic]:
        """Find all topics."""
        with self.get_session() as session:
            models = session.query(TopicModel).order_by(TopicModel.topic_number).all()
            return [self._to_entity(m) for m in models]

    def find_labeled(self) -> list[Topic]:
        """Find topics that have been manually labeled."""
        with self.get_session() as session:
            models = session.query(TopicModel).filter(TopicModel.label.isnot(None)).all()
            return [self._to_entity(m) for m in models]

    def find_significant(self, min_documents: int = 5, min_coherence: float = 0.4) -> list[Topic]:
        """
        Find significant topics based on document count and coherence.

        Args:
            min_documents: Minimum number of documents
            min_coherence: Minimum coherence score

        Returns:
            List of significant topics
        """
        with self.get_session() as session:
            models = (
                session.query(TopicModel)
                .filter(
                    TopicModel.document_count >= min_documents,
                    TopicModel.coherence_score >= min_coherence,
                )
                .all()
            )
            return [self._to_entity(m) for m in models]

    def find_by_date_range(self, start_date: datetime, end_date: datetime) -> list[Topic]:
        """Find topics discovered within a date range."""
        with self.get_session() as session:
            models = (
                session.query(TopicModel)
                .filter(
                    TopicModel.discovery_date >= start_date,
                    TopicModel.discovery_date <= end_date,
                )
                .all()
            )
            return [self._to_entity(m) for m in models]

    def find_recent(self, limit: int = 10) -> list[Topic]:
        """Find the most recently discovered topics."""
        with self.get_session() as session:
            models = (
                session.query(TopicModel)
                .order_by(TopicModel.discovery_date.desc())
                .limit(limit)
                .all()
            )
            return [self._to_entity(m) for m in models]

    def find_by_keyword(self, keyword: str, top_n: int = 10) -> list[Topic]:
        """
        Find topics that contain a specific keyword.

        Args:
            keyword: Keyword to search for
            top_n: Maximum number of topics to return

        Returns:
            List of topics containing the keyword
        """
        with self.get_session() as session:
            # Get all topics and filter in Python (SQLite JSON queries are limited)
            models = session.query(TopicModel).all()

            matching_topics = []
            keyword_lower = keyword.lower()

            for model in models:
                keywords_data = model.keywords or []
                for kw_data in keywords_data:
                    if kw_data.get("word", "").lower() == keyword_lower:
                        matching_topics.append(model)
                        break

            # Sort by coherence score (if available) or document count
            matching_topics.sort(
                key=lambda m: (m.coherence_score or 0.0, m.document_count), reverse=True
            )

            return [self._to_entity(m) for m in matching_topics[:top_n]]

    def update_document_count(self, topic_id: str, count: int) -> None:
        """
        Update the document count for a topic.

        Args:
            topic_id: Topic ID
            count: New document count
        """
        with self.get_session() as session:
            session.query(TopicModel).filter(TopicModel.topic_id == topic_id).update(
                {"document_count": count, "last_updated": datetime.utcnow()}
            )
            session.commit()

        logger.info(
            "📊 Updated topic document count",
            source="SQLiteTopicRepository",
            topic_id=topic_id,
            count=count,
        )

    def update_label(self, topic_id: str, label: str) -> None:
        """
        Update the label for a topic.

        Args:
            topic_id: Topic ID
            label: New label
        """
        with self.get_session() as session:
            session.query(TopicModel).filter(TopicModel.topic_id == topic_id).update(
                {"label": label, "last_updated": datetime.utcnow()}
            )
            session.commit()

        logger.info(
            "🏷️ Updated topic label",
            source="SQLiteTopicRepository",
            topic_id=topic_id,
            label=label,
        )

    def count_all(self) -> int:
        """Count total number of topics."""
        with self.get_session() as session:
            return session.query(TopicModel).count()

    def delete_by_id(self, topic_id: str) -> bool:
        """Delete a topic by its ID."""
        with self.get_session() as session:
            deleted = session.query(TopicModel).filter(TopicModel.topic_id == topic_id).delete()
            session.commit()

        if deleted:
            logger.info(
                "🗑️ Deleted Topic",
                source="SQLiteTopicRepository",
                topic_id=topic_id,
            )

        return deleted > 0

    # Private helper methods

    def _save_topic(self, session: Session, topic: Topic) -> None:
        """Internal method to save topic to session."""
        # Convert TopicWord objects to dicts
        keywords_data = [{"word": kw.word, "probability": kw.probability} for kw in topic.keywords]

        model = TopicModel(
            topic_id=topic.topic_id,
            topic_number=topic.topic_number,
            keywords=keywords_data,
            label=topic.label,
            coherence_score=topic.coherence_score,
            document_count=topic.document_count,
            discovery_date=topic.discovery_date,
            last_updated=topic.last_updated,
            related_cves=topic.related_cves,
            related_iocs=topic.related_iocs,
            extra_metadata=topic.metadata,
        )

        session.merge(model)  # Insert or update

    def _to_entity(self, model: TopicModel) -> Topic:
        """Convert SQLAlchemy model to domain entity."""
        # Convert keyword dicts to TopicWord objects
        keywords = [
            TopicWord(word=kw["word"], probability=kw["probability"])
            for kw in (model.keywords or [])
        ]

        return Topic(
            topic_id=model.topic_id,
            topic_number=model.topic_number,
            keywords=keywords,
            label=model.label,
            coherence_score=model.coherence_score,
            document_count=model.document_count,
            discovery_date=model.discovery_date,
            last_updated=model.last_updated,
            related_cves=model.related_cves or [],
            related_iocs=model.related_iocs or [],
            metadata=model.extra_metadata or {},
        )
