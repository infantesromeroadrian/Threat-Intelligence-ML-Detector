"""
Base repository with common SQLite functionality.

Provides database connection and common operations.
"""

from __future__ import annotations

import threading
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from ....infrastructure.config.logging_config import get_logger
from ....infrastructure.config.settings import settings

logger = get_logger(__name__)

# SQLAlchemy base for ORM models
Base = declarative_base()

# Global lock for table creation (prevents SQLite concurrency issues)
_table_creation_lock = threading.Lock()
_tables_created = False


class BaseRepository:
    """Base repository with database connection management."""

    def __init__(self, db_url: str | None = None) -> None:
        """
        Initialize repository with database connection.

        Args:
            db_url: Database URL (defaults to settings.database_url)
        """
        self.db_url = db_url or settings.database_url

        # Create engine
        self.engine = create_engine(
            self.db_url,
            echo=False,  # Set to True for SQL debugging
            connect_args={"check_same_thread": False} if "sqlite" in self.db_url else {},
        )

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        # Create tables with global lock to prevent concurrent SQLite errors
        global _tables_created
        with _table_creation_lock:
            if not _tables_created:
                try:
                    Base.metadata.create_all(bind=self.engine, checkfirst=True)
                    _tables_created = True
                except Exception as e:
                    # Ignore "table already exists" errors (edge case)
                    if "already exists" not in str(e).lower():
                        raise
                    _tables_created = True  # Mark as done even if tables exist

        logger.info(
            "✅ Database initialized",
            source=self.__class__.__name__,
            db_url=self._sanitize_url(self.db_url),
        )

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    @staticmethod
    def _sanitize_url(url: str) -> str:
        """Sanitize database URL for logging (remove credentials)."""
        if "@" in url:
            # Hide password
            parts = url.split("@")
            return parts[-1]
        return url
