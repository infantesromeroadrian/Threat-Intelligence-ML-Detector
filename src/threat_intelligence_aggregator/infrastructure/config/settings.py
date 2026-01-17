"""
Application settings using Pydantic Settings.

Loads configuration from environment variables and .env file.
Type-safe configuration with validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===================================================================
    # APPLICATION SETTINGS
    # ===================================================================
    app_name: str = Field(default="Threat Intelligence Aggregator")
    app_version: str = Field(default="0.1.0")
    app_env: Literal["development", "staging", "production"] = Field(default="development")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")

    # ===================================================================
    # DATABASE SETTINGS
    # ===================================================================
    database_url: str = Field(default="sqlite:///./threat_intel.db")

    # ===================================================================
    # API KEYS
    # ===================================================================
    nvd_api_key: str = Field(default="")
    otx_api_key: str = Field(default="")

    # ===================================================================
    # ML MODEL SETTINGS
    # ===================================================================
    spacy_model: str = Field(default="en_core_web_sm")
    bert_model: str = Field(default="bert-base-uncased")
    word2vec_vector_size: int = Field(default=100, ge=50, le=500)
    word2vec_window: int = Field(default=5, ge=1, le=20)
    word2vec_min_count: int = Field(default=2, ge=1)
    lda_num_topics: int = Field(default=10, ge=2, le=100)
    lda_passes: int = Field(default=10, ge=1, le=50)
    lda_iterations: int = Field(default=100, ge=10, le=1000)

    # ===================================================================
    # SCRAPER SETTINGS
    # ===================================================================
    scraper_user_agent: str = Field(default="ThreatIntelAggregator/0.1.0 (Educational Project)")
    scraper_rate_limit_seconds: int = Field(default=2, ge=1)
    scraper_max_retries: int = Field(default=3, ge=1, le=10)
    scraper_timeout_seconds: int = Field(default=30, ge=5, le=300)

    # ===================================================================
    # CACHE SETTINGS
    # ===================================================================
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600, ge=60)
    cache_dir: Path = Field(default=Path("./threat_intel_cache"))

    # ===================================================================
    # API SETTINGS
    # ===================================================================
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    api_reload: bool = Field(default=True)
    api_cors_origins: list[str] = Field(
        default=[
            "http://localhost",
            "http://localhost:80",
            "http://localhost:3000",
            "http://localhost:8000",
            "http://localhost:8501",
        ]
    )

    # ===================================================================
    # STREAMLIT SETTINGS
    # ===================================================================
    streamlit_port: int = Field(default=8501, ge=1024, le=65535)

    # ===================================================================
    # NOTIFICATION SETTINGS
    # ===================================================================
    slack_webhook_url: str = Field(default="")
    smtp_host: str = Field(default="")
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_user: str = Field(default="")
    smtp_password: str = Field(default="")
    smtp_from: str = Field(default="")
    smtp_to: str = Field(default="")

    # ===================================================================
    # SECURITY SETTINGS
    # ===================================================================
    secret_key: str = Field(default="change-me-in-production")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30, ge=5, le=1440)

    @field_validator("cache_dir")
    @classmethod
    def create_cache_dir(cls, v: Path) -> Path:
        """Ensure cache directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env == "production"

    @property
    def has_nvd_api_key(self) -> bool:
        """Check if NVD API key is configured."""
        return bool(self.nvd_api_key)

    @property
    def has_otx_api_key(self) -> bool:
        """Check if OTX API key is configured."""
        return bool(self.otx_api_key)

    @property
    def has_slack_webhook(self) -> bool:
        """Check if Slack webhook is configured."""
        return bool(self.slack_webhook_url)

    @property
    def has_email_config(self) -> bool:
        """Check if email configuration is complete."""
        return all([self.smtp_host, self.smtp_user, self.smtp_password, self.smtp_from])


# Global settings instance
settings = Settings()
