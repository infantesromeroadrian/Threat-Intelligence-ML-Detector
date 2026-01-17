"""
FastAPI application entry point.

Threat Intelligence Aggregator REST API.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ...infrastructure.config.settings import settings

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Threat Intelligence Aggregator - REST API for CVE, IOC, and threat analysis",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Root Endpoints
# =============================================================================


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Root endpoint - API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, object]:
    """
    Health check endpoint.

    Used by Docker healthcheck and monitoring systems.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.app_env,
        "version": settings.app_version,
    }


# =============================================================================
# Import Routers
# =============================================================================
from .routes import alerts, cves, iocs, threats, topics

app.include_router(cves.router, prefix="/api/cves", tags=["CVEs"])
app.include_router(iocs.router, prefix="/api/iocs", tags=["IOCs"])
app.include_router(threats.router, prefix="/api/threats", tags=["Threat Intelligence"])
app.include_router(topics.router, prefix="/api/topics", tags=["Topics"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])


# =============================================================================
# Startup & Shutdown Events
# =============================================================================


@app.on_event("startup")
async def startup_event() -> None:
    """Actions to perform on application startup."""
    # Future: Initialize database connections, load ML models, etc.
    pass


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Actions to perform on application shutdown."""
    # Future: Close database connections, cleanup resources, etc.
    pass
