"""
Common API models (DTOs) for requests and responses.

Shared models used across different endpoints.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

# Type variable for generic responses
T = TypeVar("T")


class PaginationParams(BaseModel):
    """Pagination query parameters."""

    skip: int = Field(default=0, ge=0, description="Number of items to skip")
    limit: int = Field(default=100, ge=1, le=1000, description="Max items to return")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""

    items: list[T] = Field(description="List of items")
    total: int = Field(description="Total number of items")
    skip: int = Field(description="Number of items skipped")
    limit: int = Field(description="Max items returned")
    has_more: bool = Field(description="Whether more items are available")

    @classmethod
    def create(cls, items: list[T], total: int, skip: int, limit: int) -> PaginatedResponse[T]:
        """
        Create paginated response.

        Args:
            items: List of items
            total: Total count
            skip: Skip offset
            limit: Limit

        Returns:
            PaginatedResponse instance
        """
        return cls(
            items=items,
            total=total,
            skip=skip,
            limit=limit,
            has_more=(skip + len(items)) < total,
        )


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(description="Error message")
    detail: str | None = Field(default=None, description="Detailed error information")
    status_code: int = Field(description="HTTP status code")


class SuccessResponse(BaseModel):
    """Generic success response."""

    message: str = Field(description="Success message")
    data: dict[str, object] | None = Field(default=None, description="Additional data")


class StatsResponse(BaseModel):
    """Statistics response."""

    metric: str = Field(description="Metric name")
    value: int | float = Field(description="Metric value")
    timestamp: str = Field(description="Timestamp (ISO format)")
