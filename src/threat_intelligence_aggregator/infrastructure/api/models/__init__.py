"""API models (DTOs) for request/response serialization."""

from .alert import (
    AlertAcknowledgeRequest,
    AlertFalsePositiveRequest,
    AlertFilterParams,
    AlertResponse,
    AlertResolveRequest,
    AlertStatsResponse,
)
from .common import (
    ErrorResponse,
    PaginatedResponse,
    PaginationParams,
    StatsResponse,
    SuccessResponse,
)
from .cve import (
    CVECreateRequest,
    CVEFilterParams,
    CVEListResponse,
    CVEResponse,
    CVESummaryResponse,
    CVSSResponse,
)
from .ioc import (
    IOCCreateRequest,
    IOCFilterParams,
    IOCResponse,
    IOCStatsResponse,
)
from .threat_intel import (
    ThreatIntelFilterParams,
    ThreatIntelResponse,
    ThreatIntelStatsResponse,
)
from .topic import (
    TopicDiscoveryRequest,
    TopicKeywordResponse,
    TopicResponse,
    TopicStatsResponse,
    TopicUpdateRequest,
)

__all__ = [
    # Common
    "PaginationParams",
    "PaginatedResponse",
    "ErrorResponse",
    "SuccessResponse",
    "StatsResponse",
    # CVE
    "CVEResponse",
    "CVSSResponse",
    "CVEListResponse",
    "CVESummaryResponse",
    "CVEFilterParams",
    "CVECreateRequest",
    # IOC
    "IOCResponse",
    "IOCFilterParams",
    "IOCCreateRequest",
    "IOCStatsResponse",
    # ThreatIntel
    "ThreatIntelResponse",
    "ThreatIntelFilterParams",
    "ThreatIntelStatsResponse",
    # Topic
    "TopicResponse",
    "TopicKeywordResponse",
    "TopicDiscoveryRequest",
    "TopicUpdateRequest",
    "TopicStatsResponse",
    # Alert
    "AlertResponse",
    "AlertFilterParams",
    "AlertAcknowledgeRequest",
    "AlertResolveRequest",
    "AlertFalsePositiveRequest",
    "AlertStatsResponse",
]
