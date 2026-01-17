"""
IOC Extractor Domain Service.

Business logic for IOC extraction and validation.
Pure Python - no infrastructure dependencies.
"""

from __future__ import annotations

from datetime import datetime

from ..entities import IOC, IOCConfidence, IOCType, ThreatIntel


class IOCExtractorService:
    """
    Domain service for IOC extraction logic.

    Coordinates IOC extraction from threat intelligence documents.
    Infrastructure-agnostic - uses ports for actual extraction.
    """

    def validate_ioc_confidence(self, ioc: IOC) -> bool:
        """
        Validate IOC confidence level based on business rules.

        Args:
            ioc: IOC to validate

        Returns:
            True if IOC meets confidence threshold
        """
        # Business rule: Only accept HIGH or MEDIUM confidence IOCs
        return ioc.confidence in (IOCConfidence.HIGH, IOCConfidence.MEDIUM)

    def deduplicate_iocs(self, iocs: list[IOC]) -> list[IOC]:
        """
        Remove duplicate IOCs, keeping highest confidence.

        Args:
            iocs: List of IOCs (may contain duplicates)

        Returns:
            Deduplicated list of IOCs
        """
        # Group by (value, type)
        ioc_map: dict[tuple[str, IOCType], IOC] = {}

        for ioc in iocs:
            key = (ioc.value, ioc.ioc_type)

            # Keep IOC with highest confidence
            if key not in ioc_map:
                ioc_map[key] = ioc
            else:
                existing = ioc_map[key]
                if self._confidence_score(ioc) > self._confidence_score(existing):
                    ioc_map[key] = ioc

        return list(ioc_map.values())

    def enrich_ioc_with_context(self, ioc: IOC, document: ThreatIntel) -> IOC:
        """
        Enrich IOC with additional context from source document.

        Args:
            ioc: IOC to enrich
            document: Source threat intelligence document

        Returns:
            Enriched IOC
        """
        # Add document tags to IOC
        if document.tags:
            ioc.tags.extend(document.tags)
            ioc.tags = list(set(ioc.tags))  # Deduplicate

        # Link to related CVEs from document
        if document.related_cves:
            ioc.related_cves.extend(document.related_cves)
            ioc.related_cves = list(set(ioc.related_cves))

        return ioc

    def filter_high_confidence_iocs(self, iocs: list[IOC]) -> list[IOC]:
        """
        Filter IOCs to only high confidence ones.

        Args:
            iocs: List of IOCs

        Returns:
            Filtered list containing only high confidence IOCs
        """
        return [ioc for ioc in iocs if ioc.is_high_confidence]

    def group_iocs_by_type(self, iocs: list[IOC]) -> dict[IOCType, list[IOC]]:
        """
        Group IOCs by their type.

        Args:
            iocs: List of IOCs

        Returns:
            Dictionary mapping IOC type to list of IOCs
        """
        grouped: dict[IOCType, list[IOC]] = {}

        for ioc in iocs:
            if ioc.ioc_type not in grouped:
                grouped[ioc.ioc_type] = []
            grouped[ioc.ioc_type].append(ioc)

        return grouped

    def calculate_ioc_statistics(self, iocs: list[IOC]) -> dict[str, int | float]:
        """
        Calculate statistics about extracted IOCs.

        Args:
            iocs: List of IOCs

        Returns:
            Dictionary with statistics
        """
        if not iocs:
            return {
                "total": 0,
                "high_confidence": 0,
                "medium_confidence": 0,
                "low_confidence": 0,
                "unique_types": 0,
            }

        return {
            "total": len(iocs),
            "high_confidence": sum(1 for ioc in iocs if ioc.confidence == IOCConfidence.HIGH),
            "medium_confidence": sum(1 for ioc in iocs if ioc.confidence == IOCConfidence.MEDIUM),
            "low_confidence": sum(1 for ioc in iocs if ioc.confidence == IOCConfidence.LOW),
            "unique_types": len(set(ioc.ioc_type for ioc in iocs)),
            "network_indicators": sum(1 for ioc in iocs if ioc.is_network_indicator),
            "file_hashes": sum(1 for ioc in iocs if ioc.is_hash),
        }

    @staticmethod
    def _confidence_score(ioc: IOC) -> int:
        """Convert confidence to numeric score for comparison."""
        confidence_scores = {
            IOCConfidence.HIGH: 3,
            IOCConfidence.MEDIUM: 2,
            IOCConfidence.LOW: 1,
            IOCConfidence.UNKNOWN: 0,
        }
        return confidence_scores.get(ioc.confidence, 0)
