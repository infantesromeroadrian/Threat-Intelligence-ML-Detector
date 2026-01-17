"""
Alert Domain Service.

Business logic for alert generation and management.
Pure Python - no infrastructure dependencies.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from ..entities import Alert, AlertSeverity, AlertStatus, AlertType, CVE, IOC, Topic


class AlertService:
    """
    Domain service for alert generation logic.

    Implements business rules for when and how to generate alerts.
    """

    def should_alert_for_cve(self, cve: CVE) -> bool:
        """
        Determine if a CVE should trigger an alert.

        Business rules:
        - CRITICAL or HIGH severity CVEs
        - OR CVEs with CVSS score >= 7.0

        Args:
            cve: CVE to evaluate

        Returns:
            True if alert should be generated
        """
        return cve.is_high_or_critical or (cve.cvss_score or 0) >= 7.0

    def should_alert_for_ioc(self, ioc: IOC) -> bool:
        """
        Determine if an IOC should trigger an alert.

        Business rules:
        - HIGH confidence network indicators (IP, domain, URL)
        - OR file hashes with malicious reputation (>= 0.7)

        Args:
            ioc: IOC to evaluate

        Returns:
            True if alert should be generated
        """
        is_high_conf_network = ioc.is_high_confidence and ioc.is_network_indicator
        is_malicious_hash = ioc.is_hash and ioc.is_malicious

        return is_high_conf_network or is_malicious_hash

    def should_alert_for_topic(self, topic: Topic) -> bool:
        """
        Determine if a topic should trigger an alert (emerging threat).

        Business rules:
        - Significant topic (>= 5 docs)
        - With rapid growth (if tracking document count over time)

        Args:
            topic: Topic to evaluate

        Returns:
            True if alert should be generated
        """
        return topic.is_significant

    def create_cve_alert(self, cve: CVE) -> Alert:
        """
        Create alert for a critical/high CVE.

        Args:
            cve: CVE that triggered the alert

        Returns:
            Generated alert
        """
        alert_id = f"alert-cve-{cve.cve_id}-{uuid.uuid4().hex[:8]}"

        # Map CVE severity to alert severity
        severity_map = {
            "CRITICAL": AlertSeverity.CRITICAL,
            "HIGH": AlertSeverity.HIGH,
            "MEDIUM": AlertSeverity.MEDIUM,
            "LOW": AlertSeverity.LOW,
        }
        alert_severity = severity_map.get(cve.severity.value, AlertSeverity.MEDIUM)

        # Determine alert type
        alert_type = AlertType.NEW_CRITICAL_CVE if cve.is_critical else AlertType.NEW_HIGH_CVE

        # Create actionable items
        actionable = [
            "Review affected systems in your infrastructure",
            "Check if patches are available",
            "Assess exploitability in your environment",
        ]

        if cve.references:
            actionable.append(f"Review references: {', '.join(cve.references[:2])}")

        return Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=alert_severity,
            title=f"New {cve.severity.value} CVE: {cve.cve_id}",
            description=cve.description[:500],  # Truncate if too long
            created_at=datetime.utcnow(),
            source_entity_type="CVE",
            source_entity_id=cve.cve_id,
            related_cves=[cve.cve_id],
            tags=cve.cwe_ids + cve.affected_vendors,
            actionable_items=actionable,
            confidence_score=0.9 if cve.is_critical else 0.8,
        )

    def create_ioc_alert(self, ioc: IOC) -> Alert:
        """
        Create alert for a high-confidence IOC.

        Args:
            ioc: IOC that triggered the alert

        Returns:
            Generated alert
        """
        alert_id = f"alert-ioc-{ioc.ioc_type.value}-{uuid.uuid4().hex[:8]}"

        # Determine severity based on IOC type and reputation
        if ioc.is_malicious:
            severity = AlertSeverity.HIGH
        elif ioc.is_high_confidence:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW

        # Create actionable items
        actionable = [
            f"Block {ioc.ioc_type.value}: {ioc.value}",
            "Search for this indicator in your logs",
            "Check firewall and IDS/IPS rules",
        ]

        if ioc.related_cves:
            actionable.append(f"Related to CVEs: {', '.join(ioc.related_cves[:3])}")

        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.IOC_DETECTED,
            severity=severity,
            title=f"High Confidence IOC Detected: {ioc.ioc_type.value}",
            description=f"IOC: {ioc.value}\nContext: {ioc.context[:200]}",
            created_at=datetime.utcnow(),
            source_entity_type="IOC",
            source_entity_id=ioc.value,
            related_iocs=[ioc.value],
            related_cves=ioc.related_cves,
            tags=ioc.tags,
            actionable_items=actionable,
            confidence_score=0.9 if ioc.is_high_confidence else 0.7,
        )

    def create_topic_alert(self, topic: Topic) -> Alert:
        """
        Create alert for an emerging threat topic.

        Args:
            topic: Topic that triggered the alert

        Returns:
            Generated alert
        """
        alert_id = f"alert-topic-{topic.topic_id}-{uuid.uuid4().hex[:8]}"

        label = topic.label or f"Topic {topic.topic_number}"
        keywords_str = topic.get_keywords_string(top_n=5)

        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.TOPIC_TRENDING,
            severity=AlertSeverity.MEDIUM,
            title=f"Emerging Threat Topic: {label}",
            description=f"Keywords: {keywords_str}\nDocument count: {topic.document_count}",
            created_at=datetime.utcnow(),
            source_entity_type="Topic",
            source_entity_id=topic.topic_id,
            related_topics=[topic.topic_id],
            related_cves=topic.related_cves,
            tags=topic.top_keywords[:5],
            actionable_items=[
                "Review related threat intelligence documents",
                "Check if this trend affects your infrastructure",
                "Update threat hunting queries",
            ],
            confidence_score=topic.coherence_score or 0.6,
        )

    def prioritize_alerts(self, alerts: list[Alert]) -> list[Alert]:
        """
        Prioritize alerts for display/notification.

        Sorting criteria:
        1. Severity (CRITICAL > HIGH > MEDIUM > LOW)
        2. Confidence score
        3. Age (newer first)

        Args:
            alerts: List of alerts

        Returns:
            Sorted list of alerts
        """
        severity_order = {
            AlertSeverity.CRITICAL: 4,
            AlertSeverity.HIGH: 3,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 1,
            AlertSeverity.INFO: 0,
        }

        def priority_key(alert: Alert) -> tuple[int, float, float]:
            return (
                severity_order.get(alert.severity, 0),
                alert.confidence_score,
                -alert.age_hours,  # Negative for descending order
            )

        return sorted(alerts, key=priority_key, reverse=True)

    def filter_active_alerts(self, alerts: list[Alert]) -> list[Alert]:
        """Get only active (unresolved) alerts."""
        return [alert for alert in alerts if alert.is_active]

    def filter_critical_alerts(self, alerts: list[Alert]) -> list[Alert]:
        """Get only critical alerts."""
        return [alert for alert in alerts if alert.is_critical]

    def calculate_alert_statistics(self, alerts: list[Alert]) -> dict[str, int | float]:
        """
        Calculate statistics about alerts.

        Args:
            alerts: List of alerts

        Returns:
            Dictionary with statistics
        """
        if not alerts:
            return {
                "total": 0,
                "active": 0,
                "critical": 0,
                "high": 0,
                "resolved": 0,
            }

        return {
            "total": len(alerts),
            "active": sum(1 for a in alerts if a.is_active),
            "critical": sum(1 for a in alerts if a.is_critical),
            "high": sum(1 for a in alerts if a.is_high_severity),
            "acknowledged": sum(1 for a in alerts if a.is_acknowledged),
            "resolved": sum(1 for a in alerts if a.is_resolved),
            "avg_age_hours": sum(a.age_hours for a in alerts) / len(alerts),
            "avg_confidence": sum(a.confidence_score for a in alerts) / len(alerts),
        }
