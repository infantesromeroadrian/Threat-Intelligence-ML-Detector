"""
Slack Notifier implementation.

Sends alert notifications to Slack using webhooks.
"""

from __future__ import annotations

import requests

from ....domain.entities import Alert
from ....infrastructure.config.logging_config import get_logger
from ....infrastructure.config.settings import settings

logger = get_logger(__name__)


class SlackNotifier:
    """
    Slack notification adapter.

    Sends formatted alert messages to a Slack channel via webhook.
    """

    def __init__(self) -> None:
        """Initialize Slack notifier."""
        self.webhook_url = settings.slack_webhook_url

        if not self.webhook_url:
            logger.warning(
                "⚠️  Slack webhook URL not configured",
                source="SlackNotifier",
            )

    def send_alert(self, alert: Alert) -> bool:
        """
        Send alert notification to Slack.

        Args:
            alert: Alert entity to send

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.webhook_url:
            logger.warning(
                "⚠️  Cannot send Slack alert - webhook not configured",
                source="SlackNotifier",
            )
            return False

        try:
            message = self._format_alert_message(alert)

            response = requests.post(
                self.webhook_url,
                json=message,
                timeout=10,
            )
            response.raise_for_status()

            logger.info(
                "✅ Sent alert to Slack",
                source="SlackNotifier",
                alert_id=alert.alert_id,
            )

            return True

        except requests.exceptions.RequestException as e:
            logger.error(
                f"❌ Failed to send Slack alert: {e}",
                source="SlackNotifier",
                alert_id=alert.alert_id,
            )
            return False

    def send_summary(self, alerts: list[Alert]) -> bool:
        """
        Send summary of multiple alerts to Slack.

        Args:
            alerts: List of alerts to summarize

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.webhook_url:
            logger.warning(
                "⚠️  Cannot send Slack summary - webhook not configured",
                source="SlackNotifier",
            )
            return False

        if not alerts:
            return True

        try:
            message = self._format_summary_message(alerts)

            response = requests.post(
                self.webhook_url,
                json=message,
                timeout=10,
            )
            response.raise_for_status()

            logger.info(
                "✅ Sent alert summary to Slack",
                source="SlackNotifier",
                count=len(alerts),
            )

            return True

        except requests.exceptions.RequestException as e:
            logger.error(
                f"❌ Failed to send Slack summary: {e}",
                source="SlackNotifier",
            )
            return False

    # =========================================================================
    # Private Methods - Message Formatting
    # =========================================================================

    def _format_alert_message(self, alert: Alert) -> dict[str, object]:
        """Format alert as Slack message with blocks."""
        # Choose color based on severity
        color_map = {
            "CRITICAL": "#FF0000",  # Red
            "HIGH": "#FF6600",  # Orange
            "MEDIUM": "#FFCC00",  # Yellow
            "LOW": "#3366FF",  # Blue
        }
        color = color_map.get(alert.severity.value, "#808080")

        # Choose emoji based on alert type
        emoji_map = {
            "CVE_CRITICAL": "🔴",
            "IOC_DETECTED": "🎯",
            "HIGH_SEVERITY_THREAT": "⚠️",
            "NEW_TOPIC_DISCOVERED": "🔍",
            "RELATED_THREATS": "🔗",
        }
        emoji = emoji_map.get(alert.alert_type.value, "🚨")

        return {
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{emoji} {alert.title}",
                            },
                        },
                        {
                            "type": "section",
                            "fields": [
                                {"type": "mrkdwn", "text": f"*Severity:*\n{alert.severity.value}"},
                                {"type": "mrkdwn", "text": f"*Type:*\n{alert.alert_type.value}"},
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Confidence:*\n{alert.confidence_score:.1%}",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Source:*\n{alert.source_entity_type}",
                                },
                            ],
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Description:*\n{alert.description}",
                            },
                        },
                    ],
                }
            ]
        }

        # Add actionable items if present
        if alert.actionable_items:
            message["attachments"][0]["blocks"].append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Recommended Actions:*\n"
                        + "\n".join(f"• {item}" for item in alert.actionable_items[:5]),
                    },
                }
            )

        return message

    def _format_summary_message(self, alerts: list[Alert]) -> dict[str, object]:
        """Format multiple alerts as summary message."""
        # Count by severity
        severity_counts = {}
        for alert in alerts:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Build summary text
        summary_lines = [f"📊 *Threat Intelligence Summary* ({len(alerts)} alerts)"]
        summary_lines.append("")

        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = severity_counts.get(severity, 0)
            if count > 0:
                emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(
                    severity, "⚪"
                )
                summary_lines.append(f"{emoji} {severity}: {count}")

        # Add top 5 critical alerts
        critical_alerts = [a for a in alerts if a.severity.value == "CRITICAL"][:5]
        if critical_alerts:
            summary_lines.append("")
            summary_lines.append("*Top Critical Alerts:*")
            for alert in critical_alerts:
                summary_lines.append(f"• {alert.title}")

        return {"text": "\n".join(summary_lines)}
