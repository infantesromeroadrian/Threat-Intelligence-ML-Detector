"""
Notifier ports (interfaces) for sending notifications.

These are abstract interfaces that infrastructure adapters must implement.
"""

from __future__ import annotations

from typing import Protocol

from ..entities import Alert


class NotificationPort(Protocol):
    """Base interface for notification services."""

    def send_alert(self, alert: Alert) -> bool:
        """
        Send an alert notification.

        Args:
            alert: Alert entity to send

        Returns:
            True if notification was sent successfully, False otherwise
        """
        ...

    def send_alert_batch(self, alerts: list[Alert]) -> int:
        """
        Send multiple alert notifications.

        Args:
            alerts: List of alert entities

        Returns:
            Number of successfully sent notifications
        """
        ...

    def is_configured(self) -> bool:
        """Check if notifier is properly configured."""
        ...

    def get_notifier_name(self) -> str:
        """Get the name of the notifier implementation."""
        ...


class SlackNotifierPort(Protocol):
    """Interface for Slack notifications."""

    def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack channel."""
        ...

    def send_message(self, message: str, channel: str | None = None) -> bool:
        """
        Send custom message to Slack.

        Args:
            message: Message text
            channel: Optional channel override

        Returns:
            True if sent successfully
        """
        ...

    def send_formatted_alert(self, alert: Alert, include_actions: bool = True) -> bool:
        """
        Send formatted alert with Slack blocks.

        Args:
            alert: Alert to send
            include_actions: Include action buttons

        Returns:
            True if sent successfully
        """
        ...

    def is_configured(self) -> bool:
        """Check if Slack webhook is configured."""
        ...


class EmailNotifierPort(Protocol):
    """Interface for email notifications."""

    def send_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        ...

    def send_email(
        self,
        to: str | list[str],
        subject: str,
        body: str,
        is_html: bool = False,
    ) -> bool:
        """
        Send custom email.

        Args:
            to: Recipient email(s)
            subject: Email subject
            body: Email body
            is_html: Whether body is HTML

        Returns:
            True if sent successfully
        """
        ...

    def send_digest(self, alerts: list[Alert], to: str | list[str]) -> bool:
        """
        Send digest email with multiple alerts.

        Args:
            alerts: List of alerts
            to: Recipient email(s)

        Returns:
            True if sent successfully
        """
        ...

    def is_configured(self) -> bool:
        """Check if email settings are configured."""
        ...
