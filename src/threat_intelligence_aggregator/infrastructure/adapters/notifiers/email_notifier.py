"""
Email Notifier implementation.

Sends alert notifications via email using SMTP.
"""

from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from ....domain.entities import Alert
from ....infrastructure.config.logging_config import get_logger
from ....infrastructure.config.settings import settings

logger = get_logger(__name__)


class EmailNotifier:
    """
    Email notification adapter.

    Sends formatted alert emails via SMTP.
    """

    def __init__(self) -> None:
        """Initialize Email notifier."""
        self.smtp_host = settings.smtp_host
        self.smtp_port = settings.smtp_port
        self.smtp_user = settings.smtp_user
        self.smtp_password = settings.smtp_password
        self.from_email = settings.smtp_from
        self.to_email = settings.smtp_to

        if not all([self.smtp_host, self.smtp_user, self.smtp_password, self.from_email]):
            logger.warning(
                "⚠️  Email configuration incomplete",
                source="EmailNotifier",
            )

    def send_alert(self, alert: Alert) -> bool:
        """
        Send alert notification via email.

        Args:
            alert: Alert entity to send

        Returns:
            True if sent successfully, False otherwise
        """
        if not all([self.smtp_host, self.smtp_user, self.smtp_password]):
            logger.warning(
                "⚠️  Cannot send email alert - SMTP not configured",
                source="EmailNotifier",
            )
            return False

        try:
            msg = self._format_alert_email(alert)

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(
                "✅ Sent alert email",
                source="EmailNotifier",
                alert_id=alert.alert_id,
            )

            return True

        except Exception as e:
            logger.error(
                f"❌ Failed to send email alert: {e}",
                source="EmailNotifier",
                alert_id=alert.alert_id,
            )
            return False

    def send_summary(self, alerts: list[Alert]) -> bool:
        """
        Send summary of multiple alerts via email.

        Args:
            alerts: List of alerts to summarize

        Returns:
            True if sent successfully, False otherwise
        """
        if not all([self.smtp_host, self.smtp_user, self.smtp_password]):
            logger.warning(
                "⚠️  Cannot send email summary - SMTP not configured",
                source="EmailNotifier",
            )
            return False

        if not alerts:
            return True

        try:
            msg = self._format_summary_email(alerts)

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(
                "✅ Sent alert summary email",
                source="EmailNotifier",
                count=len(alerts),
            )

            return True

        except Exception as e:
            logger.error(
                f"❌ Failed to send summary email: {e}",
                source="EmailNotifier",
            )
            return False

    # =========================================================================
    # Private Methods - Email Formatting
    # =========================================================================

    def _format_alert_email(self, alert: Alert) -> MIMEMultipart:
        """Format alert as email message."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{alert.severity.value}] {alert.title}"
        msg["From"] = self.from_email
        msg["To"] = self.to_email or self.from_email

        # Plain text version
        text_body = f"""
Threat Intelligence Alert

Title: {alert.title}
Severity: {alert.severity.value}
Type: {alert.alert_type.value}
Confidence: {alert.confidence_score:.1%}

Description:
{alert.description}

Source: {alert.source_entity_type} ({alert.source_entity_id})
Created: {alert.created_at}

"""
        if alert.actionable_items:
            text_body += "\nRecommended Actions:\n"
            for item in alert.actionable_items:
                text_body += f"  - {item}\n"

        if alert.related_cves:
            text_body += f"\nRelated CVEs: {', '.join(alert.related_cves)}\n"

        # HTML version
        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif;">
    <div style="background-color: #f0f0f0; padding: 20px;">
        <h2 style="color: {self._get_severity_color(alert.severity.value)};">
            🚨 Threat Intelligence Alert
        </h2>
        
        <div style="background-color: white; padding: 15px; border-radius: 5px;">
            <h3>{alert.title}</h3>
            
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 5px;"><strong>Severity:</strong></td>
                    <td style="padding: 5px; color: {self._get_severity_color(alert.severity.value)};">
                        {alert.severity.value}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><strong>Type:</strong></td>
                    <td style="padding: 5px;">{alert.alert_type.value}</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><strong>Confidence:</strong></td>
                    <td style="padding: 5px;">{alert.confidence_score:.1%}</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><strong>Created:</strong></td>
                    <td style="padding: 5px;">{alert.created_at}</td>
                </tr>
            </table>
            
            <h4>Description:</h4>
            <p>{alert.description}</p>
"""

        if alert.actionable_items:
            html_body += "<h4>Recommended Actions:</h4><ul>"
            for item in alert.actionable_items:
                html_body += f"<li>{item}</li>"
            html_body += "</ul>"

        if alert.related_cves:
            html_body += f"<p><strong>Related CVEs:</strong> {', '.join(alert.related_cves)}</p>"

        html_body += """
        </div>
    </div>
</body>
</html>
"""

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        return msg

    def _format_summary_email(self, alerts: list[Alert]) -> MIMEMultipart:
        """Format multiple alerts as summary email."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Threat Intelligence Summary ({len(alerts)} alerts)"
        msg["From"] = self.from_email
        msg["To"] = self.to_email or self.from_email

        # Count by severity
        severity_counts = {}
        for alert in alerts:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Plain text
        text_body = f"Threat Intelligence Summary\n\n"
        text_body += f"Total Alerts: {len(alerts)}\n\n"

        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = severity_counts.get(severity, 0)
            if count > 0:
                text_body += f"{severity}: {count}\n"

        # HTML
        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif;">
    <h2>📊 Threat Intelligence Summary</h2>
    <p><strong>Total Alerts:</strong> {len(alerts)}</p>
    
    <table style="border-collapse: collapse; width: 100%;">
        <tr style="background-color: #f0f0f0;">
            <th style="padding: 10px; text-align: left;">Severity</th>
            <th style="padding: 10px; text-align: right;">Count</th>
        </tr>
"""

        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = severity_counts.get(severity, 0)
            if count > 0:
                color = self._get_severity_color(severity)
                html_body += f"""
        <tr>
            <td style="padding: 10px; color: {color};"><strong>{severity}</strong></td>
            <td style="padding: 10px; text-align: right;">{count}</td>
        </tr>
"""

        html_body += """
    </table>
</body>
</html>
"""

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        return msg

    def _get_severity_color(self, severity: str) -> str:
        """Get HTML color for severity level."""
        colors = {
            "CRITICAL": "#DC143C",  # Crimson
            "HIGH": "#FF6600",  # Orange
            "MEDIUM": "#FFD700",  # Gold
            "LOW": "#4169E1",  # Royal Blue
        }
        return colors.get(severity, "#808080")
