"""
Utilities for emailing generated health reports.
"""

import os
import smtplib
from email.message import EmailMessage


def send_report_email(
    recipient_email: str,
    subject: str,
    body: str,
    attachment_data: bytes,
    attachment_filename: str,
    attachment_mime_type: str,
    sender_email: str | None = None,
    smtp_host: str | None = None,
    smtp_port: str | int | None = None,
    smtp_password: str | None = None,
) -> None:
    """
    Send a generated report as an email attachment.

    SMTP configuration can be supplied directly or read from
    environment variables when values are not provided.

    Default environment variables:

    SMTP_HOST
    SMTP_PORT
    SMTP_USERNAME
    SMTP_PASSWORD

    SMTP_PORT defaults to 587 when neither a supplied value
    nor the SMTP_PORT environment variable is available.

    Parameters
    ----------
    recipient_email : str
        Email address of the recipient.

    subject : str
        Email subject.

    body : str
        Email body.

    attachment_data : bytes
        Report file contents.

    attachment_filename : str
        Name of the attached report.

    attachment_mime_type : str
        MIME type of the attachment.

    sender_email : str | None, optional
        Email address used as both the SMTP authentication
        username and the displayed sender. If not provided,
        SMTP_USERNAME from the environment is used.

    smtp_host : str | None, optional
        SMTP server hostname. If not provided, SMTP_HOST from
        the environment is used.

    smtp_port : str | int | None, optional
        SMTP server port. If not provided, SMTP_PORT from the
        environment is used. Defaults to 587 if unavailable.

    smtp_password : str | None, optional
        SMTP password. If not provided, SMTP_PASSWORD from
        the environment is used.

    Raises
    ------
    ValueError
        If required SMTP configuration is incomplete.

    smtplib.SMTPException
        If the email cannot be sent.
    """

    # ---------------------------------------------------------
    # Resolve SMTP configuration
    # ---------------------------------------------------------

    smtp_host = (
        smtp_host.strip()
        if isinstance(smtp_host, str) and smtp_host.strip()
        else os.getenv("SMTP_HOST")
    )

    smtp_port = (
        str(smtp_port).strip()
        if smtp_port is not None
        and str(smtp_port).strip()
        else os.getenv("SMTP_PORT", "587")
    )

    smtp_username = (
        sender_email.strip()
        if isinstance(sender_email, str)
        and sender_email.strip()
        else os.getenv("SMTP_USERNAME")
    )

    smtp_password = (
        smtp_password
        if smtp_password
        else os.getenv("SMTP_PASSWORD")
    )

    sender_email = smtp_username

    # ---------------------------------------------------------
    # Validate required configuration
    # ---------------------------------------------------------

    if not smtp_host:
        raise ValueError(
            "SMTP host is not configured. "
            "Please provide an SMTP host or set SMTP_HOST."
        )

    if not smtp_username:
        raise ValueError(
            "SMTP username is not configured. "
            "Please set SMTP_USERNAME."
        )

    if not smtp_password:
        raise ValueError(
            "SMTP password is not configured. "
            "Please provide an SMTP password or set SMTP_PASSWORD."
        )

    try:

        smtp_port = int(smtp_port)

    except (TypeError, ValueError):

        raise ValueError(
            "SMTP port must be a valid number."
        )

    # ---------------------------------------------------------
    # Build email message
    # ---------------------------------------------------------

    message = EmailMessage()

    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject

    message.set_content(
        body
    )

    maintype, subtype = (
        attachment_mime_type.split(
            "/",
            1,
        )
    )

    message.add_attachment(
        attachment_data,
        maintype=maintype,
        subtype=subtype,
        filename=attachment_filename,
    )

    # ---------------------------------------------------------
    # Connect to SMTP server and send email
    # ---------------------------------------------------------

    with smtplib.SMTP(
        smtp_host,
        smtp_port,
    ) as server:

        server.starttls()

        server.login(
            smtp_username,
            smtp_password,
        )

        server.send_message(
            message
        )