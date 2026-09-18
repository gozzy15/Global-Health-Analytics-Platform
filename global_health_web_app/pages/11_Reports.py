"""
Reports page for the Global Health Dashboard.
"""

import streamlit as st
from dotenv import load_dotenv

from utils.data_loader import load_health_data
from utils.email_report import send_report_email
from utils.html_report import (
    generate_interactive_html_report,
)
from utils.pdf_report import generate_health_pdf_report


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()

from utils.style import apply_global_styles

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Reports | Global Health Dashboard",
    page_icon="📄",
    layout="wide",
)

apply_global_styles()

# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------

st.title("📄 Reports")

st.markdown(
    """
    Generate professional PDF reports from the Global Health
    Dataset and download them directly from the dashboard.
    """
)

st.divider()


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = load_health_data()

# ---------------------------------------------------------
# Report cache
# ---------------------------------------------------------

if "cached_pdf_report" not in st.session_state:

    st.session_state.cached_pdf_report = None


if "cached_html_report" not in st.session_state:

    st.session_state.cached_html_report = None


if "cached_report_dataset_signature" not in st.session_state:

    st.session_state.cached_report_dataset_signature = None


current_dataset_signature = (
    len(df),
    tuple(df.columns),
    df["year"].min(),
    df["year"].max(),
)


if (
    st.session_state.cached_report_dataset_signature
    != current_dataset_signature
):

    st.session_state.cached_pdf_report = None
    st.session_state.cached_html_report = None

    st.session_state.cached_report_dataset_signature = (
        current_dataset_signature
    )


# ---------------------------------------------------------
# Dataset status
# ---------------------------------------------------------

st.subheader("Report Data")

st.info(
    f"""
    **Records:** {len(df):,}

    **Countries:** {df["country"].nunique():,}

    **Diseases:** {df["disease_name"].nunique():,}

    **Year range:** {int(df["year"].min())}–{int(df["year"].max())}
    """
)


# ---------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------

st.subheader("PDF Report")

st.markdown(
    """
    Generate a professional PDF report containing dataset
    information, analytical tables, health trends, Composite
    Health Index analysis, country comparisons, and
    correlation visualizations.
    """
)


if st.button(
    "📊 Generate PDF with Charts",
    type="primary",
    use_container_width=True,
):

    try:

        with st.spinner(
            "Generating PDF report with charts..."
        ):

            st.session_state.cached_pdf_report = (
                generate_health_pdf_report(
                    df
                )
            )

        st.success(
            "✅ PDF report with charts generated successfully "
            "and cached for reuse."
        )

    except Exception as error:

        st.error(
            "❌ The PDF report could not be generated. "
            f"Error: {error}"
        )


if st.session_state.cached_pdf_report is not None:

    st.download_button(
        label="⬇️ Download PDF with Charts",
        data=st.session_state.cached_pdf_report,
        file_name=(
            "global_health_report_with_charts.pdf"
        ),
        mime="application/pdf",
        use_container_width=True,
    )

    st.caption(
        "A generated PDF report is currently cached and "
        "will be reused for subsequent downloads and "
        "email delivery."
    )


# ---------------------------------------------------------
# Interactive HTML report generation
# ---------------------------------------------------------

st.divider()

st.subheader(
    "Interactive HTML Report"
)

st.markdown(
    """
    Generate a standalone interactive HTML report
    containing dataset information, statistical
    summaries, interactive health trends, Composite
    Health Index analysis, country comparisons, and
    correlation visualizations.

    The generated report can be opened directly in a
    web browser and its Plotly charts remain interactive.
    """
)


if st.button(
    "🌐 Generate Interactive HTML Report",
    type="primary",
    use_container_width=True,
):

    try:

        with st.spinner(
            "Generating interactive HTML report..."
        ):

            st.session_state.cached_html_report = (
                generate_interactive_html_report(
                    df
                )
            )

        st.success(
            "✅ Interactive HTML report generated successfully "
            "and cached for reuse."
        )

    except Exception as error:

        st.error(
            "❌ The interactive HTML report could not be generated. "
            f"Error: {error}"
        )


if st.session_state.cached_html_report is not None:

    st.download_button(
        label="⬇️ Download Interactive HTML Report",
        data=st.session_state.cached_html_report,
        file_name=(
            "global_health_interactive_report.html"
        ),
        mime="text/html",
        use_container_width=True,
    )

    st.caption(
        "A generated HTML report is currently cached and "
        "will be reused for subsequent downloads and "
        "email delivery."
    )


# ---------------------------------------------------------
# Email report generation
# ---------------------------------------------------------

st.divider()

st.subheader(
    "📧 Email Reports"
)

st.markdown(
    """
    Send a generated report directly to an email address.
    Choose the report format, provide the recipient's email address, 
    and optionally customize the sender and SMTP settings before sending the report as an attachment.
    """
)


col1, col2 = st.columns(2)


with col1:

    recipient_email = st.text_input(
        "Recipient Email Address",
        placeholder="recipient@example.com",
    )

    sender_email = st.text_input(
        "SMTP Username / Sender Email",
        placeholder=(
            "e.g. youraccount@gmail.com"
        ),
        help=(
            "This email address is used to authenticate "
            "with the SMTP server and is also shown as "
            "the sender. Leave blank to use the default "
            "SMTP username."
        ),
    )

    smtp_host = st.text_input(
        "SMTP Host",
        placeholder=(
            "Leave blank to use the default SMTP host"
        ),
    )

    smtp_port = st.text_input(
        "SMTP Port",
        placeholder=(
            "Leave blank to use the default port (587)"
        ),
    )


with col2:

    smtp_password = st.text_input(
        "SMTP Password / App Password",
        type="password",
        placeholder=(
            "Leave blank to use the default SMTP password"
        ),
        help=(
            "For Gmail, use the App Password associated "
            "with the SMTP username above, not your normal "
            "Gmail account password."
        ),
    )

    report_format = st.selectbox(
        "Report Format",
        options=[
            "PDF with Charts",
            "Interactive HTML",
        ],
    )

    email_subject = st.text_input(
        "Email Subject",
        value=(
            "Global Health Dashboard Report"
        ),
    )

    email_body = st.text_area(
        "Email Message",
        value=(
            "Please find the requested Global Health "
            "Dashboard report attached."
        ),
        height=120,
    )


if st.button(
    "📧 Send Report by Email",
    type="primary",
    use_container_width=True,
):

    recipient = recipient_email.strip()

    # -----------------------------------------------------
    # Validate recipient email
    # -----------------------------------------------------

    if not recipient:

        st.warning(
            "Please enter a recipient email address."
        )

    elif (
        "@" not in recipient
        or "." not in recipient.split("@")[-1]
    ):

        st.warning(
            "Please enter a valid recipient email address."
        )

    else:

        # -------------------------------------------------
        # Retrieve or generate the selected report
        # -------------------------------------------------

        if report_format == "PDF with Charts":

            attachment_data = (
                st.session_state.cached_pdf_report
            )

            attachment_filename = (
                "global_health_report_with_charts.pdf"
            )

            attachment_mime_type = (
                "application/pdf"
            )

            if attachment_data is None:

                with st.spinner(
                    "Generating PDF report before sending..."
                ):

                    try:

                        attachment_data = (
                            generate_health_pdf_report(
                                df
                            )
                        )

                        st.session_state.cached_pdf_report = (
                            attachment_data
                        )

                    except Exception as error:

                        st.error(
                            "❌ The PDF report could not be generated. "
                            f"Error: {error}"
                        )

                        attachment_data = None


        else:

            attachment_data = (
                st.session_state.cached_html_report
            )

            attachment_filename = (
                "global_health_interactive_report.html"
            )

            attachment_mime_type = (
                "text/html"
            )

            if attachment_data is None:

                with st.spinner(
                    "Generating interactive HTML report before sending..."
                ):

                    try:

                        attachment_data = (
                            generate_interactive_html_report(
                                df
                            )
                        )

                        st.session_state.cached_html_report = (
                            attachment_data
                        )

                    except Exception as error:

                        st.error(
                            "❌ The interactive HTML report could not be generated. "
                            f"Error: {error}"
                        )

                        attachment_data = None

        # -------------------------------------------------
        # Send the generated report
        # -------------------------------------------------

        if attachment_data is not None:

            with st.spinner(
                "Sending report by email..."
            ):

                try:

                    send_report_email(
                        recipient_email=recipient,
                        subject=email_subject,
                        body=email_body,
                        attachment_data=attachment_data,
                        attachment_filename=(
                            attachment_filename
                        ),
                        attachment_mime_type=(
                            attachment_mime_type
                        ),
                        sender_email=sender_email,
                        smtp_host=smtp_host,
                        smtp_port=smtp_port,
                        smtp_password=smtp_password,
                    )

                    st.success(
                        "✅ Report sent successfully."
                    )

                except ValueError as error:

                    st.error(
                        "❌ Email configuration error. "
                        f"Please check your email settings. "
                        f"Details: {error}"
                    )

                except Exception as error:

                    st.error(
                        "❌ The report could not be sent. "
                        f"Error: {error}"
                    )