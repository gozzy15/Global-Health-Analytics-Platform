"""
PDF report generation utilities for the Global Health Dashboard.
"""

from datetime import datetime
from io import BytesIO

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
import plotly.express as px


def _plotly_figure_to_image(
    figure,
    width=700,
    height=420,
):
    """
    Convert a Plotly figure into a ReportLab image.
    """

    image_bytes = figure.to_image(
        format="png",
        width=width,
        height=height,
        scale=1,
    )

    image_buffer = BytesIO(
        image_bytes
    )

    image = Image(
        image_buffer,
        width=6.7 * inch,
        height=4.0 * inch,
    )

    return image


def generate_health_pdf_report(
    df: pd.DataFrame,
) -> bytes:
    """
    Generate a PDF summary report from the cleaned
    Global Health Dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned health dataset.

    Returns
    -------
    bytes
        Generated PDF file as bytes.
    """

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=12,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=20,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=8,
    )

    story = []

    # ---------------------------------------------------------
    # Report title
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Global Health Analytics Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Global Health Dataset — Analytical Summary",
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            (
                f"Report generated on "
                f"{datetime.now().strftime('%d %B %Y, %H:%M')}"
            ),
            subtitle_style,
        )
    )

    # ---------------------------------------------------------
    # Dataset overview
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "1. Dataset Overview",
            section_style,
        )
    )

    if df.empty:
        story.append(
            Paragraph(
                "No data is currently available for report generation.",
                body_style,
            )
        )

        document.build(story)

        return buffer.getvalue()

    record_count = len(df)

    country_count = (
        df["country"].nunique()
        if "country" in df.columns
        else 0
    )

    disease_count = (
        df["disease_name"].nunique()
        if "disease_name" in df.columns
        else 0
    )

    if "year" in df.columns:

        min_year = int(df["year"].min())
        max_year = int(df["year"].max())

    else:

        min_year = "N/A"
        max_year = "N/A"

    overview_data = [
        ["Metric", "Value"],
        ["Total records", f"{record_count:,}"],
        ["Countries", f"{country_count:,}"],
        ["Diseases", f"{disease_count:,}"],
        [
            "Year coverage",
            f"{min_year}–{max_year}",
        ],
    ]

    overview_table = Table(
        overview_data,
        colWidths=[
            3.0 * inch,
            2.5 * inch,
        ],
    )

    overview_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#2E4057"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(overview_table)

    # ---------------------------------------------------------
    # Global health indicators
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "2. Global Health Indicators",
            section_style,
        )
    )

    indicator_columns = {
        "incidence_rate_pct": "Incidence Rate (%)",
        "prevalence_rate_pct": "Prevalence Rate (%)",
        "mortality_rate_per_100_people_pct": "Mortality Rate (%)",
        "healthcare_access_pct": "Healthcare Access (%)",
        "recovery_rate_pct": "Recovery Rate (%)",
        "composite_health_index": "Composite Health Index",
    }

    indicator_rows = [
        [
            "Indicator",
            "Mean",
            "Median",
            "Std. Dev.",
        ]
    ]

    for column, label in indicator_columns.items():

        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        indicator_rows.append(
            [
                label,
                f"{values.mean():.2f}",
                f"{values.median():.2f}",
                f"{values.std():.2f}",
            ]
        )

    if len(indicator_rows) > 1:

        indicator_table = Table(
            indicator_rows,
            colWidths=[
                2.7 * inch,
                1.0 * inch,
                1.0 * inch,
                1.0 * inch,
            ],
        )

        indicator_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#2E4057"),
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                ]
            )
        )

        story.append(indicator_table)


    # ---------------------------------------------------------
    # Visual Analysis
    # ---------------------------------------------------------

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "3. Visual Analysis",
            section_style,
        )
    )

    # ---------------------------------------------------------
    # Global CHI Trend
    # ---------------------------------------------------------

    if {
        "year",
        "composite_health_index",
    }.issubset(df.columns):

        chi_trend = (
            df.groupby("year")[
                "composite_health_index"
            ]
            .mean()
            .reset_index()
        )

        if not chi_trend.empty:

            fig_chi_trend = px.line(
                chi_trend,
                x="year",
                y="composite_health_index",
                markers=True,
                title="Global Composite Health Index Trend",
                labels={
                    "year": "Year",
                    "composite_health_index": (
                        "Average Composite Health Index"
                    ),
                },
            )

            fig_chi_trend.update_layout(
                height=420,
                margin=dict(
                    l=50,
                    r=30,
                    t=60,
                    b=50,
                ),
            )

            story.append(
                Paragraph(
                    "Global CHI Trend",
                    section_style,
                )
            )

            story.append(
                _plotly_figure_to_image(
                    fig_chi_trend
                )
            )

    # ---------------------------------------------------------
    # Global Health Indicator Trends
    # ---------------------------------------------------------

    trend_columns = {
        "incidence_rate_pct": "Incidence Rate (%)",
        "prevalence_rate_pct": "Prevalence Rate (%)",
        "mortality_rate_per_100_people_pct": (
            "Mortality Rate (%)"
        ),
        "healthcare_access_pct": (
            "Healthcare Access (%)"
        ),
        "recovery_rate_pct": "Recovery Rate (%)",
    }

    available_trend_columns = [
        column
        for column in trend_columns
        if column in df.columns
    ]

    if (
        "year" in df.columns
        and available_trend_columns
    ):

        trend_data = (
            df.groupby("year")[
                available_trend_columns
            ]
            .mean()
            .reset_index()
        )

        trend_long = trend_data.melt(
            id_vars="year",
            value_vars=available_trend_columns,
            var_name="Indicator",
            value_name="Value",
        )

        trend_long["Indicator"] = (
            trend_long["Indicator"]
            .map(trend_columns)
        )

        fig_indicator_trends = px.line(
            trend_long,
            x="year",
            y="Value",
            color="Indicator",
            markers=False,
            title="Global Health Indicator Trends",
            labels={
                "year": "Year",
                "Value": "Average Value",
                "Indicator": "Indicator",
            },
            color_discrete_map={
                "Incidence Rate (%)": "#1f77b4",
                "Prevalence Rate (%)": "#ff7f0e",
                "Mortality Rate (%)": "#d62728",
                "Healthcare Access (%)": "#2ca02c",
                "Recovery Rate (%)": "#9467bd",
            },
        )

        fig_indicator_trends.update_layout(
            height=450,
            margin=dict(
                l=50,
                r=30,
                t=60,
                b=50,
            ),
        )

        story.append(
            Paragraph(
                "Global Health Indicator Trends",
                section_style,
            )
        )

        story.append(
            _plotly_figure_to_image(
                fig_indicator_trends,
                height=450,
            )
        )

    # ---------------------------------------------------------
    # Country CHI Comparison
    # ---------------------------------------------------------

    if {
        "country",
        "composite_health_index",
    }.issubset(df.columns):

        country_chi = (
            df.groupby("country")[
                "composite_health_index"
            ]
            .mean()
            .dropna()
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        if not country_chi.empty:

            fig_country_chi = px.bar(
                country_chi,
                x="composite_health_index",
                y="country",
                orientation="h",
                title="Average Composite Health Index by Country",
                labels={
                    "composite_health_index": (
                        "Average CHI"
                    ),
                    "country": "Country",
                },
                text="composite_health_index",
            )

            fig_country_chi.update_traces(
                texttemplate="%{x:.2f}",
                textposition="outside",
                cliponaxis=False,
            )

            fig_country_chi.update_layout(
                height=600,
                yaxis={
                    "categoryorder": (
                        "total ascending"
                    ),
                    "title": None,
                },
                margin=dict(
                    l=120,
                    r=30,
                    t=60,
                    b=50,
                ),
            )

            story.append(
                Paragraph(
                    "Country CHI Comparison",
                    section_style,
                )
            )

            story.append(
                _plotly_figure_to_image(
                    fig_country_chi,
                    height=600,
                )
            )

    # ---------------------------------------------------------
    # CHI Correlation Analysis
    # ---------------------------------------------------------

    if "composite_health_index" in df.columns:

        correlation_columns = [
            column
            for column in [
                "incidence_rate_pct",
                "prevalence_rate_pct",
                "mortality_rate_per_100_people_pct",
                "healthcare_access_pct",
                "doctors_per_1000",
                "hospital_beds_per_1000",
                "recovery_rate_pct",
                "per_capita_income_usd",
                "education_index",
                "urbanization_rate_pct",
                "dalys",
            ]
            if column in df.columns
        ]

        correlation_rows = []

        for column in correlation_columns:

            pair = df[
                [
                    column,
                    "composite_health_index",
                ]
            ].apply(
                pd.to_numeric,
                errors="coerce",
            ).dropna()

            if len(pair) < 2:
                continue

            correlation = pair[
                column
            ].corr(
                pair[
                    "composite_health_index"
                ]
            )

            correlation_labels = {
                "incidence_rate_pct": "Incidence Rate",
                "prevalence_rate_pct": "Prevalence Rate",
                "mortality_rate_per_100_people_pct": (
                    "Mortality Rate"
                ),
                "healthcare_access_pct": (
                    "Healthcare Access"
                ),
                "doctors_per_1000": (
                    "Doctors per 1,000"
                ),
                "hospital_beds_per_1000": (
                    "Hosp. Beds per 1,000"
                ),
                "recovery_rate_pct": (
                    "Recovery Rate"
                ),
                "per_capita_income_usd": (
                    "Per Capita Income"
                ),
                "education_index": (
                    "Education Index"
                ),
                "urbanization_rate_pct": (
                    "Urbanization Rate"
                ),
                "dalys": "DALYs",
            }

            correlation_rows.append(
                {
                    "Variable": correlation_labels.get(
                        column,
                        column.replace("_", " ").title(),
                    ),
                    "Correlation": correlation,
                }
            )

        correlation_data = pd.DataFrame(
            correlation_rows
        )

        if not correlation_data.empty:

            correlation_data[
                "Absolute Correlation"
            ] = correlation_data[
                "Correlation"
            ].abs()

            correlation_data = (
                correlation_data
                .sort_values(
                    "Absolute Correlation",
                    ascending=True,
                )
            )

            fig_correlation = px.bar(
                correlation_data,
                x="Correlation",
                y="Variable",
                orientation="h",
                title="Correlation with Composite Health Index",
                labels={
                    "Correlation": (
                        "Pearson Correlation"
                    ),
                    "Variable": "Variable",
                },
                text="Correlation",
            )

            fig_correlation.update_traces(
                texttemplate="%{x:+.3f}",
                textposition="outside",
                cliponaxis=False,
            )

            fig_correlation.update_layout(
                height=500,
                margin=dict(
                    l=130,
                    r=80,
                    t=60,
                    b=50,
                ),
                xaxis={
                    "range": [-1.1, 1.1],
                },
                yaxis={
                    "title": None,
                },
            )

            story.append(
                Paragraph(
                    "CHI Correlation Analysis",
                    section_style,
                )
            )

            story.append(
                _plotly_figure_to_image(
                    fig_correlation,
                    height=500,
                )
            )


    # ---------------------------------------------------------
    # CHI summary
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "4. Composite Health Index Summary",
            section_style,
        )
    )

    if "composite_health_index" in df.columns:

        chi_values = pd.to_numeric(
            df["composite_health_index"],
            errors="coerce",
        ).dropna()

        if not chi_values.empty:

            story.append(
                Paragraph(
                    (
                        f"The Composite Health Index (CHI) has an "
                        f"overall mean of "
                        f"<b>{chi_values.mean():.2f}</b>, with a "
                        f"median of <b>{chi_values.median():.2f}</b>. "
                        f"The observed range extends from "
                        f"<b>{chi_values.min():.2f}</b> to "
                        f"<b>{chi_values.max():.2f}</b>."
                    ),
                    body_style,
                )
            )

    # ---------------------------------------------------------
    # Country summary
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "5. Country Summary",
            section_style,
        )
    )

    if {
        "country",
        "composite_health_index",
    }.issubset(df.columns):

        country_chi = (
            df.groupby("country")[
                "composite_health_index"
            ]
            .mean()
            .dropna()
            .sort_values(ascending=False)
        )

        if not country_chi.empty:

            top_countries = country_chi.head(5)

            country_rows = [
                [
                    "Country",
                    "Average CHI",
                ]
            ]

            for country, value in top_countries.items():

                country_rows.append(
                    [
                        str(country),
                        f"{value:.2f}",
                    ]
                )

            country_table = Table(
                country_rows,
                colWidths=[
                    3.5 * inch,
                    2.0 * inch,
                ],
            )

            country_table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor("#2E4057"),
                        ),
                        (
                            "TEXTCOLOR",
                            (0, 0),
                            (-1, 0),
                            colors.white,
                        ),
                        (
                            "FONTNAME",
                            (0, 0),
                            (-1, 0),
                            "Helvetica-Bold",
                        ),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.grey,
                        ),
                        (
                            "FONTSIZE",
                            (0, 0),
                            (-1, -1),
                            9,
                        ),
                        (
                            "TOPPADDING",
                            (0, 0),
                            (-1, -1),
                            5,
                        ),
                        (
                            "BOTTOMPADDING",
                            (0, 0),
                            (-1, -1),
                            5,
                        ),
                    ]
                )
            )

            story.append(
                Paragraph(
                    "Top five countries by average CHI:",
                    body_style,
                )
            )

            story.append(country_table)

    # ---------------------------------------------------------
    # Methodology note
    # ---------------------------------------------------------

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "6. Report Methodology",
            section_style,
        )
    )

    story.append(
        Paragraph(
            """
            This report summarizes analytical outputs generated
            from the cleaned Global Health Dataset. Descriptive
            statistics are calculated from available numeric
            observations. Country-level summaries use the average
            Composite Health Index for each country. Additional
            analytical sections can be incorporated as the
            reporting framework is expanded.
            """,
            body_style,
        )
    )

    story.append(
        Paragraph(
            "7. Notes",
            section_style,
        )
    )

    story.append(
        Paragraph(
            """
            The Global Health Dataset is used for analytical and
            demonstration purposes. Report values should therefore
            be interpreted within the context of the dataset and
            its underlying methodology.
            """,
            body_style,
        )
    )

    # ---------------------------------------------------------
    # Build PDF
    # ---------------------------------------------------------

    document.build(story)

    return buffer.getvalue()