"""
Interactive HTML report generator for the Global Health Dashboard.
"""

from datetime import datetime

import pandas as pd
import plotly.express as px


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------


def _figure_to_html(
    figure,
    include_plotlyjs=False,
):
    """
    Convert a Plotly figure into an embeddable HTML fragment.
    """

    return figure.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={
            "displayModeBar": True,
            "responsive": True,
        },
    )


def _format_number(value, decimals=2):
    """
    Format a numeric value safely for the report.
    """

    if pd.isna(value):
        return "N/A"

    return f"{value:,.{decimals}f}"


# ---------------------------------------------------------
# Main report generator
# ---------------------------------------------------------


def generate_interactive_html_report(
    df: pd.DataFrame,
) -> bytes:
    """
    Generate a standalone interactive HTML report.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned Global Health Dataset.

    Returns
    -------
    bytes
        Complete standalone HTML document.
    """

    report_date = datetime.now().strftime(
        "%B %d, %Y"
    )

    # -----------------------------------------------------
    # Dataset overview
    # -----------------------------------------------------

    records = len(df)

    countries = df["country"].nunique()

    diseases = df["disease_name"].nunique()

    min_year = int(df["year"].min())

    max_year = int(df["year"].max())

    # -----------------------------------------------------
    # Global summary statistics
    # -----------------------------------------------------

    indicator_columns = {
        "Incidence Rate (%)": "incidence_rate_pct",
        "Prevalence Rate (%)": "prevalence_rate_pct",
        "Mortality Rate (%)": (
            "mortality_rate_per_100_people_pct"
        ),
        "Healthcare Access (%)": (
            "healthcare_access_pct"
        ),
        "Recovery Rate (%)": "recovery_rate_pct",
        "Composite Health Index": (
            "composite_health_index"
        ),
    }

    summary_rows = []

    for label, column in indicator_columns.items():

        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        summary_rows.append(
            {
                "Indicator": label,
                "Mean": values.mean(),
                "Median": values.median(),
                "Std. Deviation": values.std(),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # -----------------------------------------------------
    # Global CHI trend
    # -----------------------------------------------------

    chi_trend = (
        df.groupby("year")[
            "composite_health_index"
        ]
        .mean()
        .reset_index()
    )

    fig_chi_trend = px.line(
        chi_trend,
        x="year",
        y="composite_health_index",
        markers=True,
        title=(
            "Global Composite Health Index Trend"
        ),
        labels={
            "year": "Year",
            "composite_health_index": (
                "Average CHI"
            ),
        },
    )

    fig_chi_trend.update_layout(
        hovermode="x unified",
        height=500,
    )

    # -----------------------------------------------------
    # Global health indicator trends
    # -----------------------------------------------------

    trend_columns = {
        "Incidence Rate (%)": (
            "incidence_rate_pct"
        ),
        "Prevalence Rate (%)": (
            "prevalence_rate_pct"
        ),
        "Mortality Rate (%)": (
            "mortality_rate_per_100_people_pct"
        ),
        "Healthcare Access (%)": (
            "healthcare_access_pct"
        ),
        "Recovery Rate (%)": (
            "recovery_rate_pct"
        ),
    }

    trend_frames = []

    for label, column in trend_columns.items():

        if column not in df.columns:
            continue

        temp = (
            df.groupby("year")[column]
            .mean()
            .reset_index()
        )

        temp["Indicator"] = label

        temp = temp.rename(
            columns={
                column: "Value"
            }
        )

        trend_frames.append(temp)

    if trend_frames:

        trend_long = pd.concat(
            trend_frames,
            ignore_index=True,
        )

        
        fig_indicator_trends = px.line(
            trend_long,
            x="year",
            y="Value",
            color="Indicator",
            title=(
                "Global Health Indicator Trends"
            ),
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
            hovermode="x unified",
            height=550,
        )

    else:

        fig_indicator_trends = None

    # -----------------------------------------------------
    # Country CHI comparison
    # -----------------------------------------------------

    country_chi = (
        df.groupby("country")[
            "composite_health_index"
        ]
        .mean()
        .reset_index()
        .sort_values(
            "composite_health_index",
            ascending=True,
        )
    )

    fig_country_chi = px.bar(
        country_chi,
        x="composite_health_index",
        y="country",
        orientation="h",
        title=(
            "Average Composite Health Index by Country"
        ),
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
        height=650,
        margin=dict(
            l=120,
            r=80,
            t=60,
            b=50,
        ),
        yaxis={
            "title": None,
        },
    )

    # -----------------------------------------------------
    # CHI correlation analysis
    # -----------------------------------------------------

    correlation_columns = {
        "Incidence Rate": (
            "incidence_rate_pct"
        ),
        "Prevalence Rate": (
            "prevalence_rate_pct"
        ),
        "Mortality Rate": (
            "mortality_rate_per_100_people_pct"
        ),
        "Healthcare Access": (
            "healthcare_access_pct"
        ),
        "Doctors per 1,000": (
            "doctors_per_1000"
        ),
        "Hosp. Beds per 1,000": (
            "hospital_beds_per_1000"
        ),
        "Recovery Rate": (
            "recovery_rate_pct"
        ),
        "Per Capita Income": (
            "per_capita_income_usd"
        ),
        "Education Index": (
            "education_index"
        ),
        "Urbanization Rate": (
            "urbanization_rate_pct"
        ),
        "DALYs": "dalys",
    }

    correlation_rows = []

    for label, column in correlation_columns.items():

        if column not in df.columns:
            continue

        valid_data = df[
            [
                column,
                "composite_health_index",
            ]
        ].dropna()

        if len(valid_data) < 2:
            continue

        correlation = valid_data[
            column
        ].corr(
            valid_data[
                "composite_health_index"
            ]
        )

        correlation_rows.append(
            {
                "Variable": label,
                "Correlation": correlation,
            }
        )

    correlation_data = pd.DataFrame(
        correlation_rows
    )

    if not correlation_data.empty:

        correlation_data = (
            correlation_data
            .sort_values(
                "Correlation"
            )
        )

        fig_correlation = px.bar(
            correlation_data,
            x="Correlation",
            y="Variable",
            orientation="h",
            title=(
                "Correlation with Composite Health Index"
            ),
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
            height=650,
            margin=dict(
                l=150,
                r=100,
                t=60,
                b=50,
            ),
            xaxis={
                "range": [-1.1, 1.1],
                "zeroline": True,
            },
            yaxis={
                "title": None,
            },
        )

    else:

        fig_correlation = None

    # -----------------------------------------------------
    # Top countries
    # -----------------------------------------------------

    top_countries = (
        country_chi
        .sort_values(
            "composite_health_index",
            ascending=False,
        )
        .head(5)
        .copy()
    )

    top_countries[
        "composite_health_index"
    ] = top_countries[
        "composite_health_index"
    ].round(2)

    # -----------------------------------------------------
    # Build summary table HTML
    # -----------------------------------------------------

    if not summary_df.empty:

        summary_table = summary_df.copy()

        summary_table[
            "Mean"
        ] = summary_table[
            "Mean"
        ].map(
            lambda x: _format_number(x)
        )

        summary_table[
            "Median"
        ] = summary_table[
            "Median"
        ].map(
            lambda x: _format_number(x)
        )

        summary_table[
            "Std. Deviation"
        ] = summary_table[
            "Std. Deviation"
        ].map(
            lambda x: _format_number(x)
        )

        summary_table_html = (
            summary_table
            .to_html(
                index=False,
                classes="data-table",
                border=0,
            )
        )

    else:

        summary_table_html = (
            "<p>No summary statistics available.</p>"
        )

    # -----------------------------------------------------
    # Build top-country table
    # -----------------------------------------------------

    top_country_html = (
        top_countries
        .rename(
            columns={
                "country": "Country",
                "composite_health_index": (
                    "Average CHI"
                ),
            }
        )
        .to_html(
            index=False,
            classes="data-table",
            border=0,
        )
    )

    # -----------------------------------------------------
    # Plotly HTML fragments
    # -----------------------------------------------------

    chi_trend_html = _figure_to_html(
        fig_chi_trend,
        include_plotlyjs=True,
    )

    if fig_indicator_trends is not None:

        indicator_trends_html = (
            _figure_to_html(
                fig_indicator_trends,
                include_plotlyjs=False,
            )
        )

    else:

        indicator_trends_html = (
            "<p>No indicator trend data available.</p>"
        )

    country_chi_html = _figure_to_html(
        fig_country_chi,
        include_plotlyjs=False,
    )

    if fig_correlation is not None:

        correlation_html = _figure_to_html(
            fig_correlation,
            include_plotlyjs=False,
        )

    else:

        correlation_html = (
            "<p>No correlation data available.</p>"
        )

    # -----------------------------------------------------
    # Standalone HTML document
    # -----------------------------------------------------

    html = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
Global Health Dashboard — Interactive Report
</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 0;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;
    background: #f5f7fa;
    color: #1f2937;
    line-height: 1.6;
}}

.container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px 24px 60px;
}}

.header {{
    background: #ffffff;
    border-radius: 14px;
    padding: 36px;
    margin-bottom: 28px;
    box-shadow:
        0 2px 10px rgba(0, 0, 0, 0.06);
}}

.header h1 {{
    margin: 0 0 8px;
    font-size: 32px;
}}

.header p {{
    margin: 6px 0;
    color: #6b7280;
}}

.cards {{
    display: grid;
    grid-template-columns:
        repeat(4, 1fr);
    gap: 18px;
    margin-bottom: 28px;
}}

.card {{
    background: #ffffff;
    border-radius: 12px;
    padding: 24px;
    box-shadow:
        0 2px 10px rgba(0, 0, 0, 0.05);
}}

.card-label {{
    font-size: 13px;
    color: #6b7280;
    margin-bottom: 6px;
}}

.card-value {{
    font-size: 28px;
    font-weight: 700;
}}

.section {{
    background: #ffffff;
    border-radius: 14px;
    padding: 30px;
    margin-bottom: 28px;
    box-shadow:
        0 2px 10px rgba(0, 0, 0, 0.05);
}}

.section h2 {{
    margin-top: 0;
    margin-bottom: 8px;
    font-size: 24px;
}}

.section-description {{
    color: #6b7280;
    margin-top: 0;
    margin-bottom: 22px;
}}

.chart {{
    width: 100%;
    overflow-x: auto;
}}

.data-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
}}

.data-table th,
.data-table td {{
    padding: 12px 14px;
    border-bottom: 1px solid #e5e7eb;
    text-align: left;
}}

.data-table th {{
    background: #f9fafb;
    font-weight: 600;
}}

.footer {{
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-top: 35px;
}}

@media (max-width: 900px) {{

    .cards {{
        grid-template-columns:
            repeat(2, 1fr);
    }}

}}

@media (max-width: 600px) {{

    .cards {{
        grid-template-columns: 1fr;
    }}

    .container {{
        padding: 20px 12px 40px;
    }}

    .section,
    .header {{
        padding: 20px;
    }}

}}

</style>

</head>

<body>

<div class="container">

    <div class="header">

        <h1>
            Global Health Dashboard
        </h1>

        <p>
            Interactive Global Health Dataset Report
        </p>

        <p>
            Report generated on
            {report_date}
        </p>

    </div>


    <div class="cards">

        <div class="card">

            <div class="card-label">
                Records
            </div>

            <div class="card-value">
                {records:,}
            </div>

        </div>


        <div class="card">

            <div class="card-label">
                Countries
            </div>

            <div class="card-value">
                {countries:,}
            </div>

        </div>


        <div class="card">

            <div class="card-label">
                Diseases
            </div>

            <div class="card-value">
                {diseases:,}
            </div>

        </div>


        <div class="card">

            <div class="card-label">
                Year Range
            </div>

            <div class="card-value">
                {min_year}–{max_year}
            </div>

        </div>

    </div>


    <div class="section">

        <h2>
            1. Dataset Overview
        </h2>

        <p class="section-description">
            Summary of the dataset used for
            this report.
        </p>

        <table class="data-table">

            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>

            <tr>
                <td>Total Records</td>
                <td>{records:,}</td>
            </tr>

            <tr>
                <td>Countries</td>
                <td>{countries:,}</td>
            </tr>

            <tr>
                <td>Diseases</td>
                <td>{diseases:,}</td>
            </tr>

            <tr>
                <td>Year Range</td>
                <td>
                    {min_year}–{max_year}
                </td>
            </tr>

        </table>

    </div>


    <div class="section">

        <h2>
            2. Key Statistical Summary
        </h2>

        <p class="section-description">
            Descriptive statistics for major
            global health indicators.
        </p>

        {summary_table_html}

    </div>


    <div class="section">

        <h2>
            3. Global Composite Health Index Trend
        </h2>

        <p class="section-description">
            Interactive annual trend of the
            average Composite Health Index.
            Hover over points for exact values
            and use the chart controls to zoom
            or inspect the data.
        </p>

        <div class="chart">

            {chi_trend_html}

        </div>

    </div>


    <div class="section">

        <h2>
            4. Global Health Indicator Trends
        </h2>

        <p class="section-description">
            Interactive comparison of major
            health indicators across the
            reporting period.
        </p>

        <div class="chart">

            {indicator_trends_html}

        </div>

    </div>


    <div class="section">

        <h2>
            5. Average Composite Health Index
            by Country
        </h2>

        <p class="section-description">
            Interactive comparison of average
            Composite Health Index values
            across countries.
        </p>

        <div class="chart">

            {country_chi_html}

        </div>

    </div>


    <div class="section">

        <h2>
            6. CHI Correlation Analysis
        </h2>

        <p class="section-description">
            Pearson correlations between major
            health and socioeconomic indicators
            and the Composite Health Index.
            Correlation does not by itself imply
            causation.
        </p>

        <div class="chart">

            {correlation_html}

        </div>

    </div>


    <div class="section">

        <h2>
            7. Top Countries by Average CHI
        </h2>

        <p class="section-description">
            Five countries with the highest
            average Composite Health Index.
        </p>

        {top_country_html}

    </div>


    <div class="section">

        <h2>
            8. Methodology
        </h2>

        <p>
            This report is generated from the
            cleaned Global Health Dataset covering
            the period {min_year}–{max_year}.
        </p>

        <p>
            The report uses descriptive statistics,
            annual aggregation, country-level
            aggregation, and Pearson correlation
            analysis to summarize health patterns
            across countries, diseases, and years.
        </p>

        <p>
            The Composite Health Index (CHI) is
            treated as the principal overall health
            indicator for the analytical summaries
            presented in this report.
        </p>

    </div>


    <div class="section">

        <h2>
            9. Notes
        </h2>

        <p>
            This report is intended for analytical
            exploration and decision-support purposes.
        </p>

        <p>
            Correlation results describe statistical
            association and should not be interpreted
            as evidence of causation.
        </p>

        <p>
            Missing values and other data-quality
            considerations are handled according to
            the dataset cleaning pipeline used by the
            Global Health Dashboard.
        </p>

    </div>


    <div class="footer">

        Global Health Dashboard —
        Interactive HTML Report

    </div>

</div>

</body>

</html>
"""

    # -----------------------------------------------------
    # Return HTML as bytes
    # -----------------------------------------------------

    return html.encode(
        "utf-8"
    )