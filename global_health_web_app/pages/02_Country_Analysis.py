"""
Global Health Dashboard — Country Analysis Page.
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.data_loader import load_health_data, get_countries, get_years
from utils.helpers import (
    display_page_header,
    display_section_title,
    format_percentage,
)
from utils.charts import (
    country_comparison_chart,
)

from utils.style import apply_global_styles


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Country Analysis | Global Health Dashboard",
    page_icon="🌍",
    layout="wide",
)

apply_global_styles()

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = load_health_data()


@st.cache_data
def get_country_analysis_metadata(
    data: pd.DataFrame,
) -> tuple[list, list]:
    """
    Return cached country and year lists for Country Analysis.
    """

    countries = get_countries(data)
    years = get_years(data)

    return countries, years


countries, years = get_country_analysis_metadata(df)

st.title("🌍 Country Analysis")
# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------

display_page_header(
    "Compare health indicators across countries and "
    "explore country-level trends from 2000 to 2024.",
)

st.divider()

# ---------------------------------------------------------
# Filters
# ---------------------------------------------------------

display_section_title("Analysis Controls")

col1, col2, col3 = st.columns(3)


# ---------------------------------------------------------
# Country selection
# ---------------------------------------------------------

with col1:

    default_countries = (
        ["Nigeria"]
        if "Nigeria" in countries
        else countries[:1]
    )

    selected_countries = st.multiselect(
        "Select countries",
        options=countries,
        default=default_countries,
        max_selections=5,
        help=(
            "Select up to 5 countries to compare their "
            "historical trends."
        ),
    )


# ---------------------------------------------------------
# Year selection
# ---------------------------------------------------------

with col2:

    selected_year = st.selectbox(
        "Select year",
        years,
        index=years.index(max(years)),
    )


# ---------------------------------------------------------
# Indicator selection
# ---------------------------------------------------------

with col3:

    metric_options = {
        #"Composite Health Index": "composite_health_index",
        "Healthcare Access": "healthcare_access_pct",
        "Doctors per 1,000": "doctors_per_1000",
        "Hospital Beds per 1,000": "hospital_beds_per_1000",
        "Recovery Rate": "recovery_rate_pct",
        "Incidence Rate": "incidence_rate_pct",
        "Prevalence Rate": "prevalence_rate_pct",
        "Mortality Rate": "mortality_rate_per_100_people_pct",
        "DALYs": "dalys",
    }

    selected_metric_label = st.selectbox(
        "Select indicator",
        list(metric_options.keys()),
    )


selected_metric = metric_options[
    selected_metric_label
]


# ---------------------------------------------------------
# Validate country selection
# ---------------------------------------------------------

if not selected_countries:

    st.info(
        "Please select at least one country to continue."
    )

    st.stop()


# ---------------------------------------------------------
# Selected country summary
# ---------------------------------------------------------

display_section_title(
    f"Selected Countries — {selected_year}"
)


# ---------------------------------------------------------
# KPI section
# ---------------------------------------------------------

kpi_columns = st.columns(len(selected_countries))


# Filter the selected countries and year once.
kpi_data = df[
    (df["country"].isin(selected_countries))
    & (df["year"] == selected_year)
]


# Pre-calculate KPI values for all selected countries.
kpi_summary = (
    kpi_data
    .groupby("country")
    .agg(
        metric_value=(
            selected_metric,
            "mean",
        ),
        country_chi=(
            "composite_health_index",
            "mean",
        ),
        diseases_covered=(
            "disease_name",
            "nunique",
        ),
    )
)


for index, country in enumerate(selected_countries):

    if country in kpi_summary.index:

        country_metric_value = (
            kpi_summary.loc[
                country,
                "metric_value",
            ]
        )

        country_chi = (
            kpi_summary.loc[
                country,
                "country_chi",
            ]
        )

        diseases_covered = (
            kpi_summary.loc[
                country,
                "diseases_covered",
            ]
        )

    else:

        country_metric_value = float("nan")
        country_chi = float("nan")
        diseases_covered = 0


    country_chi_display = (
        "Not available"
        if pd.isna(country_chi)
        else format_percentage(
            country_chi,
            2,
        )
    )


    with kpi_columns[index]:

        st.markdown(
            f"**{country}**"
        )

        if pd.isna(country_metric_value):

            metric_display = "Not available"

        elif selected_metric.endswith("_pct"):

            metric_display = format_percentage(
                country_metric_value,
                2,
            )

        else:

            metric_display = (
                f"{country_metric_value:.2f}"
            )


        st.metric(
            selected_metric_label,
            metric_display,
        )


        st.metric(
            "Composite Health Index",
            country_chi_display,
        )


        st.metric(
            "Diseases Covered",
            diseases_covered,
        )


# ---------------------------------------------------------
# Multi-country historical trend
# ---------------------------------------------------------

display_section_title(
    f"{selected_metric_label} — Historical Trend"
)

@st.cache_data
def prepare_country_trend(
    data: pd.DataFrame,
    selected_countries_tuple: tuple,
    metric: str,
) -> pd.DataFrame:
    """
    Prepare cached country-level historical trend data.
    """

    trend_data = data[
        data["country"].isin(
            selected_countries_tuple
        )
    ]

    return (
        trend_data
        .groupby(
            ["country", "year"],
            as_index=False,
        )[metric]
        .mean()
    )


trend_data = prepare_country_trend(
    df,
    tuple(selected_countries),
    selected_metric,
)


# trend_data = (
#     trend_data
#     .groupby(
#         ["country", "year"],
#         as_index=False,
#     )[selected_metric]
#     .mean()
# )


fig_trend = go.Figure()


for country in selected_countries:

    country_trend = trend_data[
        trend_data["country"] == country
    ].sort_values("year")

    fig_trend.add_trace(
        go.Scatter(
            x=country_trend["year"],
            y=country_trend[selected_metric],
            mode="lines+markers",
            name=country,
            hovertemplate=(
                f"<b>{country}</b><br>"
                "Year: %{x}<br>"
                f"{selected_metric_label}: "
                "%{y:.2f}<extra></extra>"
            ),
        )
    )


fig_trend.update_layout(
    title=(
        f"{selected_metric_label} — "
        f"Selected Countries"
    ),
    xaxis_title="Year",
    yaxis_title=selected_metric_label,
    hovermode="x unified",
    legend_title="Country",
    template="plotly_white",
)


st.plotly_chart(
    fig_trend,
    use_container_width=True,
)


# ---------------------------------------------------------
# Country comparison
# ---------------------------------------------------------

display_section_title(
    f"Country Comparison — {selected_year}"
)

@st.cache_data
def prepare_country_comparison_chart(
    data: pd.DataFrame,
    metric: str,
    year: int,
    title: str,
):
    """
    Generate and cache the country comparison chart.
    """

    return country_comparison_chart(
        data,
        metric,
        year,
        title,
    )


fig_comparison = prepare_country_comparison_chart(
    df,
    selected_metric,
    selected_year,
    f"{selected_metric_label} by Country — {selected_year}",
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True,
)


# ---------------------------------------------------------
# Selected countries — disease-level data
# ---------------------------------------------------------

display_section_title(
    f"Selected Countries — Disease-Level Data — "
    f"{selected_year}"
)


country_year_data = (
    df[
        (df["country"].isin(selected_countries))
        & (df["year"] == selected_year)
    ]
    .sort_values(
        ["country", "disease_name"]
    )
)


st.dataframe(
    country_year_data,
    use_container_width=True,
    hide_index=True,
)