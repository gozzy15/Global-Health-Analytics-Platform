"""
Global Health Dashboard — Disease Analysis Page.
"""

import pandas as pd
import streamlit as st

from utils.data_loader import load_health_data, get_years
from utils.helpers import (
    display_page_header,
    display_section_title,
)
from utils.charts import (
    disease_comparison_chart,
    disease_trend_chart,
)

from utils.style import apply_global_styles

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Disease Analysis | Global Health Dashboard",
    page_icon="🦠",
    layout="wide",
)

apply_global_styles()

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = load_health_data()

@st.cache_data
def get_disease_analysis_metadata(
    data: pd.DataFrame,
) -> tuple[list, list]:
    """
    Return cached year and disease lists for Disease Analysis.
    """

    years = get_years(data)

    diseases = sorted(
        data["disease_name"]
        .dropna()
        .unique()
        .tolist()
    )

    return years, diseases


years, diseases = get_disease_analysis_metadata(df)


st.title("🦠 Disease Analysis")
# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------

display_page_header(
    "Explore disease-level patterns, compare diseases, "
    "and examine how key health indicators change over time.",
)

st.divider()

# ---------------------------------------------------------
# Analysis controls
# ---------------------------------------------------------

display_section_title("Analysis Controls")

col1, col2, col3 = st.columns(3)


with col1:

    selected_diseases = st.multiselect(
        "Select diseases",
        options=diseases,
        default=([diseases[0]] if diseases else []) if len(diseases) <= 10 else ([diseases[13]] if diseases else []),                   
        max_selections=5,
        help="Select up to 5 diseases for comparison.",
    )


with col2:

    selected_year = st.selectbox(
        "Select year",
        years,
        index=years.index(max(years)),
    )


with col3:

    metric_options = {
        "Incidence Rate": "incidence_rate_pct",
        "Prevalence Rate": "prevalence_rate_pct",
        "Mortality Rate": "mortality_rate_per_100_people_pct",
        "Recovery Rate": "recovery_rate_pct",
        "DALYs": "dalys",
        "Composite Health Index": "composite_health_index",
    }

    selected_metric_label = st.selectbox(
        "Select indicator",
        list(metric_options.keys()),
    )


selected_metric = metric_options[
    selected_metric_label
]


# ---------------------------------------------------------
# Validate disease selection
# ---------------------------------------------------------

if not selected_diseases:

    st.info(
        "Please select at least one disease to begin the analysis."
    )

    st.stop()


# ---------------------------------------------------------
# Selected disease data
# ---------------------------------------------------------

@st.cache_data
def prepare_disease_year_data(
    data: pd.DataFrame,
    selected_diseases_tuple: tuple,
    year: int,
) -> pd.DataFrame:
    """
    Prepare cached data for the selected diseases and year.
    """

    return (
        data[
            data["disease_name"].isin(
                selected_diseases_tuple
            )
            & (data["year"] == year)
        ]
        .sort_values(
            ["disease_name", "country"]
        )
    )


year_data = prepare_disease_year_data(
    df,
    tuple(selected_diseases),
    selected_year,
)


# ---------------------------------------------------------
# Disease summary
# ---------------------------------------------------------

display_section_title(
    f"Selected Diseases — {selected_year}"
)

kpi1, kpi2, kpi3 = st.columns(3)


with kpi1:

    st.metric(
        "Diseases Selected",
        len(selected_diseases),
    )


with kpi2:

    selected_value = year_data[
        selected_metric
    ].mean()

    if pd.isna(selected_value):

        selected_value_display = "Not available"

    elif selected_metric.endswith("_pct"):

        selected_value_display = (
            f"{selected_value:.2f}%"
        )

    else:

        selected_value_display = (
            f"{selected_value:.2f}"
        )

    st.metric(
        f"Average {selected_metric_label}",
        selected_value_display,
    )


with kpi3:

    st.metric(
        "Countries Covered",
        year_data["country"].nunique(),
    )


# ---------------------------------------------------------
# Selected diseases
# ---------------------------------------------------------

st.caption(
    "Selected diseases: "
    + ", ".join(selected_diseases)
)


# ---------------------------------------------------------
# Disease trend
# ---------------------------------------------------------

display_section_title(
    "Historical Trend"
)

@st.cache_data
def prepare_disease_trend_chart(
    data: pd.DataFrame,
    selected_diseases_tuple: tuple,
    metric: str,
    title: str,
):
    """
    Generate and cache the disease historical trend chart.
    """

    return disease_trend_chart(
        data,
        list(selected_diseases_tuple),
        metric,
        title,
    )


fig_trend = prepare_disease_trend_chart(
    df,
    tuple(selected_diseases),
    selected_metric,
    f"{selected_metric_label} — Selected Diseases",
)

st.plotly_chart(
    fig_trend,
    use_container_width=True,
)


# ---------------------------------------------------------
# Disease comparison
# ---------------------------------------------------------

display_section_title(
    f"Disease Comparison — {selected_year}"
)

@st.cache_data
def prepare_disease_comparison_chart(
    data: pd.DataFrame,
    metric: str,
    year: int,
    title: str,
):
    """
    Generate and cache the disease comparison chart.
    """

    return disease_comparison_chart(
        data,
        metric,
        year,
        title,
    )


fig_comparison = prepare_disease_comparison_chart(
    df,
    selected_metric,
    selected_year,
    f"{selected_metric_label} by Disease — {selected_year}",
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True,
)


# ---------------------------------------------------------
# Disease-level country data
# ---------------------------------------------------------

display_section_title(
    "Selected Diseases — Country Data"
)

disease_year_data = year_data

st.dataframe(
    disease_year_data,
    use_container_width=True,
    hide_index=True,
)