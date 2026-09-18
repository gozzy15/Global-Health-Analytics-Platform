"""
Overview page for the Global Health Dashboard.
"""

import streamlit as st

from utils.data_loader import load_health_data
from utils.charts import (
    create_health_index_trend,
    create_disease_burden_chart,
)

from utils.style import apply_global_styles


# ---------------------------------------------------------
# Cached data preparation
# ---------------------------------------------------------

@st.cache_data
def get_overview_metadata(data):
    """
    Calculate dataset metadata and overview KPI values.

    Cached so repeated Streamlit reruns do not repeatedly
    scan the entire dataset for the same values.
    """

    total_records = len(data)

    total_countries = (
        data["country"].nunique()
        if "country" in data.columns
        else 0
    )

    total_diseases = (
        data["disease_name"].nunique()
        if "disease_name" in data.columns
        else 0
    )

    if "year" in data.columns:

        year_values = (
            data["year"]
            .dropna()
            .astype(int)
        )

        min_year = (
            year_values.min()
            if not year_values.empty
            else "N/A"
        )

        max_year = (
            year_values.max()
            if not year_values.empty
            else "N/A"
        )

    else:

        min_year = "N/A"
        max_year = "N/A"

    avg_health_index = (
        data["composite_health_index"].mean()
        if "composite_health_index" in data.columns
        else None
    )

    avg_healthcare_access = (
        data["healthcare_access_pct"].mean()
        if "healthcare_access_pct" in data.columns
        else None
    )

    avg_recovery_rate = (
        data["recovery_rate_pct"].mean()
        if "recovery_rate_pct" in data.columns
        else None
    )

    return {
        "total_records": total_records,
        "total_countries": total_countries,
        "total_diseases": total_diseases,
        "min_year": min_year,
        "max_year": max_year,
        "avg_health_index": avg_health_index,
        "avg_healthcare_access": avg_healthcare_access,
        "avg_recovery_rate": avg_recovery_rate,
    }


# ---------------------------------------------------------
# Cached chart preparation
# ---------------------------------------------------------

@st.cache_data
def prepare_health_index_chart(data):
    """
    Prepare the Health Index trend chart.

    Cached because chart construction can be relatively
    expensive compared with simply displaying an existing
    chart object.
    """

    return create_health_index_trend(data)


@st.cache_data
def prepare_disease_burden_chart(data):
    """
    Prepare the disease burden chart.

    Cached so the chart does not need to be rebuilt on every
    Streamlit rerun when the underlying dataset has not changed.
    """

    return create_disease_burden_chart(data)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Overview | Global Health Dashboard",
    page_icon="🌍",
    layout="wide",
)

apply_global_styles()


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = load_health_data()


# ---------------------------------------------------------
# Prepare cached metadata and KPIs
# ---------------------------------------------------------

overview = get_overview_metadata(df)

total_records = overview["total_records"]
total_countries = overview["total_countries"]
total_diseases = overview["total_diseases"]
min_year = overview["min_year"]
max_year = overview["max_year"]

avg_health_index = overview["avg_health_index"]
avg_healthcare_access = overview["avg_healthcare_access"]
avg_recovery_rate = overview["avg_recovery_rate"]


# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------

st.title("🌍 Global Health Overview")

st.markdown(
    """
    A high-level view of global health conditions across
    countries, diseases, and years. Explore key health
    indicators, trends, disease burden, and the structure
    of the cleaned dataset used throughout this dashboard.
    """
)

st.divider()


# ---------------------------------------------------------
# KPI cards
# ---------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Records",
        f"{total_records:,}",
    )

with col2:
    st.metric(
        "Countries",
        total_countries,
    )

with col3:
    st.metric(
        "Diseases",
        total_diseases,
    )

with col4:
    st.metric(
        "Avg. CHI",
        (
            f"{avg_health_index:.2f}"
            if avg_health_index is not None
            else "N/A"
        ),
    )

with col5:
    st.metric(
        "Avg. Recovery Rate",
        (
            f"{avg_recovery_rate:.2f}%"
            if avg_recovery_rate is not None
            else "N/A"
        ),
    )


st.divider()


# ---------------------------------------------------------
# Global health indicators
# ---------------------------------------------------------

st.subheader("Global Health Indicators")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Average Healthcare Access",
        (
            f"{avg_healthcare_access:.2f}%"
            if avg_healthcare_access is not None
            else "N/A"
        ),
    )

with col2:
    st.metric(
        "Average Recovery Rate",
        (
            f"{avg_recovery_rate:.2f}%"
            if avg_recovery_rate is not None
            else "N/A"
        ),
    )


st.divider()


# ---------------------------------------------------------
# Health Index Trend
# ---------------------------------------------------------

st.subheader("Health Index Trend")

try:

    health_index_chart = prepare_health_index_chart(df)

    st.plotly_chart(
        health_index_chart,
        use_container_width=True,
    )

except Exception as exc:

    st.warning(
        f"Unable to display the Health Index trend chart: {exc}"
    )


# ---------------------------------------------------------
# Disease Burden
# ---------------------------------------------------------

st.subheader("Disease Burden")

try:

    disease_burden_chart = prepare_disease_burden_chart(df)

    st.plotly_chart(
        disease_burden_chart,
        use_container_width=True,
    )

except Exception as exc:

    st.warning(
        f"Unable to display the disease burden chart: {exc}"
    )


# ---------------------------------------------------------
# About the cleaned dataset
# ---------------------------------------------------------

st.divider()

st.subheader("About the Cleaned Dataset")

st.markdown(
    """
    The **Global Health Dataset** contains health, demographic,
    healthcare, socioeconomic, and disease-related information.
    It is synthetic but realistically generated across multiple countries,
    diseases, and years.

    The dataset used by this dashboard has been cleaned and
    standardized to improve consistency, data quality, and
    suitability for statistical analysis, visualization,
    and machine learning.

    Each row represents a combination of a **country, year,
    and disease**, together with associated health and
    socioeconomic indicators.
    """
)


# ---------------------------------------------------------
# Dataset coverage
# ---------------------------------------------------------

st.markdown("### Dataset Coverage")

coverage_col1, coverage_col2, coverage_col3, coverage_col4 = (
    st.columns(4)
)

with coverage_col1:

    st.metric(
        "Records",
        f"{total_records:,}",
    )

with coverage_col2:

    st.metric(
        "Countries",
        total_countries,
    )

with coverage_col3:

    st.metric(
        "Diseases",
        total_diseases,
    )

with coverage_col4:

    st.metric(
        "Year Range",
        (
            f"{min_year}–{max_year}"
            if min_year != "N/A"
            and max_year != "N/A"
            else "N/A"
        ),
    )


# ---------------------------------------------------------
# Cleaned dataset preview
# ---------------------------------------------------------

st.markdown("### Cleaned Dataset Preview")

st.markdown(
    """
    The table below shows the cleaned dataset currently
    being used by the dashboard. Use the horizontal and
    vertical controls to explore the available records
    and columns.
    """
)

preview_rows = min(100, len(df))

st.dataframe(
    df.head(preview_rows),
    use_container_width=True,
    height=500,
)


# ---------------------------------------------------------
# Column guide
# ---------------------------------------------------------

st.markdown("### Column Guide")

st.markdown(
    """
    The following guide briefly explains the variables
    contained in the cleaned dataset.
    """
)


column_definitions = {
    "row_num": (
        "Unique row identifier for each record."
    ),

    "country": (
        "Country associated with the health record."
    ),

    "year": (
        "Year in which the health observation was recorded."
    ),

    "disease_name": (
        "Disease or health condition associated with the record."
    ),

    "country_pop": (
        "Population of the country in the corresponding year."
    ),

    "incidence_rate_pct": (
        "Percentage of the population that developed or "
        "was newly affected by the disease."
    ),

    "prevalence_rate_pct": (
        "Percentage of the population living with the disease "
        "during the specified period."
    ),

    "mortality_rate_per_100_people_pct": (
        "Percentage of people associated with the disease "
        "who died from it, expressed per 100 people."
    ),

    "population_affected": (
        "Estimated number of people affected by the disease."
    ),

    "pop_affected_male": (
        "Estimated number of affected male individuals."
    ),

    "pop_affected_female": (
        "Estimated number of affected female individuals."
    ),

    "ages_0_18_pct": (
        "Percentage of affected individuals aged 0–18."
    ),

    "ages_19_35_pct": (
        "Percentage of affected individuals aged 19–35."
    ),

    "ages_36_60_pct": (
        "Percentage of affected individuals aged 36–60."
    ),

    "ages_61_plus_pct": (
        "Percentage of affected individuals aged 61 and above."
    ),

    "pop_affected_urban_pct": (
        "Percentage of affected individuals living in urban areas."
    ),

    "pop_affected_rural_pct": (
        "Percentage of affected individuals living in rural areas."
    ),

    "healthcare_access_pct": (
        "Percentage representing the population's access to "
        "healthcare services."
    ),

    "doctors_per_1000": (
        "Number of doctors available per 1,000 people."
    ),

    "hospital_beds_per_1000": (
        "Number of hospital beds available per 1,000 people."
    ),

    "treatment_type": (
        "Primary treatment approach associated with the disease."
    ),

    "recovery_rate_pct": (
        "Percentage of affected individuals who recover "
        "from the disease."
    ),

    "dalys": (
        "Disability-Adjusted Life Years, representing the "
        "combined impact of premature death and time lived "
        "with disability."
    ),

    "improvement_in_5_years_pct": (
        "Percentage improvement in the relevant health "
        "measure over a five-year period."
    ),

    "average_annual_treatment_cost_usd": (
        "Average annual treatment cost associated with the "
        "disease, expressed in US dollars."
    ),

    "availability_of_vaccines_treatment": (
        "Indicates the availability level of vaccines or "
        "treatment options."
    ),

    "composite_health_index": (
        "Composite indicator representing overall health "
        "conditions using multiple underlying health-related "
        "factors."
    ),

    "per_capita_income_usd": (
        "Average income per person in the country, expressed "
        "in US dollars."
    ),

    "education_index": (
        "Index representing the level of educational attainment "
        "within the country."
    ),

    "urbanization_rate_pct": (
        "Percentage of the country's population living in "
        "urban areas."
    ),
}


# ---------------------------------------------------------
# Display column guide
# ---------------------------------------------------------

with st.expander(
    f"View all {len(df.columns)} dataset columns and definitions"
):

    for column in df.columns:

        definition = column_definitions.get(
            column,
            "Description not available for this column.",
        )

        st.markdown(
            f"**`{column}`**  \n"
            f"{definition}"
        )

        st.divider()


# ---------------------------------------------------------
# Synthetic dataset warning
# ---------------------------------------------------------

st.markdown(
    """
    **⚠ Note:** *The dataset is synthetic and generated for
    demonstration purposes. It does not represent real-world
    health data. Hence, should not be used for policy or clinical decision-making.*
    """
)