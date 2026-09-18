"""
Global Health Dashboard — Trends Analysis Page.
"""

import pandas as pd
import streamlit as st
import io

from utils.data_loader import (
    load_health_data,
    get_countries,
    get_years,
)

from utils.helpers import (
    display_page_header,
    display_section_title,
    format_percentage,
)

from utils.charts import (
    trends_analysis_chart,
)

from utils.style import apply_global_styles


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Trends Analysis | Global Health Dashboard",
    page_icon="📈",
    layout="wide",
)

apply_global_styles()

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = load_health_data()


# ---------------------------------------------------------
# Cached Trends Analysis metadata
# ---------------------------------------------------------

@st.cache_data
def get_trends_analysis_metadata(
    data: pd.DataFrame,
) -> tuple[list, list, list]:
    """
    Prepare and cache countries, years, and diseases
    used by Trends Analysis.
    """

    countries = get_countries(data)

    years = get_years(data)

    diseases = sorted(
        data["disease_name"]
        .dropna()
        .unique()
        .tolist()
    )

    return countries, years, diseases


(
    countries,
    years,
    diseases,
) = get_trends_analysis_metadata(df)


st.title("📈 Trends Analysis")
# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------

display_page_header(
    "Explore how key health indicators change over time "
    "across selected countries and diseases.",
)

st.divider()

# ---------------------------------------------------------
# Analysis controls
# ---------------------------------------------------------

display_section_title(
    "Analysis Controls"
)


# ---------------------------------------------------------
# Country and disease controls
# ---------------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    selected_countries = st.multiselect(
        "Select countries",
        options=countries,
        default=(
            ["Nigeria"]
            if "Nigeria" in countries
            else countries[:1]
        ),
        max_selections=8,
        help=(
            "Select up to 8 countries to compare "
            "their trends."
        ),
    )


with col2:

    selected_diseases = st.multiselect(
        "Select diseases",
        options=diseases,
        placeholder="All diseases",
        help=(
            "Select one or more diseases. "
            "Leave empty to include all diseases."
        ),
    )


# ---------------------------------------------------------
# Indicator and year controls
# ---------------------------------------------------------

col3, col4 = st.columns(2)


with col3:

    metric_options = {
        "Composite Health Index":
            "composite_health_index",

        "Healthcare Access":
            "healthcare_access_pct",

        "Doctors per 1,000":
            "doctors_per_1000",

        "Hospital Beds per 1,000":
            "hospital_beds_per_1000",

        "Recovery Rate":
            "recovery_rate_pct",

        "Incidence Rate":
            "incidence_rate_pct",

        "Prevalence Rate":
            "prevalence_rate_pct",

        "Mortality Rate":
            "mortality_rate_per_100_people_pct",

        "DALYs":
            "dalys",
    }

    selected_metric_label = st.selectbox(
        "Select indicator",
        list(metric_options.keys()),
    )


with col4:

    selected_year_range = st.slider(
        "Select year range",
        min_value=min(years),
        max_value=max(years),
        value=(
            min(years),
            max(years),
        ),
    )


selected_metric = metric_options[
    selected_metric_label
]


start_year, end_year = selected_year_range


# ---------------------------------------------------------
# Validate country selection
# ---------------------------------------------------------

if not selected_countries:

    st.warning(
        "Please select at least one country "
        "to display the trend analysis."
    )

    st.stop()


# ---------------------------------------------------------
# Cached trend filtering
# ---------------------------------------------------------

@st.cache_data
def prepare_trend_data(
    data: pd.DataFrame,
    selected_countries_tuple: tuple,
    selected_diseases_tuple: tuple,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Filter and cache the dataset used by Trends Analysis.
    """

    filter_mask = (
        data["country"].isin(
            selected_countries_tuple
        )
    ) & (
        data["year"].between(
            start_year,
            end_year,
        )
    )

    if selected_diseases_tuple:

        filter_mask &= data[
            "disease_name"
        ].isin(
            selected_diseases_tuple
        )

    return data.loc[
        filter_mask
    ].copy()


trend_df = prepare_trend_data(
    df,
    tuple(selected_countries),
    tuple(selected_diseases),
    start_year,
    end_year,
)


# ---------------------------------------------------------
# Trend summary
# ---------------------------------------------------------

display_section_title(
    "Trend Summary"
)


@st.cache_data
def prepare_trend_summary(
    data: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Group and cache country-level trend summaries.
    """

    return (
        data
        .groupby(
            ["year", "country"],
            as_index=False,
        )[metric]
        .mean()
        .sort_values(
            ["country", "year"]
        )
    )


trend_summary = prepare_trend_summary(
    trend_df,
    selected_metric,
)


summary_col1, summary_col2, summary_col3 = (
    st.columns(3)
)


with summary_col1:

    st.metric(
        "Countries Selected",
        len(selected_countries),
    )


with summary_col2:

    if selected_diseases:

        st.metric(
            "Diseases Selected",
            len(selected_diseases),
        )

    else:

        st.metric(
            "Diseases Included",
            trend_df["disease_name"].nunique(),
        )


with summary_col3:

    st.metric(
        "Years",
        f"{start_year}–{end_year}",
    )


# ---------------------------------------------------------
# Historical trend
# ---------------------------------------------------------

display_section_title(
    f"{selected_metric_label} — Historical Trend"
)


@st.cache_data
def prepare_trends_analysis_chart(
    data: pd.DataFrame,
    selected_countries_tuple: tuple,
    selected_diseases_tuple: tuple,
    metric: str,
    title: str,
):
    """
    Generate and cache the historical trend chart.
    """

    return trends_analysis_chart(
        data,
        list(selected_countries_tuple),
        list(selected_diseases_tuple),
        metric,
        title,
    )


fig_trend = prepare_trends_analysis_chart(
    trend_df,
    tuple(selected_countries),
    tuple(selected_diseases),
    selected_metric,
    f"{selected_metric_label} — Country Trends",
)


st.plotly_chart(
    fig_trend,
    use_container_width=True,
)


# ---------------------------------------------------------
# Country trend summary
# ---------------------------------------------------------

display_section_title(
    "Country Trend Summary"
)


summary_rows = []


for country in selected_countries:

    country_data = trend_summary[
        trend_summary["country"] == country
    ].sort_values("year")


    if country_data.empty:
        continue


    first_value = country_data[
        selected_metric
    ].iloc[0]


    last_value = country_data[
        selected_metric
    ].iloc[-1]


    absolute_change = (
        last_value - first_value
    )


    if first_value != 0:

        percentage_change = (
            absolute_change
            / abs(first_value)
        ) * 100

    else:

        percentage_change = None


    summary_rows.append(
        {
            "Country": country,
            "Start Year": int(
                country_data["year"].iloc[0]
            ),
            "End Year": int(
                country_data["year"].iloc[-1]
            ),
            "Starting Value": first_value,
            "Ending Value": last_value,
            "Absolute Change": absolute_change,
            "Percentage Change": percentage_change,
        }
    )


trend_summary_table = pd.DataFrame(
    summary_rows
)


if not trend_summary_table.empty:

    st.dataframe(
        trend_summary_table,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# Trend data
# ---------------------------------------------------------

display_section_title(
    "Trend Data"
)


trend_data_display = (
    trend_summary
    .sort_values(
        ["year", "country"]
    )
    .reset_index(drop=True)
)


st.dataframe(
    trend_data_display,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# Export trend data
# ---------------------------------------------------------

display_section_title(
    "Export Data"
)


# ---------------------------------------------------------
# Determine export filename
# ---------------------------------------------------------

if len(selected_countries) == 1:

    country_name = (
        selected_countries[0]
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    export_base_name = (
        f"global_health_trends_{country_name}"
    )

elif len(selected_countries) > 1:

    export_base_name = (
        "global_health_trends_multi_country"
    )

else:

    export_base_name = (
        "global_health_trends_all_countries"
    )


# ---------------------------------------------------------
# Cached trend exports
# ---------------------------------------------------------

@st.cache_data
def prepare_trend_exports(
    trend_data: pd.DataFrame,
    summary_data: pd.DataFrame,
) -> tuple[bytes, bytes]:
    """
    Prepare and cache CSV and Excel exports
    for the Trends Analysis page.
    """

    csv_data = (
        trend_data
        .to_csv(index=False)
        .encode("utf-8")
    )

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl",
    ) as writer:

        trend_data.to_excel(
            writer,
            index=False,
            sheet_name="Trend Data",
        )

        if not summary_data.empty:

            summary_data.to_excel(
                writer,
                index=False,
                sheet_name="Country Summary",
            )

    excel_data = excel_buffer.getvalue()

    return csv_data, excel_data


csv_data, excel_data = prepare_trend_exports(
    trend_data_display,
    trend_summary_table,
)


# ---------------------------------------------------------
# Download buttons
# ---------------------------------------------------------

export_col1, export_col2 = st.columns(2)


with export_col1:

    st.download_button(
        label="⬇️ Download Trend Data (CSV)",
        data=csv_data,
        file_name=f"{export_base_name}.csv",
        mime="text/csv",
        use_container_width=True,
    )


with export_col2:

    st.download_button(
        label="📊 Download Trend Analysis (Excel)",
        data=excel_data,
        file_name=f"{export_base_name}.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
    )