"""
Data Explorer page for the Global Health Dashboard.
"""

import io
import re

import pandas as pd
import streamlit as st

from utils.data_loader import load_health_data

from utils.style import apply_global_styles


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Data Explorer | Global Health Dashboard",
    page_icon="🔎",
    layout="wide",
)

apply_global_styles()

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = load_health_data()


# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------

st.title("🔎 Data Explorer")

st.markdown(
    """
    Explore and filter the cleaned Global Health Dataset
    across countries, diseases, years, and treatment types.
    """
)

st.divider()


# ---------------------------------------------------------
# Cached filter options
# ---------------------------------------------------------

@st.cache_data
def get_data_explorer_options(
    data: pd.DataFrame,
) -> tuple[list, list, list, list]:
    """
    Prepare and cache the filter options used by Data Explorer.
    """

    country_options = sorted(
        data["country"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    year_options = sorted(
        data["year"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    disease_options = sorted(
        data["disease_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    treatment_options = sorted(
        data["treatment_type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return (
        country_options,
        year_options,
        disease_options,
        treatment_options,
    )


(
    country_options,
    year_options,
    disease_options,
    treatment_options,
) = get_data_explorer_options(df)


# ---------------------------------------------------------
# Filter options
# ---------------------------------------------------------

st.subheader("Explorer Filters")

col1, col2, col3, col4 = st.columns(4)


with col1:

    selected_countries = st.multiselect(
        "Select countries",
        options=country_options,
        placeholder="All countries",
        help=(
            "Select one or more countries. "
            "Leave empty to include all countries."
        ),
    )


with col2:

    selected_years = st.multiselect(
        "Select years",
        options=year_options,
        placeholder="All years",
        help=(
            "Select one or more years. "
            "Leave empty to include all years."
        ),
    )


with col3:

    selected_diseases = st.multiselect(
        "Select diseases",
        options=disease_options,
        placeholder="All diseases",
        help=(
            "Select one or more diseases. "
            "Leave empty to include all diseases."
        ),
    )


with col4:

    selected_treatments = st.multiselect(
        "Treatment types",
        options=treatment_options,
        placeholder="All treatment types",
        help=(
            "Select one or more treatment types. "
            "Leave empty to include all treatment types."
        ),
    )


# ---------------------------------------------------------
# Apply filters
# ---------------------------------------------------------

filter_mask = pd.Series(
    True,
    index=df.index,
)


if selected_countries:

    filter_mask &= df["country"].isin(
        selected_countries
    )


if selected_years:

    filter_mask &= df["year"].isin(
        selected_years
    )


if selected_diseases:

    filter_mask &= df["disease_name"].isin(
        selected_diseases
    )


if selected_treatments:

    filter_mask &= df["treatment_type"].isin(
        selected_treatments
    )


filtered_df = df.loc[
    filter_mask
]


# ---------------------------------------------------------
# Filter summary
# ---------------------------------------------------------

st.divider()

st.subheader("Filtered Dataset")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Matching Records",
        f"{len(filtered_df):,}",
    )


with col2:

    st.metric(
        "Columns",
        len(filtered_df.columns),
    )


with col3:

    st.metric(
        "Countries",
        filtered_df["country"].nunique(),
    )


# ---------------------------------------------------------
# Dataset preview
# ---------------------------------------------------------

st.dataframe(
    filtered_df,
    use_container_width=True,
    height=500,
)


# ---------------------------------------------------------
# Download filtered data
# ---------------------------------------------------------

st.divider()

st.subheader("Export Data")

# Determine export filename based on selected countries
if len(selected_countries) == 1:
    country_name = re.sub(
        r"[^a-z0-9]+",
        "_",
        selected_countries[0].lower(),
    ).strip("_")

    export_base_name = (
        f"global_health_filtered_{country_name}"
    )

elif len(selected_countries) > 1:
    export_base_name = (
        "global_health_filtered_multi_country"
    )

else:
    export_base_name = (
        "global_health_filtered_all_countries"
    )

# ---------------------------------------------------------
# Cached export preparation
# ---------------------------------------------------------

@st.cache_data
def prepare_filtered_exports(
    data: pd.DataFrame,
) -> tuple[bytes, bytes]:
    """
    Prepare and cache CSV and Excel exports
    for the filtered dataset.
    """

    csv_data = (
        data
        .to_csv(index=False)
        .encode("utf-8")
    )

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl",
    ) as writer:

        data.to_excel(
            writer,
            index=False,
            sheet_name="Filtered Data",
        )

    excel_data = excel_buffer.getvalue()

    return csv_data, excel_data


csv_data, excel_data = prepare_filtered_exports(
    filtered_df,
)

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_data,
        file_name=f"{export_base_name}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col2:
    st.download_button(
        label="📊 Download Excel",
        data=excel_data,
        file_name=f"{export_base_name}.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
    )