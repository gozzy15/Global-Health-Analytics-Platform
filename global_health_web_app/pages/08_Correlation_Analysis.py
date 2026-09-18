"""
Global Health Dashboard — Correlation Analysis Page.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_health_data
from utils.helpers import (
    display_page_header,
    display_section_title,
)

from utils.style import apply_global_styles


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Correlation Analysis | Global Health Dashboard",
    page_icon="🔗",
    layout="wide",
)

apply_global_styles()

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = load_health_data()


st.title("🔗 Correlation Analysis")
# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------

display_page_header(
    "Explore relationships between health, healthcare, "
    "and socioeconomic indicators across the dataset.",
)

st.divider()

# ---------------------------------------------------------
# Correlation-ready indicators
# ---------------------------------------------------------

indicator_options = {
    "Composite Health Index": "composite_health_index",
    "Incidence Rate": "incidence_rate_pct",
    "Prevalence Rate": "prevalence_rate_pct",
    "Mortality Rate": "mortality_rate_per_100_people_pct",
    "Healthcare Access": "healthcare_access_pct",
    "Doctors per 1,000": "doctors_per_1000",
    "Hospital Beds per 1,000": "hospital_beds_per_1000",
    "Recovery Rate": "recovery_rate_pct",
    "DALYs": "dalys",
    "Per Capita Income": "per_capita_income_usd",
    "Education Index": "education_index",
    "Urbanization Rate": "urbanization_rate_pct",
}


# ---------------------------------------------------------
# Analysis controls
# ---------------------------------------------------------

display_section_title("Analysis Controls")


selected_labels = st.multiselect(
    "Select indicators",
    options=list(indicator_options.keys()),
    default=list(indicator_options.keys()),
    help=(
        "Select at least two numerical indicators "
        "to calculate their correlations."
    ),
)


selected_columns = [
    indicator_options[label]
    for label in selected_labels
]


# ---------------------------------------------------------
# Validate indicator selection
# ---------------------------------------------------------

if len(selected_columns) < 2:

    st.warning(
        "Please select at least two indicators "
        "to generate the correlation analysis."
    )

    st.stop()


# ---------------------------------------------------------
# Prepare correlation dataset
# ---------------------------------------------------------

correlation_data = df[
    selected_columns
].copy()


# ---------------------------------------------------------
# Calculate correlation matrix
# ---------------------------------------------------------

@st.cache_data
def calculate_correlation_matrix(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate and cache the correlation matrix
    for the selected indicators.
    """

    return data.corr()


correlation_matrix = calculate_correlation_matrix(
    correlation_data,
)


# ---------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------

display_section_title(
    "Correlation Heatmap"
)

heatmap_data = correlation_matrix.copy()

heatmap_data.columns = selected_labels
heatmap_data.index = selected_labels


fig = px.imshow(
    heatmap_data,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Indicator Correlation Matrix",
)

fig.update_layout(
    xaxis_title="Indicator",
    yaxis_title="Indicator",
    coloraxis_colorbar_title="Correlation",
    height=700,
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ---------------------------------------------------------
# Correlation table
# ---------------------------------------------------------

display_section_title(
    "Correlation Coefficients"
)

st.dataframe(
    heatmap_data.style.format(
        "{:.2f}"
    ),
    use_container_width=True,
)


# ---------------------------------------------------------
# Extract unique correlation pairs
# ---------------------------------------------------------

correlation_pairs = []

for i in range(
    len(selected_columns)
):

    for j in range(
        i + 1,
        len(selected_columns)
    ):

        column_1 = selected_columns[i]
        column_2 = selected_columns[j]

        correlation_value = correlation_matrix.loc[
            column_1,
            column_2,
        ]

        correlation_pairs.append(
            {
                "Indicator 1": (
                    selected_labels[i]
                ),
                "Indicator 2": (
                    selected_labels[j]
                ),
                "Correlation": (
                    correlation_value
                ),
                "Absolute Correlation": (
                    abs(correlation_value)
                ),
            }
        )


pairs_df = pd.DataFrame(
    correlation_pairs
)


# ---------------------------------------------------------
# Strongest relationships
# ---------------------------------------------------------

display_section_title(
    "Strongest Relationships"
)


@st.cache_data
def prepare_strongest_relationships(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort and classify the strongest correlation pairs.
    """

    strongest = (
        data
        .sort_values(
            "Absolute Correlation",
            ascending=False,
        )
        .head(10)
        .copy()
    )

    strongest["Relationship"] = (
        strongest["Correlation"]
        .apply(
            lambda value:
                "Strong Positive"
                if value >= 0.70
                else (
                    "Moderate Positive"
                    if value >= 0.30
                    else (
                        "Strong Negative"
                        if value <= -0.70
                        else (
                            "Moderate Negative"
                            if value <= -0.30
                            else "Weak"
                        )
                    )
                )
        )
    )

    return strongest


strongest_pairs = (
    prepare_strongest_relationships(
        pairs_df,
    )
)


st.dataframe(
    strongest_pairs[
        [
            "Indicator 1",
            "Indicator 2",
            "Correlation",
            "Relationship",
        ]
    ].style.format(
        {
            "Correlation": "{:.3f}"
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# CHI-focused analysis
# ---------------------------------------------------------

if (
    "composite_health_index"
    in selected_columns
):

    display_section_title(
        "Composite Health Index Relationships"
    )


    @st.cache_data
    def prepare_chi_correlations(
        matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare and classify CHI correlation results.
        """

        chi_result = (
            matrix[
                "composite_health_index"
            ]
            .drop(
                "composite_health_index"
            )
            .sort_values(
                ascending=False,
            )
            .reset_index()
        )

        chi_result.columns = [
            "Indicator",
            "Correlation",
        ]

        chi_result["Relationship"] = (
            chi_result["Correlation"]
            .apply(
                lambda value:
                    "Strong Positive"
                    if value >= 0.70
                    else (
                        "Moderate Positive"
                        if value >= 0.30
                        else (
                            "Strong Negative"
                            if value <= -0.70
                            else (
                                "Moderate Negative"
                                if value <= -0.30
                                else "Weak"
                            )
                        )
                    )
            )
        )

        return chi_result


    chi_correlations = (
        prepare_chi_correlations(
            correlation_matrix,
        )
    )

    st.dataframe(
        chi_correlations.style.format(
            {
                "Correlation": "{:.3f}"
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# Interpretation guide
# ---------------------------------------------------------

display_section_title(
    "Correlation Interpretation"
)

st.markdown(
    """
    **Correlation coefficient (r)** ranges from **-1 to +1**.

    - **+1.00** → Perfect positive relationship
    - **+0.70 to +0.99** → Strong positive relationship
    - **+0.30 to +0.69** → Moderate positive relationship
    - **-0.29 to +0.29** → Weak or little linear relationship
    - **-0.30 to -0.69** → Moderate negative relationship
    - **-0.70 to -0.99** → Strong negative relationship
    - **-1.00** → Perfect negative relationship

    **Important:** Correlation describes association, not causation.
    A strong correlation does not by itself mean that one
    indicator causes the other.
    """
)