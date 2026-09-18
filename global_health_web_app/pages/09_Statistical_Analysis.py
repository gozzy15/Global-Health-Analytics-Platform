"""
Global Health Dashboard — Statistical Analysis Page.
"""

import io

import streamlit as st
import pandas as pd
import plotly.express as px


from utils.data_loader import (
    load_health_data,
    get_years,
)

from utils.helpers import (
    display_page_header,
    display_section_title,
)

from utils.style import apply_global_styles

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Statistical Analysis | Global Health Dashboard",
    page_icon="📐",
    layout="wide",
)

apply_global_styles()

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = load_health_data()


@st.cache_data
def get_statistical_analysis_years(
    data: pd.DataFrame,
) -> list:
    """
    Return cached years for Statistical Analysis.
    """

    return get_years(data)


years = get_statistical_analysis_years(df)


st.title("📐 Statistical Analysis")
# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------

display_page_header(
    "Examine the statistical properties, distributions, "
    "variation, and outliers of key global health indicators.",
)

st.divider()

# ---------------------------------------------------------
# Indicator options
# ---------------------------------------------------------

metric_options = {
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

display_section_title(
    "Analysis Controls"
)

col1, col2 = st.columns(2)


with col1:

    selected_metric_label = st.selectbox(
        "Select indicator",
        list(metric_options.keys()),
    )


with col2:

    selected_year = st.selectbox(
        "Select year",
        years,
        index=years.index(max(years)),
    )


selected_metric = metric_options[
    selected_metric_label
]


# ---------------------------------------------------------
# Prepare selected indicator
# ---------------------------------------------------------

@st.cache_data
def prepare_statistical_metric_data(
    data: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Prepare cached data for the selected statistical indicator.
    """

    metric_data = data[
        [
            "country",
            "disease_name",
            "year",
            metric,
        ]
    ].copy()

    return metric_data.dropna(
        subset=[metric]
    )


metric_data = prepare_statistical_metric_data(
    df,
    selected_metric,
)

if metric_data.empty:

    st.warning(
        f"No usable data is available for "
        f"{selected_metric_label}."
    )

    st.stop()

values = metric_data[
    selected_metric
]


# ---------------------------------------------------------
# Overall statistics
# ---------------------------------------------------------

@st.cache_data
def calculate_statistical_results(
    data: pd.DataFrame,
    metric: str,
):
    """
    Calculate and cache reusable statistical results.
    """

    values = data[
        metric
    ]

    overall_mean = values.mean()
    overall_median = values.median()
    overall_std = values.std()
    overall_min = values.min()
    overall_max = values.max()

    descriptive_stats = (
        values
        .describe()
        .to_frame(
            name="Value"
        )
    )

    descriptive_stats.loc[
        "variance"
    ] = values.var()

    descriptive_stats.loc[
        "skewness"
    ] = values.skew()

    descriptive_stats.loc[
        "kurtosis"
    ] = values.kurtosis()

    descriptive_stats = (
        descriptive_stats
        .reindex(
            [
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
                "variance",
                "skewness",
                "kurtosis",
            ]
        )
    )

    yearly_statistics = (
        data
        .groupby("year")[metric]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
        .reset_index()
        .sort_values("year")
    )

    yearly_statistics["year"] = (
        yearly_statistics["year"]
        .round()
        .astype(int)
    )

    country_statistics = (
        data
        .groupby("country")[metric]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
        .reset_index()
        .sort_values(
            "mean",
            ascending=False,
        )
    )

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = data[
        (data[metric] < lower_bound)
        | (data[metric] > upper_bound)
    ].copy()

    return (
        overall_mean,
        overall_median,
        overall_std,
        overall_min,
        overall_max,
        descriptive_stats,
        yearly_statistics,
        country_statistics,
        q1,
        q3,
        lower_bound,
        upper_bound,
        outliers,
    )


(
    overall_mean,
    overall_median,
    overall_std,
    overall_min,
    overall_max,
    descriptive_stats,
    yearly_statistics,
    country_statistics,
    q1,
    q3,
    lower_bound,
    upper_bound,
    outliers,
) = calculate_statistical_results(
    metric_data,
    selected_metric,
)


stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = (
    st.columns(5)
)


with stat_col1:

    st.metric(
        "Mean",
        f"{overall_mean:.2f}",
    )


with stat_col2:

    st.metric(
        "Median",
        f"{overall_median:.2f}",
    )


with stat_col3:

    st.metric(
        "Std. Deviation",
        f"{overall_std:.2f}",
    )


with stat_col4:

    st.metric(
        "Minimum",
        f"{overall_min:.2f}",
    )


with stat_col5:

    st.metric(
        "Maximum",
        f"{overall_max:.2f}",
    )


# ---------------------------------------------------------
# Detailed descriptive statistics
# ---------------------------------------------------------

display_section_title(
    "Descriptive Statistics"
)


st.dataframe(
    descriptive_stats.style.format(
        "{:.4f}"
    ),
    use_container_width=True,
)


# ---------------------------------------------------------
# Distribution
# ---------------------------------------------------------

display_section_title(
    f"{selected_metric_label} — Distribution"
)


@st.cache_data
def create_statistical_histogram(
    data: pd.DataFrame,
    metric: str,
    metric_label: str,
):
    """
    Create and cache the statistical distribution histogram.
    """

    figure = px.histogram(
        data,
        x=metric,
        nbins=30,
        marginal="box",
        title=(
            f"Distribution of {metric_label}"
        ),
    )

    figure.update_layout(
        xaxis_title=metric_label,
        yaxis_title="Frequency",
    )

    return figure


fig_histogram = create_statistical_histogram(
    metric_data,
    selected_metric,
    selected_metric_label,
)


st.plotly_chart(
    fig_histogram,
    use_container_width=True,
)


# ---------------------------------------------------------
# Yearly statistics
# ---------------------------------------------------------

display_section_title(
    f"{selected_metric_label} — Yearly Statistics"
)


st.dataframe(
    yearly_statistics.style.format(
        {
            "mean": "{:.3f}",
            "median": "{:.3f}",
            "std": "{:.3f}",
            "min": "{:.3f}",
            "max": "{:.3f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# Yearly trend
# ---------------------------------------------------------
@st.cache_data
def create_yearly_statistics_chart(
    data: pd.DataFrame,
    metric_label: str,
):
    """
    Create and cache the yearly statistical trend chart.
    """

    figure = px.line(
        data,
        x="year",
        y="mean",
        markers=True,
        title=(
            f"Average {metric_label} "
            "by Year"
        ),
    )

    figure.update_layout(
        xaxis_title="Year",
        yaxis_title=(
            f"Average {metric_label}"
        ),
        hovermode="x unified",
    )

    return figure


fig_yearly = create_yearly_statistics_chart(
    yearly_statistics,
    selected_metric_label,
)


st.plotly_chart(
    fig_yearly,
    use_container_width=True,
)


# ---------------------------------------------------------
# Selected year statistics
# ---------------------------------------------------------

display_section_title(
    f"{selected_metric_label} — {selected_year}"
)


selected_year_data = metric_data[
    metric_data["year"] == selected_year
]


if selected_year_data.empty:

    st.warning(
        "No data is available for the selected year."
    )

else:

    year_values = selected_year_data[
        selected_metric
    ]

    year_col1, year_col2, year_col3, year_col4 = (
        st.columns(4)
    )

    with year_col1:

        st.metric(
            "Mean",
            f"{year_values.mean():.2f}",
        )

    with year_col2:

        st.metric(
            "Median",
            f"{year_values.median():.2f}",
        )

    with year_col3:

        st.metric(
            "Std. Deviation",
            f"{year_values.std():.2f}",
        )

    with year_col4:

        st.metric(
            "Observations",
            f"{len(year_values):,}",
        )


# ---------------------------------------------------------
# Country-level statistics
# ---------------------------------------------------------

display_section_title(
    f"{selected_metric_label} — Country Statistics"
)


st.dataframe(
    country_statistics.style.format(
        {
            "mean": "{:.3f}",
            "median": "{:.3f}",
            "std": "{:.3f}",
            "min": "{:.3f}",
            "max": "{:.3f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# Box plot by country
# ---------------------------------------------------------

display_section_title(
    f"{selected_metric_label} — Country Distribution"
)


@st.cache_data
def create_country_box_plot(
    data: pd.DataFrame,
    metric: str,
    metric_label: str,
):
    """
    Create and cache the country distribution box plot.
    """

    figure = px.box(
        data,
        x="country",
        y=metric,
        points="outliers",
        title=(
            f"{metric_label} Distribution "
            "by Country"
        ),
    )

    figure.update_layout(
        xaxis_title="Country",
        yaxis_title=metric_label,
        xaxis_tickangle=-45,
    )

    return figure


fig_box = create_country_box_plot(
    metric_data,
    selected_metric,
    selected_metric_label,
)


st.plotly_chart(
    fig_box,
    use_container_width=True,
)


# ---------------------------------------------------------
# Outlier analysis
# ---------------------------------------------------------

display_section_title(
    "Outlier Analysis"
)


outlier_col1, outlier_col2, outlier_col3 = (
    st.columns(3)
)


with outlier_col1:

    st.metric(
        "Q1",
        f"{q1:.2f}",
    )


with outlier_col2:

    st.metric(
        "Q3",
        f"{q3:.2f}",
    )


with outlier_col3:

    st.metric(
        "Outlier Records",
        f"{len(outliers):,}",
    )


st.caption(
    f"Lower outlier boundary: {lower_bound:.2f} | "
    f"Upper outlier boundary: {upper_bound:.2f}"
)


if outliers.empty:

    st.success(
        "No observations were identified as "
        "outliers using the 1.5 × IQR rule."
    )

else:

    st.dataframe(
        outliers[
            [
                "country",
                "disease_name",
                "year",
                selected_metric,
            ]
        ]
        .sort_values(
            selected_metric
        ),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# Statistical interpretation
# ---------------------------------------------------------

display_section_title(
    "Statistical Interpretation"
)


if overall_mean > overall_median:

    distribution_interpretation = (
        "The mean is higher than the median, "
        "which may indicate a right-skewed distribution."
    )

elif overall_mean < overall_median:

    distribution_interpretation = (
        "The mean is lower than the median, "
        "which may indicate a left-skewed distribution."
    )

else:

    distribution_interpretation = (
        "The mean and median are approximately equal, "
        "suggesting a relatively balanced distribution."
    )


if abs(values.skew()) < 0.5:

    skew_interpretation = (
        "Skewness is relatively low."
    )

elif values.skew() > 0:

    skew_interpretation = (
        "The distribution has positive (right) skew."
    )

else:

    skew_interpretation = (
        "The distribution has negative (left) skew."
    )


st.info(
    f"""
    **Distribution:** {distribution_interpretation}

    **Skewness:** {skew_interpretation}

    **Standard deviation:** The standard deviation of
    {overall_std:.2f} indicates the typical spread of
    observations around the mean.

    **Outliers:** {len(outliers):,} observations were
    identified using the 1.5 × IQR rule.

    These statistics describe the data distribution and
    association patterns; they do not by themselves establish
    causation.
    """
)


# ---------------------------------------------------------
# Export statistical analysis
# ---------------------------------------------------------

display_section_title(
    "Export Statistical Analysis"
)


# ---------------------------------------------------------
# Prepare export tables
# ---------------------------------------------------------

descriptive_export = (
    descriptive_stats
    .reset_index()
    .rename(
        columns={
            "index": "Statistic"
        }
    )
)


outlier_export = outliers[
    [
        "country",
        "disease_name",
        "year",
        selected_metric,
    ]
].sort_values(
    selected_metric
)


if selected_year_data.empty:

    selected_year_export = pd.DataFrame(
        {
            "Statistic": [
                "Mean",
                "Median",
                "Std. Deviation",
                "Observations",
            ],
            "Value": [
                float("nan"),
                float("nan"),
                float("nan"),
                0,
            ],
        }
    )

else:

    year_values = selected_year_data[
        selected_metric
    ]

    selected_year_export = pd.DataFrame(
        {
            "Statistic": [
                "Mean",
                "Median",
                "Std. Deviation",
                "Observations",
            ],
            "Value": [
                year_values.mean(),
                year_values.median(),
                year_values.std(),
                len(year_values),
            ],
        }
    )


# ---------------------------------------------------------
# Determine export filename
# ---------------------------------------------------------

export_base_name = (
    "global_health_statistical_analysis_"
    f"{selected_metric}"
)


# ---------------------------------------------------------
# CSV export
# ---------------------------------------------------------

@st.cache_data
def prepare_statistical_exports(
    descriptive_data: pd.DataFrame,
    yearly_data: pd.DataFrame,
    country_data: pd.DataFrame,
    selected_year_data_export: pd.DataFrame,
    outlier_data: pd.DataFrame,
) -> tuple[bytes, bytes]:
    """
    Generate and cache CSV and Excel statistical exports.
    """

    csv_data = (
        descriptive_data
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )


    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl",
    ) as writer:

        descriptive_data.to_excel(
            writer,
            index=False,
            sheet_name="Descriptive Stats",
        )

        yearly_data.to_excel(
            writer,
            index=False,
            sheet_name="Yearly Statistics",
        )

        country_data.to_excel(
            writer,
            index=False,
            sheet_name="Country Statistics",
        )

        selected_year_data_export.to_excel(
            writer,
            index=False,
            sheet_name="Selected Year",
        )

        outlier_data.to_excel(
            writer,
            index=False,
            sheet_name="Outlier Analysis",
        )


    excel_data = (
        excel_buffer.getvalue()
    )

    return csv_data, excel_data


csv_data, excel_data = prepare_statistical_exports(
    descriptive_export,
    yearly_statistics,
    country_statistics,
    selected_year_export,
    outlier_export,
)


# ---------------------------------------------------------
# Download buttons
# ---------------------------------------------------------

export_col1, export_col2 = st.columns(2)


with export_col1:

    st.download_button(
        label="⬇️ Download Descriptive Statistics (CSV)",
        data=csv_data,
        file_name=f"{export_base_name}.csv",
        mime="text/csv",
        use_container_width=True,
    )


with export_col2:

    st.download_button(
        label="📊 Download Statistical Analysis (Excel)",
        data=excel_data,
        file_name=f"{export_base_name}.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
    )