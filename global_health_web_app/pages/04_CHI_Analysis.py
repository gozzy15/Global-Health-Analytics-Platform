"""
CHI Analysis page for the Global Health Dashboard.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils.data_loader import load_health_data

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="CHI Analysis | Global Health Dashboard",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = load_health_data()

# ---------------------------------------------------------
# Cached CHI analysis helpers
# ---------------------------------------------------------

def interpret_correlation(value):
    """Return a human-readable interpretation of a Pearson correlation."""
    if pd.isna(value):
        return "Insufficient data"

    absolute_value = abs(value)

    if absolute_value >= 0.90:
        strength = "Very strong"
    elif absolute_value >= 0.70:
        strength = "Strong"
    elif absolute_value >= 0.50:
        strength = "Moderate"
    elif absolute_value >= 0.30:
        strength = "Weak"
    else:
        strength = "Very weak"

    if value > 0:
        direction = "positive"
    elif value < 0:
        direction = "negative"
    else:
        direction = "none"

    return f"{strength} {direction}"

def significance_label(p_value):
    """Return a human-readable significance label."""
    if p_value < 0.001:
        return "Highly significant (p < 0.001)"
    if p_value < 0.01:
        return "Very significant (p < 0.01)"
    if p_value < 0.05:
        return "Significant (p < 0.05)"
    return "Not significant (p ≥ 0.05)"

def interpret_vif(vif):
    """Classify a variance inflation factor."""
    if vif < 5:
        return "Low"
    if vif < 10:
        return "Moderate"
    return "High"

@st.cache_data
def get_chi_analysis_metadata(data: pd.DataFrame):
    """Cache reusable country and year filter options."""
    countries = sorted(
        data["country"].dropna().astype(str).unique().tolist()
    )
    years = sorted(
        data["year"].dropna().astype(int).unique().tolist()
    )
    return countries, years

@st.cache_data
def prepare_chi_overview_data(
    data: pd.DataFrame,
    selected_countries_tuple: tuple,
    selected_year: int,
):
    """Prepare all reusable Phase 1 country/year data in one cached call."""
    selected_mask = (
        data["country"].isin(selected_countries_tuple)
        & (data["year"] == selected_year)
    )
    year_mask = data["year"] == selected_year
    country_mask = data["country"].isin(selected_countries_tuple)

    country_year_df = data.loc[selected_mask].copy()
    global_year_df = data.loc[year_mask].copy()

    country_trend = (
        data.loc[country_mask]
        .groupby(["year", "country"], as_index=False)[
            "composite_health_index"
        ]
        .mean()
        .sort_values(["country", "year"])
    )

    country_comparison = (
        global_year_df.loc[
            global_year_df["country"].isin(selected_countries_tuple)
        ]
        .groupby("country", as_index=False)["composite_health_index"]
        .mean()
        .sort_values("composite_health_index", ascending=False)
    )

    return (
        country_year_df,
        global_year_df,
        country_trend,
        country_comparison,
    )

@st.cache_data
def prepare_chi_relationship_data(
    data: pd.DataFrame,
    relationship_column: str,
) -> pd.DataFrame:
    """Cache the selected CHI relationship dataset."""
    return data[["composite_health_index", relationship_column]].dropna()

@st.cache_data
def prepare_chi_audit_data(
    data: pd.DataFrame,
    audit_columns_tuple: tuple,
) -> pd.DataFrame:
    """Create and cache the country-year CHI audit dataset."""
    audit_data = data[list(audit_columns_tuple)].copy()
    audit_data["year"] = pd.to_numeric(
        audit_data["year"],
        errors="coerce",
    ).astype("Int64")

    return (
        audit_data
        .groupby(["country", "year"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["country", "year"])
        .reset_index(drop=True)
    )

@st.cache_data
def prepare_chi_correlation_results(
    audit_data: pd.DataFrame,
    correlation_variables_tuple: tuple,
) -> pd.DataFrame:
    """Calculate and cache CHI correlation results."""
    correlation_variables = dict(correlation_variables_tuple)
    correlation_results = []

    for label, column in correlation_variables.items():
        correlation_data = audit_data[
            ["composite_health_index", column]
        ].dropna()

        if len(correlation_data) < 2:
            correlation_value = float("nan")
        else:
            correlation_value = correlation_data[
                "composite_health_index"
            ].corr(correlation_data[column])

        correlation_results.append(
            {
                "Variable": label,
                "Correlation with CHI": correlation_value,
                "Observations": len(correlation_data),
            }
        )

    result = pd.DataFrame(correlation_results)
    result["Interpretation"] = result[
        "Correlation with CHI"
    ].apply(interpret_correlation)
    result["Correlation with CHI"] = result[
        "Correlation with CHI"
    ].round(3)
    result["Absolute Correlation"] = result[
        "Correlation with CHI"
    ].abs()

    return (
        result
        .sort_values("Absolute Correlation", ascending=False)
        .drop(columns=["Absolute Correlation"])
        .reset_index(drop=True)
    )

@st.cache_data
def prepare_regression_data(
    audit_data: pd.DataFrame,
    regression_columns_tuple: tuple,
) -> pd.DataFrame:
    """Prepare and cache the complete country-year regression dataset."""
    columns = list(regression_columns_tuple)
    return audit_data[
        ["composite_health_index", *columns]
    ].dropna().copy()

@st.cache_data
def prepare_vif_results(
    audit_data: pd.DataFrame,
    regression_columns_tuple: tuple,
    regression_variables_tuple: tuple,
) -> pd.DataFrame:
    """Calculate and cache VIF results."""
    columns = list(regression_columns_tuple)
    variable_labels = dict(regression_variables_tuple)
    vif_data = audit_data[columns].dropna().copy()

    if len(vif_data) <= len(columns):
        return pd.DataFrame()

    vif_input = sm.add_constant(vif_data)
    vif_results = []

    for index, column in enumerate(vif_input.columns):
        if column == "const":
            continue

        vif_value = variance_inflation_factor(
            vif_input.values,
            index,
        )

        vif_results.append(
            {
                "Variable": variable_labels.get(column, column),
                "VIF": vif_value,
            }
        )

    result = pd.DataFrame(vif_results)
    result["Multicollinearity"] = result["VIF"].apply(interpret_vif)
    result["VIF"] = result["VIF"].round(2)

    return result.sort_values(
        "VIF",
        ascending=False,
    ).reset_index(drop=True)

@st.cache_data
def prepare_feature_importance(
    regression_data: pd.DataFrame,
    regression_columns_tuple: tuple,
    regression_variables_tuple: tuple,
) -> pd.DataFrame:
    """Calculate and cache standardized regression coefficients."""
    columns = list(regression_columns_tuple)
    variable_labels = dict(regression_variables_tuple)

    importance_data = regression_data[
        columns + ["composite_health_index"]
    ].dropna().copy()

    standardized_data = (
        importance_data - importance_data.mean()
    ) / importance_data.std()

    X_standardized = sm.add_constant(
        standardized_data[columns]
    )
    y_standardized = standardized_data[
        "composite_health_index"
    ]

    standardized_model = sm.OLS(
        y_standardized,
        X_standardized,
    ).fit()

    importance_results = []
    for column in columns:
        coefficient = standardized_model.params[column]
        p_value = standardized_model.pvalues[column]

        importance_results.append(
            {
                "Variable": variable_labels.get(column, column),
                "Standardized Coefficient": coefficient,
                "Absolute Importance": abs(coefficient),
                "P-Value": p_value,
            }
        )

    result = pd.DataFrame(importance_results)
    result = (
        result
        .sort_values("Absolute Importance", ascending=False)
        .reset_index(drop=True)
    )
    result["Significance"] = result["P-Value"].apply(
        lambda p: "Significant" if p < 0.05 else "Not Significant"
    )
    return result

@st.cache_data
def prepare_reverse_engineering_data(
    data: pd.DataFrame,
    reverse_columns_tuple: tuple,
) -> pd.DataFrame:
    """Prepare and cache the raw disease-level reverse-engineering dataset."""
    columns = list(reverse_columns_tuple)
    return data[
        columns + ["composite_health_index"]
    ].dropna().copy()

# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------

st.title("📊 Composite Health Index Analysis")

# ---------------------------------------------------------
# ---------------------------------------------------------
# PHASE TABS
# ---------------------------------------------------------

(
    phase1_tab,
    phase2_tab,
    phase3_tab,
    residual_tab,
    feature_importance_tab,
    reverse_engineering_tab,
) = st.tabs([
    "CHI Overview",
    "CHI Relationships",
    "CHI Regression Analysis",
    "Residual Analysis",
    "CHI Feature Importance",
    "CHI Reverse Engineering",
])

with phase1_tab:
    # PHASE 1 — CHI OVERVIEW
    # ---------------------------------------------------------

    st.divider()

    st.header("Phase 1 — CHI Overview")

    st.markdown(
        """
        This phase provides a descriptive overview of the
        Composite Health Index across countries and years.
        """
    )

    st.divider()

    # ---------------------------------------------------------
    # Prepare filter options
    # ---------------------------------------------------------

    countries, years = get_chi_analysis_metadata(df)

    # ---------------------------------------------------------
    # Controls
    # ---------------------------------------------------------

    st.subheader("Analysis Controls")

    col1, col2 = st.columns(2)

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
                "Composite Health Index trends."
            ),
        )

    with col2:

        selected_year = st.selectbox(
            "Select year",
            years,
            index=len(years) - 1,
        )

    # ---------------------------------------------------------
    # Validate country selection
    # ---------------------------------------------------------

    if not selected_countries:

        st.warning(
            "Please select at least one country to continue."
        )

        st.stop()

    # ---------------------------------------------------------
    # Prepare Phase 1 overview data
    # ---------------------------------------------------------

    (
        country_year_df,
        global_year_df,
        country_trend,
        country_comparison,
    ) = prepare_chi_overview_data(
        df,
        tuple(selected_countries),
        selected_year,
    )

    # ---------------------------------------------------------
    # KPI calculations

    # ---------------------------------------------------------
    # KPI calculations
    # ---------------------------------------------------------

    selected_chi = (
        country_year_df[
            "composite_health_index"
        ].mean()
    )

    global_chi = (
        global_year_df[
            "composite_health_index"
        ].mean()
    )

    chi_difference = selected_chi - global_chi

    # ---------------------------------------------------------
    # KPI cards
    # ---------------------------------------------------------

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Selected Countries",
            len(selected_countries),
        )

    with col2:

        st.metric(
            "Average Selected CHI",
            f"{selected_chi:.2f}",
        )

    with col3:

        st.metric(
            "Global Average CHI",
            f"{global_chi:.2f}",
        )

    with col4:

        st.metric(
            "Difference from Global",
            f"{chi_difference:+.2f}",
        )

    # ---------------------------------------------------------
    # Selected countries
    # ---------------------------------------------------------

    st.caption(
        f"Selected countries: {', '.join(selected_countries)}"
    )

    # ---------------------------------------------------------
    # Multi-country CHI trend
    # ---------------------------------------------------------

    st.divider()
    st.subheader(
        "CHI Trend — Selected Countries"
    )

    if country_trend.empty:

        st.info(
            "No CHI trend data is available for the selected countries."
        )

    else:

        fig_chi_trend = px.line(
            country_trend,
            x="year",
            y="composite_health_index",
            color="country",
            markers=True,
            title="Composite Health Index Trend",
            labels={
                "year": "Year",
                "composite_health_index": "Composite Health Index",
                "country": "Country",
            },
        )

        fig_chi_trend.update_layout(
            height=500,
        )

        st.plotly_chart(
            fig_chi_trend,
            use_container_width=True,
        )

    # ---------------------------------------------------------
    # Country CHI comparison
    # ---------------------------------------------------------

    st.divider()
    st.subheader(
        f"Country CHI Comparison — {selected_year}"
    )

    if country_comparison.empty:

        st.info(
            f"No CHI comparison data is available for {selected_year}."
        )

    else:

        fig_chi_comparison = px.bar(
            country_comparison,
            x="country",
            y="composite_health_index",
            title=(
                f"Country Composite Health Index Comparison — "
                f"{selected_year}"
            ),
            labels={
                "country": "Country",
                "composite_health_index": "Composite Health Index",
            },
        )

        fig_chi_comparison.update_layout(
            height=450,
        )

        st.plotly_chart(
            fig_chi_comparison,
            use_container_width=True,
        )

        comparison_display = (
            country_comparison
            .rename(
                columns={
                    "country": "Country",
                    "composite_health_index":
                        "Composite Health Index",
                }
            )
            .copy()
        )

        comparison_display[
            "Composite Health Index"
        ] = comparison_display[
            "Composite Health Index"
        ].round(2)

        st.dataframe(
            comparison_display,
            use_container_width=True,
            hide_index=True,
        )

    # ---------------------------------------------------------
    # CHI distribution
    # ---------------------------------------------------------

    st.divider()

    st.subheader("CHI Distribution")

    fig_distribution = px.histogram(
        df,
        x="composite_health_index",
        nbins=30,
        labels={
            "composite_health_index": (
                "Composite Health Index"
            ),
            "count": "Number of Records",
        },
    )

    fig_distribution.update_layout(
        height=400,
    )

    st.plotly_chart(
        fig_distribution,
        use_container_width=True,
    )

    # ---------------------------------------------------------

with phase2_tab:
    # PHASE 2 — CHI RELATIONSHIPS
    # ---------------------------------------------------------

    st.divider()

    st.header("🔗 Phase 2 — CHI Relationships")

    st.markdown(
        """
        This phase examines how the Composite Health Index
        relates to key health and socioeconomic indicators.
        """
    )

    # ---------------------------------------------------------
    # CHI relationships
    # ---------------------------------------------------------

    st.divider()

    st.subheader("CHI Relationships")

    relationship_options = {
        "Healthcare Access": "healthcare_access_pct",
        "Doctors per 1,000": "doctors_per_1000",
        "Hospital Beds per 1,000": (
            "hospital_beds_per_1000"
        ),
        "Per Capita Income": (
            "per_capita_income_usd"
        ),
        "Education Index": "education_index",
        "Urbanization Rate": (
            "urbanization_rate_pct"
        ),
    }

    selected_relationship = st.selectbox(
        "Select indicator",
        list(relationship_options.keys()),
    )

    relationship_column = (
        relationship_options[
            selected_relationship
        ]
    )

    relationship_df = prepare_chi_relationship_data(
        df,
        relationship_column,
    )

    fig_relationship = px.scatter(
        relationship_df,
        x=relationship_column,
        y="composite_health_index",
        trendline="ols",
        labels={
            relationship_column: selected_relationship,
            "composite_health_index": (
                "Composite Health Index"
            ),
        },
    )

    fig_relationship.update_layout(
        height=500,
    )

    st.plotly_chart(
        fig_relationship,
        use_container_width=True,
    )

    # ---------------------------------------------------------
    # CHI summary
    # ---------------------------------------------------------

    st.divider()

    st.subheader("CHI Summary")

    if len(selected_countries) == 1:

        selected_country = selected_countries[0]

        st.markdown(
            f"""
            **{selected_country}** has an average Composite
            Health Index of **{selected_chi:.2f}** in
            **{int(selected_year)}**, compared with a global
            average of **{global_chi:.2f}**.

            The difference between the selected country's CHI
            and the global average is
            **{chi_difference:+.2f} points**.
            """
        )

    else:

        st.markdown(
            f"""
            The selected **{len(selected_countries)} countries**
            have an average Composite Health Index of
            **{selected_chi:.2f}** in **{int(selected_year)}**.

            The global average CHI for the same year is
            **{global_chi:.2f}**, giving the selected countries
            an average difference of **{chi_difference:+.2f} points**.
            """
        )

    # ---------------------------------------------------------
    # CHI Construction Audit — Country-Year Dataset
    # ---------------------------------------------------------

    st.divider()

    st.subheader("🔍 CHI Construction Audit")

    st.markdown(
        """
        This section prepares a country-year-level dataset for
        examining whether the Composite Health Index (CHI) can be
        explained or reconstructed from the underlying health and
        socioeconomic indicators.
        """
    )

    # ---------------------------------------------------------
    # Variables used in the CHI audit
    # ---------------------------------------------------------

    chi_audit_columns = [
        "country",
        "year",
        "composite_health_index",
        "healthcare_access_pct",
        "doctors_per_1000",
        "hospital_beds_per_1000",
        "recovery_rate_pct",
        "per_capita_income_usd",
        "education_index",
        "urbanization_rate_pct",
    ]

    # ---------------------------------------------------------
    # Check that required columns exist
    # ---------------------------------------------------------

    missing_audit_columns = [
        column
        for column in chi_audit_columns
        if column not in df.columns
    ]

    if missing_audit_columns:

        st.error(
            "The following columns required for the CHI audit "
            "are missing from the dataset:"
        )

        st.write(
            missing_audit_columns
        )

    else:

        # -----------------------------------------------------
        # Create cached audit dataset
        # -----------------------------------------------------

        chi_audit_df = prepare_chi_audit_data(
            df,
            tuple(chi_audit_columns),
        )

        # -----------------------------------------------------
        # Audit dataset metrics

        # -----------------------------------------------------
        # Audit dataset metrics
        # -----------------------------------------------------

        audit_col1, audit_col2, audit_col3 = (
            st.columns(3)
        )

        with audit_col1:

            st.metric(
                "Country-Year Observations",
                f"{len(chi_audit_df):,}",
            )

        with audit_col2:

            st.metric(
                "Countries",
                chi_audit_df[
                    "country"
                ].nunique(),
            )

        with audit_col3:

            st.metric(
                "Years",
                chi_audit_df[
                    "year"
                ].nunique(),
            )

        # -----------------------------------------------------
        # Preview
        # -----------------------------------------------------

        st.caption(
            """
            Each row now represents one country in one year.
            Disease-level records have been aggregated so that
            each country-year receives equal weight in the audit.
            """
        )

        st.dataframe(
            chi_audit_df,
            use_container_width=True,
            hide_index=True,
        )

        # -----------------------------------------------------
        # Missing values in audit variables
        # -----------------------------------------------------

        st.subheader(
            "CHI Audit Variable Completeness"
        )

        audit_missing = (
            chi_audit_df
            .isna()
            .sum()
            .reset_index()
        )

        audit_missing.columns = [
            "Variable",
            "Missing Values",
        ]

        audit_missing[
            "Complete (%)"
        ] = (
            (
                1
                - (
                    audit_missing[
                        "Missing Values"
                    ]
                    / len(chi_audit_df)
                )
            )
            * 100
        ).round(2)

        st.dataframe(
            audit_missing,
            use_container_width=True,
            hide_index=True,
        )

    # ---------------------------------------------------------
    # CHI Correlation Analysis
    # ---------------------------------------------------------

    st.divider()

    st.subheader("📈 CHI Correlation Analysis")

    st.markdown(
        """
        This analysis measures the Pearson correlation between the
        Composite Health Index (CHI) and the health and socioeconomic
        variables used in the CHI construction audit.

        Correlation values range from -1 to +1. Values closer to +1
        indicate a strong positive linear relationship, while values
        closer to -1 indicate a strong negative relationship.
        Values near 0 indicate a weak or negligible linear relationship.
        """
    )

    # ---------------------------------------------------------
    # Correlation variables
    # ---------------------------------------------------------

    correlation_variables = {
        "Healthcare Access": "healthcare_access_pct",
        "Doctors per 1,000": "doctors_per_1000",
        "Hospital Beds per 1,000": "hospital_beds_per_1000",
        "Recovery Rate": "recovery_rate_pct",
        "Per Capita Income": "per_capita_income_usd",
        "Education Index": "education_index",
        "Urbanization Rate": "urbanization_rate_pct",
    }

    # ---------------------------------------------------------
    # Calculate and cache correlations
    # ---------------------------------------------------------

    correlation_df = prepare_chi_correlation_results(
        chi_audit_df,
        tuple(correlation_variables.items()),
    )

    # ---------------------------------------------------------
    # Display correlation table

    # ---------------------------------------------------------
    # Display correlation table
    # ---------------------------------------------------------

    st.dataframe(
        correlation_df,
        use_container_width=True,
        hide_index=True,
    )

    # ---------------------------------------------------------
    # Correlation bar chart
    # ---------------------------------------------------------

    correlation_chart_df = (
        correlation_df
        .sort_values(
            "Correlation with CHI"
        )
    )

    fig_correlation = px.bar(
        correlation_chart_df,
        x="Correlation with CHI",
        y="Variable",
        orientation="h",
        title="Correlation Between CHI and Underlying Variables",
        labels={
            "Correlation with CHI": "Pearson Correlation",
            "Variable": "Variable",
        },
    )

    fig_correlation.update_layout(
        height=500,
        xaxis_range=[-1, 1],
    )

    st.plotly_chart(
        fig_correlation,
        use_container_width=True,
    )

    # ---------------------------------------------------------
    # Strongest correlation
    # ---------------------------------------------------------

    valid_correlations = correlation_df.dropna(
        subset=[
            "Correlation with CHI"
        ]
    )

    if not valid_correlations.empty:

        strongest_row = (
            valid_correlations
            .iloc[0]
        )

        strongest_variable = (
            strongest_row[
                "Variable"
            ]
        )

        strongest_value = (
            strongest_row[
                "Correlation with CHI"
            ]
        )

        strongest_interpretation = (
            strongest_row[
                "Interpretation"
            ]
        )

        st.info(
            f"""
            **Strongest relationship:** {strongest_variable}

            Pearson correlation with CHI:
            **{strongest_value:+.3f}**

            Interpretation:
            **{strongest_interpretation}**
            """
        )

    # ---------------------------------------------------------

with phase3_tab:
    # PHASE 3 — CHI REGRESSION ANALYSIS
    # ---------------------------------------------------------

    st.divider()

    st.header("📈 Phase 3 — CHI Regression Analysis")

    st.markdown(
        """
        This phase uses multiple linear regression to determine
        how the selected health and socioeconomic variables
        collectively explain variation in CHI.
        """
    )

    # ---------------------------------------------------------
    # CHI Multiple Linear Regression
    # ---------------------------------------------------------

    st.divider()

    st.subheader("CHI Multiple Linear Regression")

    st.markdown(
        """
        This regression model examines whether the selected health
        and socioeconomic indicators can collectively explain the
        Composite Health Index (CHI).

        The model estimates the contribution of each predictor while
        holding the other predictors constant.
        """
    )

    # ---------------------------------------------------------
    # Regression predictors
    # ---------------------------------------------------------

    regression_variables = {
        "Healthcare Access": "healthcare_access_pct",
        "Doctors per 1,000": "doctors_per_1000",
        "Hospital Beds per 1,000": "hospital_beds_per_1000",
        "Recovery Rate": "recovery_rate_pct",
        "Per Capita Income": "per_capita_income_usd",
        "Education Index": "education_index",
        "Urbanization Rate": "urbanization_rate_pct",
    }

    regression_columns = list(
        regression_variables.values()
    )

    # ---------------------------------------------------------
    # Prepare regression dataset
    # ---------------------------------------------------------

    regression_df = prepare_regression_data(
        chi_audit_df,
        tuple(regression_columns),
    )

    # ---------------------------------------------------------
    # Check observations
    # ---------------------------------------------------------

    if len(regression_df) < len(regression_columns) + 2:

        st.warning(
            "There are not enough complete observations to "
            "run the multiple linear regression model."
        )

    else:

        # -----------------------------------------------------
        # Define dependent and independent variables
        # -----------------------------------------------------

        X = regression_df[
            regression_columns
        ]

        y = regression_df[
            "composite_health_index"
        ]

        # -----------------------------------------------------
        # Add intercept
        # -----------------------------------------------------

        X_with_constant = sm.add_constant(
            X
        )

        # -----------------------------------------------------
        # Fit OLS regression model
        # -----------------------------------------------------

        regression_model = sm.OLS(
            y,
            X_with_constant,
        ).fit()

        # -----------------------------------------------------
        # Model performance
        # -----------------------------------------------------

        regression_col1, regression_col2, regression_col3, regression_col4 = (
            st.columns(4)
        )

        with regression_col1:

            st.metric(
                "R²",
                f"{regression_model.rsquared:.4f}",
            )

        with regression_col2:

            st.metric(
                "Adjusted R²",
                f"{regression_model.rsquared_adj:.4f}",
            )

        with regression_col3:

            st.metric(
                "Observations",
                f"{int(regression_model.nobs):,}",
            )

        with regression_col4:

            st.metric(
                "Predictors",
                len(regression_columns),
            )

        # -----------------------------------------------------
        # Regression coefficient table
        # -----------------------------------------------------

        st.subheader(
            "Regression Coefficients"
        )

        coefficient_df = pd.DataFrame(
            {
                "Variable": [
                    "Intercept"
                ]
                + [
                    label
                    for label in regression_variables.keys()
                ],

                "Coefficient": regression_model.params.values,

                "Std. Error": regression_model.bse.values,

                "t-statistic": regression_model.tvalues.values,

                "p-value": regression_model.pvalues.values,
            }
        )

        # -----------------------------------------------------
        # Statistical significance
        # -----------------------------------------------------

        def significance_label(p_value):

            if p_value < 0.001:
                return "Highly significant (p < 0.001)"

            if p_value < 0.01:
                return "Very significant (p < 0.01)"

            if p_value < 0.05:
                return "Significant (p < 0.05)"

            return "Not significant (p ≥ 0.05)"

        coefficient_df[
            "Significance"
        ] = coefficient_df[
            "p-value"
        ].apply(
            significance_label
        )

        # -----------------------------------------------------
        # Round displayed values
        # -----------------------------------------------------

        coefficient_df[
            "Coefficient"
        ] = coefficient_df[
            "Coefficient"
        ].round(4)

        coefficient_df[
            "Std. Error"
        ] = coefficient_df[
            "Std. Error"
        ].round(4)

        coefficient_df[
            "t-statistic"
        ] = coefficient_df[
            "t-statistic"
        ].round(3)

        coefficient_df[
            "p-value"
        ] = coefficient_df[
            "p-value"
        ].round(6)

        st.dataframe(
            coefficient_df,
            use_container_width=True,
            hide_index=True,
        )

        # ---------------------------------------------------------
        # Model equation
        # ---------------------------------------------------------

        st.subheader(
            "Estimated Regression Equation"
        )

        equation_parts = [
            f"{regression_model.params['const']:.4f}"
        ]

        for label, column in regression_variables.items():

            coefficient = regression_model.params[column]

            sign = "+" if coefficient >= 0 else "-"

            equation_parts.append(
                f" {sign} "
                f"{abs(coefficient):.4f}"
                f" × "
                f"{label}"
            )

        regression_equation = (
            "CHI = "
            + "".join(equation_parts)
        )

        st.code(
            regression_equation,
            language="text",
        )

        # -----------------------------------------------------
        # Predicted CHI and residuals
        # -----------------------------------------------------

        regression_df[
            "predicted_chi"
        ] = regression_model.predict(
            X_with_constant
        )

        regression_df[
            "residual"
        ] = (
            regression_df[
                "composite_health_index"
            ]
            - regression_df[
                "predicted_chi"
            ]
        )

        # -----------------------------------------------------
        # Actual vs predicted CHI
        # -----------------------------------------------------

        st.subheader(
            "Actual vs Predicted CHI"
        )

        prediction_fig = px.scatter(
            regression_df,
            x="composite_health_index",
            y="predicted_chi",
            trendline="ols",
            labels={
                "composite_health_index": "Actual CHI",
                "predicted_chi": "Predicted CHI",
            },
            title="Actual CHI vs Predicted CHI",
        )

        prediction_fig.update_layout(
            height=500,
        )

        st.plotly_chart(
            prediction_fig,
            use_container_width=True,
        )

        # -----------------------------------------------------
        # Regression interpretation
        # -----------------------------------------------------

        significant_predictors = coefficient_df[
            (
                coefficient_df["p-value"] < 0.05
            )
            & (
                coefficient_df["Variable"]
                != "Intercept"
            )
        ]

        st.subheader(
            "Regression Interpretation"
        )

        if regression_model.rsquared >= 0.90:

            model_strength = (
                "very strong explanatory power"
            )

        elif regression_model.rsquared >= 0.70:

            model_strength = (
                "strong explanatory power"
            )

        elif regression_model.rsquared >= 0.50:

            model_strength = (
                "moderate explanatory power"
            )

        else:

            model_strength = (
                "limited explanatory power"
            )

        st.markdown(
            f"""
            The multiple linear regression model explains approximately
            **{regression_model.rsquared * 100:.2f}%** of the variation
            in CHI, indicating **{model_strength}**.

            The model contains **{len(regression_columns)} predictors**
            and is based on **{int(regression_model.nobs):,} complete
            country-year observations**.

            **{len(significant_predictors)} of the predictors are
            statistically significant at the 5% level.**
            """
        )

    # ---------------------------------------------------------
    # VIF / Multicollinearity Analysis
    # ---------------------------------------------------------

    st.divider()

    st.subheader("🔍 Multicollinearity Analysis — VIF")

    st.markdown(
        """
        Variance Inflation Factor (VIF) measures how strongly each
        predictor is related to the other predictors in the regression
        model.

        High VIF values indicate that predictors contain overlapping
        information, which can make individual regression coefficients
        less stable and harder to interpret.
        """
    )

    # ---------------------------------------------------------
    # Prepare predictors for VIF
    # ---------------------------------------------------------

    vif_data = chi_audit_df[
        regression_columns
    ].dropna().copy()

    # ---------------------------------------------------------
    # Check sufficient observations
    # ---------------------------------------------------------

    if len(vif_data) <= len(regression_columns):

        st.warning(
            "There are not enough complete observations to calculate VIF."
        )

    else:

        # -----------------------------------------------------
        # Calculate cached VIF results
        # -----------------------------------------------------

        vif_df = prepare_vif_results(
            chi_audit_df,
            tuple(regression_columns),
            tuple(regression_variables.items()),
        )

        # -----------------------------------------------------
        # Display VIF table

        # -----------------------------------------------------
        # Display VIF table
        # -----------------------------------------------------

        st.dataframe(
            vif_df,
            use_container_width=True,
            hide_index=True,
        )

        # -----------------------------------------------------
        # VIF chart
        # -----------------------------------------------------

        fig_vif = px.bar(
            vif_df,
            x="VIF",
            y="Variable",
            orientation="h",
            title="Variance Inflation Factor by Predictor",
            labels={
                "VIF": "Variance Inflation Factor",
                "Variable": "Predictor",
            },
        )

        fig_vif.update_layout(
            height=500,
        )

        st.plotly_chart(
            fig_vif,
            use_container_width=True,
        )

        # -----------------------------------------------------
        # VIF summary
        # -----------------------------------------------------

        high_vif = vif_df[
            vif_df["VIF"] >= 10
        ]

        moderate_vif = vif_df[
            (
                vif_df["VIF"] >= 5
            )
            & (
                vif_df["VIF"] < 10
            )
        ]

        low_vif = vif_df[
            vif_df["VIF"] < 5
        ]

        vif_col1, vif_col2, vif_col3 = (
            st.columns(3)
        )

        with vif_col1:

            st.metric(
                "Low VIF",
                len(low_vif),
            )

        with vif_col2:

            st.metric(
                "Moderate VIF",
                len(moderate_vif),
            )

        with vif_col3:

            st.metric(
                "High VIF",
                len(high_vif),
            )

        # -----------------------------------------------------
        # Interpretation
        # -----------------------------------------------------

        st.subheader(
            "VIF Interpretation"
        )

        if not high_vif.empty:

            high_variables = ", ".join(
                high_vif["Variable"].tolist()
            )

            st.warning(
                f"""
                **High multicollinearity detected.**

                The following predictors have VIF values of 10 or
                greater:

                **{high_variables}**

                These variables contain substantial overlapping
                information with other predictors in the model.
                """
            )

        elif not moderate_vif.empty:

            moderate_variables = ", ".join(
                moderate_vif["Variable"].tolist()
            )

            st.info(
                f"""
                **Moderate multicollinearity detected.**

                The following predictors have VIF values between
                5 and 10:

                **{moderate_variables}**
                """
            )

        else:

            st.success(
                """
                **No substantial multicollinearity detected.**

                All predictors have VIF values below 5.
                """
            )

with residual_tab:
    # ---------------------------------------------------------
    # Residual Analysis
    # ---------------------------------------------------------

    st.divider()

    st.subheader("📉 Residual Analysis")

    st.markdown(
        """
        Residuals represent the difference between the observed CHI
        and the CHI predicted by the regression model.

        Residual analysis helps determine whether the regression model
        captures the underlying relationship without obvious systematic
        patterns in its errors.
        """
    )

    # ---------------------------------------------------------
    # Check regression results exist
    # ---------------------------------------------------------

    if "regression_model" not in locals():

        st.warning(
            "Residual analysis requires a successfully fitted "
            "multiple linear regression model."
        )

    else:

        # -----------------------------------------------------
        # Reuse cached regression predictions and residuals
        # -----------------------------------------------------

        residual_df = regression_df.copy()
        residual_df["absolute_residual"] = (
            residual_df["residual"].abs()
        )

        # -----------------------------------------------------
        # Residual statistics

        # -----------------------------------------------------
        # Residual statistics
        # -----------------------------------------------------

        residual_col1, residual_col2, residual_col3, residual_col4 = (
            st.columns(4)
        )

        with residual_col1:

            st.metric(
                "Mean Residual",
                f"{residual_df['residual'].mean():.4f}",
            )

        with residual_col2:

            st.metric(
                "Residual Std. Dev.",
                f"{residual_df['residual'].std():.4f}",
            )

        with residual_col3:

            st.metric(
                "Maximum Residual",
                f"{residual_df['residual'].max():.4f}",
            )

        with residual_col4:

            st.metric(
                "Maximum Absolute Residual",
                f"{residual_df['absolute_residual'].max():.4f}",
            )

        # -----------------------------------------------------
        # Residuals vs predicted CHI
        # -----------------------------------------------------

        st.subheader(
            "Residuals vs Predicted CHI"
        )

        residual_scatter = px.scatter(
            residual_df,
            x="predicted_chi",
            y="residual",
            labels={
                "predicted_chi": "Predicted CHI",
                "residual": "Residual",
            },
            title="Residuals vs Predicted CHI",
        )

        residual_scatter.add_hline(
            y=0,
            line_dash="dash",
        )

        residual_scatter.update_layout(
            height=500,
        )

        st.plotly_chart(
            residual_scatter,
            use_container_width=True,
        )

        # -----------------------------------------------------
        # Residual distribution
        # -----------------------------------------------------

        st.subheader(
            "Residual Distribution"
        )

        residual_histogram = px.histogram(
            residual_df,
            x="residual",
            nbins=30,
            marginal="box",
            labels={
                "residual": "Residual",
            },
            title="Distribution of Regression Residuals",
        )

        residual_histogram.add_vline(
            x=0,
            line_dash="dash",
        )

        residual_histogram.update_layout(
            height=500,
        )

        st.plotly_chart(
            residual_histogram,
            use_container_width=True,
        )

        # -----------------------------------------------------
        # Largest residuals
        # -----------------------------------------------------

        st.subheader(
            "Largest Absolute Residuals"
        )

        largest_residuals = (
            residual_df[
                [
                    "composite_health_index",
                    "predicted_chi",
                    "residual",
                    "absolute_residual",
                ]
                + regression_columns
            ]
            .sort_values(
                "absolute_residual",
                ascending=False,
            )
            .head(10)
            .copy()
        )

        largest_residuals = (
            largest_residuals
            .rename(
                columns={
                    "composite_health_index": "Actual CHI",
                    "predicted_chi": "Predicted CHI",
                    "residual": "Residual",
                    "absolute_residual": "Absolute Residual",
                }
            )
        )

        largest_residuals[
            "Actual CHI"
        ] = largest_residuals[
            "Actual CHI"
        ].round(4)

        largest_residuals[
            "Predicted CHI"
        ] = largest_residuals[
            "Predicted CHI"
        ].round(4)

        largest_residuals[
            "Residual"
        ] = largest_residuals[
            "Residual"
        ].round(4)

        largest_residuals[
            "Absolute Residual"
        ] = largest_residuals[
            "Absolute Residual"
        ].round(4)

        st.dataframe(
            largest_residuals,
            use_container_width=True,
            hide_index=True,
        )

        # -----------------------------------------------------
        # Residual interpretation
        # -----------------------------------------------------

        st.subheader(
            "Residual Interpretation"
        )

        mean_residual = (
            residual_df[
                "residual"
            ].mean()
        )

        residual_std = (
            residual_df[
                "residual"
            ].std()
        )

        max_absolute_residual = (
            residual_df[
                "absolute_residual"
            ].max()
        )

        # -----------------------------------------------------
        # Mean residual assessment
        # -----------------------------------------------------

        if abs(mean_residual) < 0.01:

            mean_assessment = (
                "The mean residual is very close to zero, "
                "which is expected for an OLS model with an intercept."
            )

        elif abs(mean_residual) < 0.05:

            mean_assessment = (
                "The mean residual is relatively small, "
                "although there is some average prediction error."
            )

        else:

            mean_assessment = (
                "The mean residual is noticeably different from zero, "
                "which warrants further investigation."
            )

        st.write(
            mean_assessment
        )

        # -----------------------------------------------------
        # Residual spread assessment
        # -----------------------------------------------------

        st.write(
            f"""
            The residual standard deviation is approximately
            **{residual_std:.4f}**, while the largest absolute residual
            is approximately **{max_absolute_residual:.4f}**.
            """
        )

        # -----------------------------------------------------
        # Overall assessment
        # -----------------------------------------------------

        st.markdown(
            """
            **What to look for in the residual plot:**

            - Residuals should generally be scattered around zero.
            - There should not be a strong curve or systematic pattern.
            - The spread should ideally remain reasonably consistent
              across predicted CHI values.
            - A few large residuals may indicate observations that the
              model does not explain particularly well.
            """
        )

with feature_importance_tab:
    # ---------------------------------------------------------
    # Feature Importance Analysis
    # ---------------------------------------------------------

    st.divider()

    st.subheader("📊 CHI Feature Importance")

    st.markdown(
        """
        Feature importance shows the relative contribution of each
        predictor to the Composite Health Index (CHI).

        Because the predictors are measured on different scales,
        standardized regression coefficients are used to make their
        relative importance comparable.
        """
    )

    # ---------------------------------------------------------
    # Check regression model
    # ---------------------------------------------------------

    if "regression_model" not in locals():

        st.warning(
            "Feature importance requires a successfully fitted "
            "multiple linear regression model."
        )

    else:

        # -----------------------------------------------------
        # Prepare cached feature importance
        # -----------------------------------------------------

        importance_df = prepare_feature_importance(
            regression_df,
            tuple(regression_columns),
            tuple(regression_variables.items()),
        )

        importance_display = importance_df.copy()

        importance_display["Standardized Coefficient"] = (
            importance_display["Standardized Coefficient"].round(4)
        )
        importance_display["Absolute Importance"] = (
            importance_display["Absolute Importance"].round(4)
        )
        importance_display["P-Value"] = (
            importance_display["P-Value"].round(4)
        )

        # -----------------------------------------------------
        # Display table

        # -----------------------------------------------------
        # Display table
        # -----------------------------------------------------

        st.subheader(
            "Standardized Regression Coefficients"
        )

        st.dataframe(
            importance_display[
                [
                    "Variable",
                    "Standardized Coefficient",
                    "Absolute Importance",
                    "P-Value",
                    "Significance",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        # -----------------------------------------------------
        # Importance chart
        # -----------------------------------------------------

        st.subheader(
            "Relative Feature Importance"
        )

        fig_importance = px.bar(
            importance_df,
            x="Absolute Importance",
            y="Variable",
            orientation="h",
            title="Relative Importance of CHI Predictors",
            labels={
                "Absolute Importance": (
                    "Absolute Standardized Coefficient"
                ),
                "Variable": "Predictor",
            },
        )

        fig_importance.update_layout(
            height=500,
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True,
        )

        # -----------------------------------------------------
        # Most influential predictor
        # -----------------------------------------------------

        most_important = importance_df.iloc[0]

        st.info(
            f"""
            **Most influential predictor:** {most_important["Variable"]}

            Its standardized regression coefficient is
            **{most_important["Standardized Coefficient"]:.4f}**.

            This indicates that, after accounting for the other
            predictors in the model, this variable has the largest
            standardized contribution to CHI among the predictors
            included in the model.
            """
        )

        # -----------------------------------------------------
        # Positive and negative contributors
        # -----------------------------------------------------

        positive_features = importance_df[
            importance_df[
                "Standardized Coefficient"
            ] > 0
        ]

        negative_features = importance_df[
            importance_df[
                "Standardized Coefficient"
            ] < 0
        ]

        importance_col1, importance_col2 = (
            st.columns(2)
        )

        with importance_col1:

            st.markdown(
                "**Positive contributors**"
            )

            if not positive_features.empty:

                for _, row in positive_features.iterrows():

                    st.write(
                        f"• {row['Variable']}: "
                        f"{row['Standardized Coefficient']:.4f}"
                    )

            else:

                st.write(
                    "No positive contributors."
                )

        with importance_col2:

            st.markdown(
                "**Negative contributors**"
            )

            if not negative_features.empty:

                for _, row in negative_features.iterrows():

                    st.write(
                        f"• {row['Variable']}: "
                        f"{row['Standardized Coefficient']:.4f}"
                    )

            else:

                st.write(
                    "No negative contributors."
                )

        # -----------------------------------------------------
        # Interpretation
        # -----------------------------------------------------

        st.subheader(
            "Feature Importance Interpretation"
        )

        st.markdown(
            """
            **How to interpret this analysis:**

            - A larger absolute standardized coefficient indicates
              a stronger contribution to CHI after controlling for
              the other predictors.
            - A positive coefficient indicates that higher values
              of the predictor are associated with higher CHI,
              holding the other predictors constant.
            - A negative coefficient indicates an inverse relationship
              after controlling for the other predictors.
            - Statistical significance is assessed using a p-value
              threshold of 0.05.
            - Feature importance should be interpreted together with
              the correlation and VIF analyses rather than in isolation.
            """
        )

with reverse_engineering_tab:
    # ---------------------------------------------------------
    # CHI Reverse-Engineering Analysis
    # ---------------------------------------------------------

    st.divider()

    st.subheader("🔬 CHI Reverse-Engineering Analysis")

    st.markdown(
        """
        This analysis tests whether the Composite Health Index (CHI)
        can be reproduced from the health and socioeconomic variables
        used elsewhere in the dataset.

        The objective is to determine whether CHI behaves like an
        independent composite indicator or largely reflects a
        weighted combination of its underlying variables.
        """
    )

    # ---------------------------------------------------------
    # Variables used for reverse engineering
    # ---------------------------------------------------------

    reverse_engineering_variables = {
        "Healthcare Access": "healthcare_access_pct",
        "Recovery Rate": "recovery_rate_pct",
        "Education Index": "education_index",
        "Doctors per 1,000": "doctors_per_1000",
        "Hospital Beds per 1,000": "hospital_beds_per_1000",
        "Per Capita Income": "per_capita_income_usd",
        "Urbanization Rate": "urbanization_rate_pct",
    }

    reverse_columns = list(
        reverse_engineering_variables.values()
    )

    reverse_df = prepare_reverse_engineering_data(
        df,
        tuple(reverse_columns),
    )

    # Reverse engineering intentionally uses raw disease-level rows,
    # unlike the country-year CHI audit used by the main regression.

    # ---------------------------------------------------------
    # Check sufficient observations
    # ---------------------------------------------------------

    if len(reverse_df) < 10:

        st.warning(
            "There are not enough complete observations to perform "
            "the CHI reverse-engineering analysis."
        )

    else:

        # -----------------------------------------------------
        # Prepare target and predictors
        # -----------------------------------------------------

        X_reverse = reverse_df[
            reverse_columns
        ]

        y_reverse = reverse_df[
            "composite_health_index"
        ]

        # -----------------------------------------------------
        # Add intercept
        # -----------------------------------------------------

        X_reverse_constant = sm.add_constant(
            X_reverse
        )

        # -----------------------------------------------------
        # Fit reverse-engineering model
        # -----------------------------------------------------

        reverse_model = sm.OLS(
            y_reverse,
            X_reverse_constant,
        ).fit()

        # -----------------------------------------------------
        # Predictions and residuals
        # -----------------------------------------------------

        reverse_df[
            "predicted_chi"
        ] = reverse_model.predict(
            X_reverse_constant
        )

        reverse_df[
            "chi_residual"
        ] = (
            reverse_df[
                "composite_health_index"
            ]
            - reverse_df[
                "predicted_chi"
            ]
        )

        # -----------------------------------------------------
        # Model statistics
        # -----------------------------------------------------

        reverse_r2 = reverse_model.rsquared

        reverse_adjusted_r2 = (
            reverse_model.rsquared_adj
        )

        reverse_rmse = (
            (
                reverse_df["chi_residual"] ** 2
            ).mean()
            ** 0.5
        )

        reverse_mae = (
            reverse_df[
                "chi_residual"
            ]
            .abs()
            .mean()
        )

        # -----------------------------------------------------
        # KPI cards
        # -----------------------------------------------------

        st.subheader(
            "Reverse-Engineering Model Performance"
        )

        reverse_col1, reverse_col2, reverse_col3, reverse_col4 = (
            st.columns(4)
        )

        with reverse_col1:

            st.metric(
                "R²",
                f"{reverse_r2:.4f}",
            )

        with reverse_col2:

            st.metric(
                "Adjusted R²",
                f"{reverse_adjusted_r2:.4f}",
            )

        with reverse_col3:

            st.metric(
                "RMSE",
                f"{reverse_rmse:.4f}",
            )

        with reverse_col4:

            st.metric(
                "MAE",
                f"{reverse_mae:.4f}",
            )

        # -----------------------------------------------------
        # Regression coefficients
        # -----------------------------------------------------

        st.subheader(
            "Reverse-Engineering Coefficients"
        )

        reverse_coefficients = []

        for display_name, column in (
            reverse_engineering_variables.items()
        ):

            reverse_coefficients.append(
                {
                    "Variable": display_name,
                    "Coefficient": reverse_model.params[column],
                    "P-Value": reverse_model.pvalues[column],
                    "Significant": (
                        "Yes"
                        if reverse_model.pvalues[column] < 0.05
                        else "No"
                    ),
                }
            )

        reverse_coefficients_df = pd.DataFrame(
            reverse_coefficients
        )

        reverse_coefficients_df[
            "Coefficient"
        ] = (
            reverse_coefficients_df[
                "Coefficient"
            ].round(6)
        )

        reverse_coefficients_df[
            "P-Value"
        ] = (
            reverse_coefficients_df[
                "P-Value"
            ].round(6)
        )

        st.dataframe(
            reverse_coefficients_df,
            use_container_width=True,
            hide_index=True,
        )

        # -----------------------------------------------------
        # Estimated CHI equation
        # -----------------------------------------------------

        st.subheader(
            "Estimated CHI Reconstruction Equation"
        )

        intercept = reverse_model.params["const"]

        equation_parts = [
            f"{intercept:.4f}"
        ]

        for display_name, column in (
            reverse_engineering_variables.items()
        ):

            coefficient = reverse_model.params[column]

            sign = "+" if coefficient >= 0 else "-"

            equation_parts.append(
                f"{sign} "
                f"{abs(coefficient):.4f} × "
                f"{display_name}"
            )

        equation = (
            "Predicted CHI = "
            + " ".join(equation_parts)
        )

        st.code(
            equation,
            language="text",
        )

        # -----------------------------------------------------
        # Actual vs predicted CHI
        # -----------------------------------------------------

        st.subheader(
            "Actual CHI vs Predicted CHI"
        )

        fig_actual_predicted = px.scatter(
            reverse_df,
            x="composite_health_index",
            y="predicted_chi",
            trendline="ols",
            labels={
                "composite_health_index": (
                    "Actual CHI"
                ),
                "predicted_chi": (
                    "Predicted CHI"
                ),
            },
            title=(
                "Actual CHI vs Predicted CHI"
            ),
        )

        # Perfect prediction reference line
        min_value = min(
            reverse_df[
                "composite_health_index"
            ].min(),
            reverse_df[
                "predicted_chi"
            ].min(),
        )

        max_value = max(
            reverse_df[
                "composite_health_index"
            ].max(),
            reverse_df[
                "predicted_chi"
            ].max(),
        )

        fig_actual_predicted.add_shape(
            type="line",
            x0=min_value,
            y0=min_value,
            x1=max_value,
            y1=max_value,
            line=dict(
                dash="dash",
            ),
        )

        fig_actual_predicted.update_layout(
            height=500,
        )

        st.plotly_chart(
            fig_actual_predicted,
            use_container_width=True,
        )

        # -----------------------------------------------------
        # Prediction error distribution
        # -----------------------------------------------------

        st.subheader(
            "CHI Reconstruction Error Distribution"
        )

        fig_error = px.histogram(
            reverse_df,
            x="chi_residual",
            nbins=30,
            labels={
                "chi_residual": (
                    "Prediction Error"
                ),
                "count": "Number of Observations",
            },
            title=(
                "Distribution of CHI Reconstruction Errors"
            ),
        )

        fig_error.add_vline(
            x=0,
            line_dash="dash",
        )

        fig_error.update_layout(
            height=400,
        )

        st.plotly_chart(
            fig_error,
            use_container_width=True,
        )

        # -----------------------------------------------------
        # Residual statistics
        # -----------------------------------------------------

        st.subheader(
            "Reconstruction Error Statistics"
        )

        error_col1, error_col2, error_col3, error_col4 = (
            st.columns(4)
        )

        with error_col1:

            st.metric(
                "Mean Error",
                f"{reverse_df['chi_residual'].mean():.4f}",
            )

        with error_col2:

            st.metric(
                "Std. Dev. Error",
                f"{reverse_df['chi_residual'].std():.4f}",
            )

        with error_col3:

            st.metric(
                "Maximum Error",
                f"{reverse_df['chi_residual'].max():.4f}",
            )

        with error_col4:

            st.metric(
                "Maximum Absolute Error",
                f"{reverse_df['chi_residual'].abs().max():.4f}",
            )

        # -----------------------------------------------------
        # Interpretation
        # -----------------------------------------------------

        st.subheader(
            "CHI Reconstruction Interpretation"
        )

        if reverse_r2 >= 0.95:

            st.success(
                f"""
                The selected variables reproduce CHI with a very high
                R² of **{reverse_r2:.4f}**.

                This indicates that the Composite Health Index is
                strongly explained by the health and socioeconomic
                variables included in this reconstruction model.
                """
            )

        elif reverse_r2 >= 0.80:

            st.info(
                f"""
                The selected variables explain a substantial proportion
                of CHI variation, with an R² of **{reverse_r2:.4f}**.

                CHI is strongly related to these variables, although
                additional components may contribute to its construction.
                """
            )

        else:

            st.warning(
                f"""
                The selected variables explain an R² of
                **{reverse_r2:.4f}**.

                This suggests that CHI contains substantial variation
                that is not reproduced by the selected variables.
                """
            )

        # -----------------------------------------------------
        # Final assessment
        # -----------------------------------------------------

        st.subheader(
            "Overall CHI Construction Assessment"
        )

        if reverse_r2 >= 0.95 and reverse_rmse < 2:

            st.markdown(
                """
                ### 🔎 Strong evidence of reconstructability

                The regression model is able to reproduce CHI with very
                high explanatory power and relatively small prediction
                errors.

                This provides strong statistical evidence that CHI is
                largely determined by the health and socioeconomic
                variables included in this analysis.

                **Important:** This does not by itself prove the exact
                formula originally used to construct CHI. It demonstrates
                that CHI can be closely reconstructed from the selected
                variables.
                """
            )

        elif reverse_r2 >= 0.80:

            st.markdown(
                """
                ### 🔎 Moderate evidence of reconstructability

                The selected variables explain a substantial portion of
                CHI variation, but the reconstruction is not sufficiently
                precise to conclude that CHI is simply a direct weighted
                combination of these variables.
                """
            )

        else:

            st.markdown(
                """
                ### 🔎 Limited evidence of reconstructability

                The selected variables do not reproduce CHI sufficiently
                well to suggest that CHI is primarily determined by these
                variables alone.

                Additional variables, nonlinear relationships, or another
                construction methodology may be involved.
                """
            )

        # -----------------------------------------------------
        # Preview reconstructed values
        # -----------------------------------------------------

        st.subheader(
            "CHI Reconstruction Preview"
        )

        reconstruction_preview = reverse_df[
            [
                "composite_health_index",
                "predicted_chi",
                "chi_residual",
            ]
        ].head(20).copy()

        reconstruction_preview.columns = [
            "Actual CHI",
            "Predicted CHI",
            "Residual",
        ]

        reconstruction_preview = (
            reconstruction_preview.round(4)
        )

        st.dataframe(
            reconstruction_preview,
            use_container_width=True,
            hide_index=True,
        )
