"""
Reusable Plotly chart functions for the Global Health Dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------
# Overview charts
# ---------------------------------------------------------

def create_health_index_trend(df: pd.DataFrame):
    """
    Create a line chart showing average Composite Health Index
    by year.
    """

    trend = (
        df.groupby(
            "year",
            as_index=False
        )["composite_health_index"]
        .mean()
        .sort_values("year")
    )

    fig = px.line(
        trend,
        x="year",
        y="composite_health_index",
        markers=True,
        title="Global Composite Health Index Trend",
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Average Composite Health Index",
        hovermode="x unified",
    )

    return fig


def create_disease_burden_chart(df: pd.DataFrame):
    """
    Create a bar chart showing average DALYs by disease.
    """

    burden = (
        df.groupby(
            "disease_name",
            as_index=False
        )["dalys"]
        .mean()
        .sort_values(
            "dalys",
            ascending=False
        )
    )

    fig = px.bar(
        burden,
        x="disease_name",
        y="dalys",
        title="Average Disease Burden by Disease",
    )

    fig.update_layout(
        xaxis_title="Disease",
        yaxis_title="Average DALYs",
        xaxis_tickangle=-45,
    )

    return fig


# ---------------------------------------------------------
# Country analysis charts
# ---------------------------------------------------------

def country_trend_chart(
    df: pd.DataFrame,
    country: str,
    metric: str,
    title: str,
):
    """
    Create a historical trend chart for a selected country
    and indicator.

    Parameters
    ----------
    df : pandas.DataFrame
        Health dataset.

    country : str
        Selected country.

    metric : str
        Dataset column containing the selected indicator.

    title : str
        Chart title.
    """

    filtered = (
        df[df["country"] == country]
        .groupby(
            "year",
            as_index=False
        )[metric]
        .mean()
        .sort_values("year")
    )

    fig = px.line(
        filtered,
        x="year",
        y=metric,
        markers=True,
        title=title,
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=metric.replace(
            "_",
            " "
        ).title(),
        hovermode="x unified",
    )

    return fig


def country_comparison_chart(
    df: pd.DataFrame,
    metric: str,
    year: int,
    title: str,
):
    """
    Create a country comparison chart for a selected
    indicator and year.
    """

    filtered = (
        df[df["year"] == year]
        .groupby(
            "country",
            as_index=False
        )[metric]
        .mean()
        .sort_values(
            metric,
            ascending=False
        )
    )

    fig = px.bar(
        filtered,
        x="country",
        y=metric,
        title=title,
    )

    fig.update_layout(
        xaxis_title="Country",
        yaxis_title=metric.replace(
            "_",
            " "
        ).title(),
        xaxis_tickangle=-45,
    )

    return fig


# ---------------------------------------------------------
# Disease analysis charts
# ---------------------------------------------------------

def disease_trend_chart(
    df: pd.DataFrame,
    diseases: list[str],
    metric: str,
    title: str,
):
    """
    Create a historical trend chart for multiple selected
    diseases and an indicator.

    Each selected disease is displayed as a separate line.
    """

    filtered = (
        df[
            df["disease_name"].isin(diseases)
        ]
        .groupby(
            ["year", "disease_name"],
            as_index=False
        )[metric]
        .mean()
        .sort_values(
            ["disease_name", "year"]
        )
    )

    fig = px.line(
        filtered,
        x="year",
        y=metric,
        color="disease_name",
        markers=True,
        title=title,
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=metric.replace(
            "_",
            " "
        ).title(),
        hovermode="x unified",
        legend_title="Disease",
    )

    return fig


def disease_comparison_chart(
    df: pd.DataFrame,
    metric: str,
    year: int,
    title: str,
):
    """
    Create a disease comparison chart for a selected
    indicator and year.
    """

    filtered = (
        df[df["year"] == year]
        .groupby(
            "disease_name",
            as_index=False
        )[metric]
        .mean()
        .sort_values(
            metric,
            ascending=False
        )
    )

    fig = px.bar(
        filtered,
        x="disease_name",
        y=metric,
        title=title,
    )

    fig.update_layout(
        xaxis_title="Disease",
        yaxis_title=metric.replace(
            "_",
            " "
        ).title(),
        xaxis_tickangle=-45,
    )

    return fig

# ---------------------------------------------------------
# Trends analysis charts
# ---------------------------------------------------------

def trends_analysis_chart(
    df: pd.DataFrame,
    countries: list[str],
    diseases: list[str],
    metric: str,
    title: str,
):
    """
    Create a historical trend chart for the selected
    countries and diseases.

    Each selected country is displayed as a separate line.
    When multiple diseases are selected, their values are
    averaged within each country-year combination.

    Parameters
    ----------
    df : pandas.DataFrame
        Health dataset.

    countries : list[str]
        Selected countries.

    diseases : list[str]
        Selected diseases.

    metric : str
        Dataset column containing the selected indicator.

    title : str
        Chart title.
    """

    filtered = df.copy()

    # -----------------------------------------------------
    # Apply country filter
    # -----------------------------------------------------

    if countries:
        filtered = filtered[
            filtered["country"].isin(countries)
        ]

    # -----------------------------------------------------
    # Apply disease filter
    # -----------------------------------------------------

    if diseases:
        filtered = filtered[
            filtered["disease_name"].isin(diseases)
        ]

    # -----------------------------------------------------
    # Aggregate by country and year
    # -----------------------------------------------------

    trend = (
        filtered.groupby(
            ["year", "country"],
            as_index=False
        )[metric]
        .mean()
        .sort_values(
            ["country", "year"]
        )
    )

    # -----------------------------------------------------
    # Create chart
    # -----------------------------------------------------

    fig = px.line(
        trend,
        x="year",
        y=metric,
        color="country",
        markers=True,
        title=title,
    )

    # -----------------------------------------------------
    # Chart formatting
    # -----------------------------------------------------

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=metric.replace(
            "_",
            " "
        ).title(),
        hovermode="x unified",
        legend_title="Country",
    )

    return fig