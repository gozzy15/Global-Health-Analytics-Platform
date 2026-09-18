"""
Calculation utilities for the Global Health Dashboard.
"""

import pandas as pd


# ---------------------------------------------------------
# Global summaries
# ---------------------------------------------------------

def calculate_global_averages(
    df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """
    Calculate yearly global averages for selected metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Health dataset.
    metrics : list[str]
        Numeric columns to average.

    Returns
    -------
    pd.DataFrame
        Yearly global averages.
    """

    available_metrics = [
        metric for metric in metrics
        if metric in df.columns
    ]

    return (
        df.groupby("year")[available_metrics]
        .mean()
        .reset_index()
        .sort_values("year")
    )


# ---------------------------------------------------------
# Country calculations
# ---------------------------------------------------------

def calculate_country_average(
    df: pd.DataFrame,
    country: str,
    metric: str,
) -> float:
    """
    Calculate the average value of a metric for one country.
    """

    country_data = df[df["country"] == country]

    if country_data.empty or metric not in country_data.columns:
        return float("nan")

    return country_data[metric].mean()


def calculate_country_year_average(
    df: pd.DataFrame,
    country: str,
    year: int,
    metric: str,
) -> float:
    """
    Calculate the average value of a metric for a
    specific country and year.
    """

    filtered = df[
        (df["country"] == country)
        & (df["year"] == year)
    ]

    if filtered.empty or metric not in filtered.columns:
        return float("nan")

    return filtered[metric].mean()


# ---------------------------------------------------------
# Disease calculations
# ---------------------------------------------------------

def calculate_disease_average(
    df: pd.DataFrame,
    disease: str,
    metric: str,
) -> float:
    """
    Calculate the average value of a metric for one disease.
    """

    disease_data = df[df["disease_name"] == disease]

    if disease_data.empty or metric not in disease_data.columns:
        return float("nan")

    return disease_data[metric].mean()


# ---------------------------------------------------------
# CHI calculations
# ---------------------------------------------------------

def calculate_chi_average(
    df: pd.DataFrame,
) -> float:
    """
    Calculate the overall average Composite Health Index.
    """

    if "composite_health_index" not in df.columns:
        return float("nan")

    return df["composite_health_index"].mean()


def calculate_country_chi(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate average Composite Health Index by country.
    """

    return (
        df.groupby("country")["composite_health_index"]
        .mean()
        .reset_index()
        .sort_values(
            "composite_health_index",
            ascending=False,
        )
    )


def calculate_yearly_chi(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate average Composite Health Index by year.
    """

    return (
        df.groupby("year")["composite_health_index"]
        .mean()
        .reset_index()
        .sort_values("year")
    )


# ---------------------------------------------------------
# Percentage change
# ---------------------------------------------------------

def calculate_percentage_change(
    start_value: float,
    end_value: float,
) -> float:
    """
    Calculate percentage change between two values.

    Returns NaN when the starting value is zero.
    """

    if pd.isna(start_value) or pd.isna(end_value):
        return float("nan")

    if start_value == 0:
        return float("nan")

    return ((end_value - start_value) / start_value) * 100