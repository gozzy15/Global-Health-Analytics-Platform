"""
Utilities for cleaning and standardizing text fields in the
Global Health Dataset.

This module normalizes textual values and applies
column-specific standardization for countries, diseases,
treatment types, and vaccine/treatment availability.
"""

import unicodedata
from typing import Any

import pandas as pd

from .config import (
    COUNTRY_CORRECTIONS,
    DISEASE_CORRECTIONS,
    TREATMENT_MAP,
    AVAILABILITY_MAPPING,
)


def normalize_text(value: Any) -> Any:
    """
    Normalize a single text value.

    The function:
    - preserves missing values;
    - converts non-missing values to strings;
    - normalizes Unicode characters;
    - removes leading and trailing whitespace;
    - collapses consecutive whitespace characters.

    Parameters
    ----------
    value : Any
        Value to normalize.

    Returns
    -------
    Any
        Normalized string value, or the original missing value
        if the input is missing.
    """

    if pd.isna(value):
        return value

    value = str(value)

    value = unicodedata.normalize("NFKC", value)

    value = value.strip()

    value = " ".join(value.split())

    return value


def clean_country(series: pd.Series) -> pd.Series:
    """
    Standardize country names.

    The function normalizes text values and applies the
    configured country correction mappings.

    Parameters
    ----------
    series : pandas.Series
        Country values to clean.

    Returns
    -------
    pandas.Series
        Cleaned country values.
    """

    series = series.apply(normalize_text)

    series = series.replace(COUNTRY_CORRECTIONS)

    return series


def clean_disease(series: pd.Series) -> pd.Series:
    """
    Standardize disease names.

    The function normalizes text values and applies the
    configured disease correction mappings.

    Parameters
    ----------
    series : pandas.Series
        Disease names to clean.

    Returns
    -------
    pandas.Series
        Cleaned disease names.
    """

    series = series.apply(normalize_text)

    series = series.replace(DISEASE_CORRECTIONS)

    return series


def clean_treatment_type(series: pd.Series) -> pd.Series:
    """
    Standardize treatment type values.

    Values are normalized, converted to lowercase, and then
    mapped to the project's standardized treatment categories.

    Parameters
    ----------
    series : pandas.Series
        Treatment type values to clean.

    Returns
    -------
    pandas.Series
        Standardized treatment type values.
    """

    series = series.apply(normalize_text)

    series = series.str.lower()

    series = series.replace(TREATMENT_MAP)

    return series


def clean_availability(series: pd.Series) -> pd.Series:
    """
    Standardize vaccine/treatment availability categories.

    Values are normalized, converted to lowercase, and then
    mapped to the project's standardized availability
    categories.

    Parameters
    ----------
    series : pandas.Series
        Vaccine/treatment availability values to clean.

    Returns
    -------
    pandas.Series
        Standardized availability values.
    """

    series = series.apply(normalize_text)

    series = series.str.lower()

    series = series.replace(AVAILABILITY_MAPPING)

    return series


def clean_text_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply all configured text-cleaning operations.

    The function standardizes the following columns:

    - country
    - disease_name
    - treatment_type
    - availability_of_vaccines_treatment

    A copy of the input DataFrame is created so that the
    original DataFrame is not modified in place.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the text columns to clean.

    Returns
    -------
    pandas.DataFrame
        DataFrame with standardized text values.
    """

    df = df.copy()

    df["country"] = clean_country(
        df["country"]
    )

    df["disease_name"] = clean_disease(
        df["disease_name"]
    )

    df["treatment_type"] = clean_treatment_type(
        df["treatment_type"]
    )

    df["availability_of_vaccines_treatment"] = (
        clean_availability(
            df["availability_of_vaccines_treatment"]
        )
    )

    return df