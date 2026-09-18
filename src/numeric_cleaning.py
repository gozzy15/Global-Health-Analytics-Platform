"""
Utilities for cleaning and converting numeric fields in the
Global Health Dataset.

This module converts numeric values stored as text into
proper numeric data types while tracking conversion
statistics for each processed column.
"""

import numpy as np
import pandas as pd

from .config import NUMERIC_COLUMNS


def clean_numeric_value(value) -> float:
    """
    Clean and convert a single value to a numeric value.

    The function handles common formatting characters such
    as commas, currency symbols, percentage signs, and
    apostrophes. Missing or non-convertible values are
    returned as NaN.

    Parameters
    ----------
    value : Any
        Value to clean and convert.

    Returns
    -------
    float
        Numeric representation of the value, or NaN if the
        value is missing or cannot be converted.
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "":
        return np.nan

    # Remove common numeric formatting characters.
    for char in ["'", ",", "$", "%"]:
        value = value.replace(char, "")

    try:
        return float(value)

    except ValueError:
        return np.nan


def convert_numeric_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Convert configured numeric columns to numeric data types.

    Each column listed in NUMERIC_COLUMNS is processed using
    clean_numeric_value() before being converted with
    pandas.to_numeric().

    A conversion report is generated for every processed
    column, including:

    - Original data type
    - Final data type
    - Missing values before conversion
    - Missing values after conversion
    - New missing values introduced during conversion

    Columns listed in NUMERIC_COLUMNS but absent from the
    input DataFrame are skipped.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the numeric columns to convert.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        DataFrame with configured numeric columns converted
        to numeric data types.

    conversion_report : list of dict
        Conversion statistics for each processed numeric
        column.
    """

    df = df.copy()

    conversion_report = []

    # --------------------------------------------------
    # Process each configured numeric column
    # --------------------------------------------------
    for column in NUMERIC_COLUMNS:

        if column not in df.columns:
            continue

        # --------------------------------------------------
        # Record original state
        # --------------------------------------------------
        original_dtype = df[column].dtype

        missing_before = df[column].isna().sum()

        # --------------------------------------------------
        # Clean individual values
        # --------------------------------------------------
        df[column] = df[column].apply(
            clean_numeric_value
        )

        # --------------------------------------------------
        # Convert column to numeric dtype
        # --------------------------------------------------
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        # --------------------------------------------------
        # Record conversion results
        # --------------------------------------------------
        missing_after = df[column].isna().sum()

        new_missing = max(
            0,
            missing_after - missing_before,
        )

        final_dtype = df[column].dtype

        # --------------------------------------------------
        # Store conversion report
        # --------------------------------------------------
        conversion_report.append(
            {
                "Column": column,
                "Original dtype": str(original_dtype),
                "Final dtype": str(final_dtype),
                "Missing Before": int(missing_before),
                "Missing After": int(missing_after),
                "New NaNs": int(new_missing),
            }
        )

    return df, conversion_report