"""
Utilities for identifying, preserving, and imputing missing
values in the Global Health Dataset.

This module provides functions for handling missing data
using dataset-specific strategies, including peer-country
imputation, country-wise interpolation, structural
missing-value preservation, and identifier validation.
"""

import pandas as pd

from .config import (
    IDENTIFIER_COLUMNS,
    STRUCTURAL_MISSING_COLUMNS,
    PEER_COUNTRIES,
)


def summarize_missing(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize missing values in a DataFrame.

    Only columns containing at least one missing value are
    included in the returned summary. Results are sorted by
    descending missing count.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to inspect for missing values.

    Returns
    -------
    pandas.DataFrame
        Missing-value summary containing:

        - Missing Count
        - Missing Percent
    """

    missing = df.isna().sum()

    summary = pd.DataFrame(
        {
            "Missing Count": missing,
            "Missing Percent": (
                missing / len(df) * 100
            ).round(2),
        }
    )

    summary = summary[
        summary["Missing Count"] > 0
    ]

    summary = summary.sort_values(
        by="Missing Count",
        ascending=False,
    )

    return summary


def drop_missing_identifiers(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Remove rows missing essential identifier fields.

    Rows are removed when any configured identifier column
    contains a missing value. The configured identifiers are
    defined by ``IDENTIFIER_COLUMNS``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        DataFrame with rows missing required identifiers removed.

    report : dict
        Summary containing the number of rows before and after
        removal, rows removed, and rows containing missing
        identifiers.
    """

    df = df.copy()

    identifier_missing = (
        df[IDENTIFIER_COLUMNS]
        .isna()
        .any(axis=1)
        .sum()
    )

    rows_before = len(df)

    df = df.dropna(
        subset=IDENTIFIER_COLUMNS
    )

    rows_after = len(df)

    rows_removed = rows_before - rows_after

    report = {
        "Rows Before": rows_before,
        "Rows After": rows_after,
        "Rows Removed": rows_removed,
        "Rows With Missing Identifiers": (
            int(identifier_missing)
        ),
    }

    return df, report


# NOTE:
# This function was developed during exploratory data cleaning.
# It is not currently used in the final pipeline because the
# relevant columns were either treated as structural missing
# values or imputed using peer-country medians. It is retained
# as a reusable utility for future projects.


def fill_with_group_median(
    df: pd.DataFrame,
    column: str,
    group_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Fill missing values using group and global medians.

    Missing values are first replaced with the median calculated
    within the specified groups. Any values that remain missing
    are then replaced with the overall median of the target column.

    This function is retained as a reusable imputation utility
    but is not currently used in the final Global Health Dataset
    cleaning pipeline.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    column : str
        Name of the column containing missing values.

    group_cols : list[str]
        Column names defining the groups used to calculate
        group-specific medians.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        DataFrame with eligible missing values imputed.

    report : dict
        Summary containing the target column, missing values
        before and after imputation, and the number of values
        filled using group and global medians.
    """

    df = df.copy()

    missing_before = df[column].isna().sum()

    group_medians = (
        df.groupby(group_cols)[column]
        .transform("median")
    )

    df[column] = df[column].fillna(
        group_medians
    )

    missing_after_group = df[column].isna().sum()

    filled_by_group = (
        missing_before - missing_after_group
    )

    global_median = df[column].median()

    df[column] = df[column].fillna(
        global_median
    )

    missing_after = df[column].isna().sum()

    filled_by_global = (
        missing_after_group - missing_after
    )

    report = {
        "Column": column,
        "Missing Before": int(missing_before),
        "Filled by Group Median": int(filled_by_group),
        "Filled by Global Median": int(filled_by_global),
        "Missing After": int(missing_after),
    }

    return df, report


def identify_structural_missingness(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Identify columns containing structural missing values.

    Structural missing values are reported but deliberately
    preserved as NaN because their absence is considered part
    of the dataset's underlying structure rather than a value
    that should be statistically imputed.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to inspect.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        Unmodified copy of the input DataFrame.

    report : list of dict
        Per-column summary containing the number of structural
        missing values and the action taken.
    """

    df = df.copy()

    report = []

    for column in STRUCTURAL_MISSING_COLUMNS:

        missing = df[column].isna().sum()

        report.append(
            {
                "Column": column,
                "Structural Missing": int(missing),
                "Action": "Preserve as NaN",
            }
        )

    return df, report


def fill_peer_country_median(
    df: pd.DataFrame,
    column: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Fill missing values using designated peer countries.

    For each target country defined in ``PEER_COUNTRIES``,
    missing values are replaced with the median value from
    its designated peer countries for the same year and
    disease.

    A value is filled only when at least one valid peer-country
    value is available. Otherwise, the missing value is preserved.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing country, year, disease,
        and target-value columns.

    column : str
        Name of the numeric column whose missing values should
        be imputed.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        Copy of the input DataFrame with eligible missing values
        filled using peer-country medians.

    report : dict
        Summary containing the target column, missing values
        before and after imputation, and the number of values
        successfully filled.

    Notes
    -----
    Peer-country mappings are defined by ``PEER_COUNTRIES``.
    Peer values are matched by both ``year`` and
    ``disease_name`` to preserve contextual comparability.
    """

    df = df.copy()

    missing_before = df[column].isna().sum()

    filled = 0

    for target_country, peers in PEER_COUNTRIES.items():

        missing_rows = df[
            (df["country"] == target_country)
            &
            (df[column].isna())
        ]

        for index, row in missing_rows.iterrows():

            peer_values = df[
                (df["country"].isin(peers))
                &
                (df["year"] == row["year"])
                &
                (df["disease_name"] == row["disease_name"])
            ][column]

            median_value = peer_values.median()

            if pd.notna(median_value):

                df.at[index, column] = median_value

                filled += 1

    missing_after = df[column].isna().sum()

    report = {
        "Column": column,
        "Missing Before": int(missing_before),
        "Filled": int(filled),
        "Missing After": int(missing_after),
    }

    return df, report


def impute_per_capita_income(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing per-capita income values by country.

    Missing values are filled using linear interpolation across
    years within each country. The original row order is restored
    after interpolation.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing ``per_capita_income_usd``,
        ``country``, and ``year`` columns.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        DataFrame with eligible missing income values interpolated.

    report : dict
        Summary containing the target column, missing values before
        and after imputation, and the number of values filled.
    """

    df = df.copy()

    column = "per_capita_income_usd"

    missing_before = df[column].isna().sum()

    # Ensure interpolation operates on numeric data.
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    # Preserve original order.
    df["_original_order"] = df.index

    # Sort so interpolation follows chronological order.
    df = df.sort_values(
        by=["country", "year"]
    )

    # Interpolate within each country.
    df[column] = (
        df.groupby("country")[column]
        .transform(
            lambda s: s.interpolate(
                method="linear"
            )
        )
    )

    missing_after = df[column].isna().sum()

    filled = missing_before - missing_after

    # Restore original order.
    df = (
        df.sort_values("_original_order")
        .drop(columns="_original_order")
        .reset_index(drop=True)
    )

    report = {
        "Column": column,
        "Missing Before": int(missing_before),
        "Filled by Interpolation": int(filled),
        "Missing After": int(missing_after),
    }

    return df, report


def impute_education_index(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing education index values by country.

    Missing values are first filled using linear interpolation
    across years within each country. Remaining missing values
    are then filled using backward and forward filling within
    each country.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing ``education_index``,
        ``country``, and ``year`` columns.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        DataFrame with eligible Education Index values imputed.

    report : dict
        Summary containing the target column, imputation method,
        missing values before and after treatment, and number
        of values filled.
    """

    df = df.copy()

    column = "education_index"

    missing_before = df[column].isna().sum()

    # Ensure interpolation operates on numeric data.
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    df["_original_order"] = df.index

    df = df.sort_values(
        by=["country", "year"]
    )

    df[column] = (
        df.groupby("country")[column]
        .transform(
            lambda s: (
                s.interpolate(
                    method="linear"
                )
                .bfill()
                .ffill()
            )
        )
    )

    missing_after = df[column].isna().sum()

    filled = missing_before - missing_after

    df = (
        df.sort_values("_original_order")
        .drop(columns="_original_order")
        .reset_index(drop=True)
    )

    report = {
        "Column": column,
        "Method": "Country-wise linear interpolation",
        "Missing Before": int(missing_before),
        "Filled": int(filled),
        "Missing After": int(missing_after),
    }

    return df, report


def preserve_improvement_missing(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Preserve structural missing values in the five-year
    improvement metric.

    Missing values in ``improvement_in_5_years_pct`` are retained
    as NaN because the earliest years do not contain sufficient
    historical data to calculate the five-year improvement.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        Unmodified copy of the input DataFrame.

    report : dict
        Summary containing the target column, treatment action,
        and missing-value counts before and after treatment.
    """

    df = df.copy()

    column = "improvement_in_5_years_pct"

    missing_before = df[column].isna().sum()

    report = {
        "Column": column,
        "Action": (
            "Preserved structural missing values "
            "(earliest years lack sufficient historical data)"
        ),
        "Missing Before": int(missing_before),
        "Missing After": int(missing_before),
    }

    return df, report


def handle_missing_values(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Execute the complete missing-value treatment pipeline.

    The pipeline:

    1. Removes rows missing essential identifiers.
    2. Imputes healthcare access using peer-country medians.
    3. Imputes doctors per 1000 using peer-country medians.
    4. Imputes hospital beds per 1000 using peer-country medians.
    5. Interpolates per-capita income within each country.
    6. Interpolates education index within each country.
    7. Identifies structural missing values.
    8. Preserves structural missing values in the five-year
       improvement metric.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw or partially cleaned DataFrame.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        DataFrame after missing-value treatment.

    reports : dict
        Dictionary containing the report generated by each
        missing-value treatment step.
    """

    reports = {}

    # --------------------------------------------------
    # Remove rows missing essential identifiers
    # --------------------------------------------------
    df, reports["identifier_rows"] = (
        drop_missing_identifiers(df)
    )

    # --------------------------------------------------
    # Healthcare Access
    # --------------------------------------------------
    df, reports["healthcare_access"] = (
        fill_peer_country_median(
            df,
            column="healthcare_access_pct",
        )
    )

    # --------------------------------------------------
    # Doctors per 1000
    # --------------------------------------------------
    df, reports["doctors_per_1000"] = (
        fill_peer_country_median(
            df,
            column="doctors_per_1000",
        )
    )

    # --------------------------------------------------
    # Hospital Beds per 1000
    # --------------------------------------------------
    df, reports["hospital_beds_per_1000"] = (
        fill_peer_country_median(
            df,
            column="hospital_beds_per_1000",
        )
    )

    # --------------------------------------------------
    # Per Capita Income
    # --------------------------------------------------
    df, reports["per_capita_income"] = (
        impute_per_capita_income(df)
    )

    # --------------------------------------------------
    # Education Index
    # --------------------------------------------------
    df, reports["education_index"] = (
        impute_education_index(df)
    )

    # --------------------------------------------------
    # Structural missing values
    # --------------------------------------------------
    df, reports["structural_missing"] = (
        identify_structural_missingness(df)
    )

    # --------------------------------------------------
    # Improvement in 5 Years
    # --------------------------------------------------
    df, reports["improvement"] = (
        preserve_improvement_missing(df)
    )

    return df, reports