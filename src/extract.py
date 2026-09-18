"""
Utilities for loading and preparing the Global Health Dataset.

This module provides functions for reading raw data files,
validating the raw schema, normalizing column names, and
preparing already-loaded DataFrames for the cleaning pipeline.
"""

from pathlib import Path

import pandas as pd

from .config import (
    EXPECTED_COLUMNS,
    COLUMN_RENAME_MAP,
    NA_VALUES,
)


def prepare_dataframe(
    df: pd.DataFrame,
    source_file: str = "DataFrame",
) -> tuple[pd.DataFrame, dict]:
    """
    Validate and prepare an already-loaded DataFrame.

    This function is used when data has already been loaded
    by another interface, such as Streamlit file upload.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw DataFrame to prepare.

    source_file : str, default="DataFrame"
        Name used to identify the source in metadata.

    Returns
    -------
    df : pandas.DataFrame
        Prepared and column-normalized DataFrame.

    metadata : dict
        Metadata describing the source DataFrame.

    Raises
    ------
    ValueError
        If the DataFrame contains a different number of
        columns, missing expected columns, or unexpected columns.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Expected a pandas DataFrame."
        )

    # Work on a copy so the caller's DataFrame is not modified.
    df = df.copy()

    # --------------------------------------------------
    # Convert configured missing-value representations
    # --------------------------------------------------
    df = df.replace(NA_VALUES, pd.NA)

    # --------------------------------------------------
    # Validate raw column count
    # --------------------------------------------------
    if len(df.columns) != len(EXPECTED_COLUMNS):
        raise ValueError(
            f"Expected {len(EXPECTED_COLUMNS)} columns "
            f"but found {len(df.columns)}."
        )

    # --------------------------------------------------
    # Check for missing expected columns
    # --------------------------------------------------
    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    # --------------------------------------------------
    # Check for unexpected columns
    # --------------------------------------------------
    unexpected_columns = [
        column
        for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if unexpected_columns:
        raise ValueError(
            f"Unexpected columns: {unexpected_columns}"
        )

    # --------------------------------------------------
    # Normalize column names
    # --------------------------------------------------
    df = df.rename(
        columns=COLUMN_RENAME_MAP
    )

    # --------------------------------------------------
    # Build metadata
    # --------------------------------------------------
    metadata = {
        "source_file": source_file,
        "source_path": None,
        "encoding": None,
        "rows": len(df),
        "columns": len(df.columns),
        "shape": df.shape,
        "memory_mb": float(
            round(
                df.memory_usage(deep=True).sum()
                / 1024**2,
                2,
            )
        ),
    }

    return df, metadata


def load_data(
    file_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Load and prepare the Global Health Dataset from a CSV file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the raw CSV dataset.

    Returns
    -------
    df : pandas.DataFrame
        Loaded and column-normalized dataset.

    metadata : dict
        Metadata describing the source file.
    """

    csv_path = Path(file_path)

    # --------------------------------------------------
    # Check that the dataset exists
    # --------------------------------------------------
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    # --------------------------------------------------
    # Load the raw dataset
    # --------------------------------------------------
    encoding = "cp1252"

    df = pd.read_csv(
        csv_path,
        encoding=encoding,
        low_memory=False,
        keep_default_na=False,
        na_values=NA_VALUES,
    )

    # --------------------------------------------------
    # Prepare and validate DataFrame
    # --------------------------------------------------
    df, metadata = prepare_dataframe(
        df,
        source_file=csv_path.name,
    )

    # --------------------------------------------------
    # Add file-specific metadata
    # --------------------------------------------------
    metadata["source_path"] = str(csv_path)
    metadata["encoding"] = encoding

    return df, metadata