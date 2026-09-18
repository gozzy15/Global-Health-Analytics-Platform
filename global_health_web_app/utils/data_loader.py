"""
Data loading utilities for the Global Health Dashboard.
"""

from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Default dataset path
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_DATA_PATH = (
    BASE_DIR
    / "data"
    / "Global Health Dataset_cleaned.csv"
)


# ---------------------------------------------------------
# Load default dataset
# ---------------------------------------------------------

@st.cache_data
def load_default_health_data() -> pd.DataFrame:
    """
    Load the application's default health dataset.

    Raises
    ------
    FileNotFoundError
        If the default dataset does not exist.

    ValueError
        If the dataset is empty.

    RuntimeError
        If the dataset cannot be read successfully.
    """

    if not DEFAULT_DATA_PATH.exists():
        raise FileNotFoundError(
            "The default Global Health dataset could not be found. "
            "Please verify that the file exists at: "
            f"{DEFAULT_DATA_PATH}"
        )

    try:

        df = pd.read_csv(
            DEFAULT_DATA_PATH
        )

    except pd.errors.EmptyDataError as error:

        raise ValueError(
            "The default Global Health dataset is empty."
        ) from error

    except pd.errors.ParserError as error:

        raise RuntimeError(
            "The default Global Health dataset could not be "
            "parsed correctly. The CSV file may be corrupted "
            "or incorrectly formatted."
        ) from error

    except OSError as error:

        raise RuntimeError(
            "The default Global Health dataset could not be "
            "read from disk."
        ) from error

    if df.empty:

        raise ValueError(
            "The default Global Health dataset contains no records."
        )

    return df


# ---------------------------------------------------------
# Main data loader
# ---------------------------------------------------------

def load_health_data() -> pd.DataFrame:
    """
    Return the active dataset.

    If the user has uploaded a dataset during the current
    session,
    that dataset is returned.

    Otherwise, the default Global Health dataset is used.

    Raises
    ------
    ValueError
        If the uploaded dataset is empty or invalid.
    """

    if "uploaded_data" in st.session_state:

        uploaded_df = st.session_state["uploaded_data"]

        if uploaded_df is not None:

            if not isinstance(
                uploaded_df,
                pd.DataFrame,
            ):
                raise ValueError(
                    "The uploaded dataset is not in a valid "
                    "tabular format."
                )

            if uploaded_df.empty:

                raise ValueError(
                    "The uploaded dataset contains no records."
                )

            return uploaded_df.copy()

    return load_default_health_data().copy()


# ---------------------------------------------------------
# Dataset metadata helpers
# ---------------------------------------------------------

def get_countries(df: pd.DataFrame) -> list:
    """
    Return sorted unique country names when available.
    """

    if "country" not in df.columns:
        return []

    return sorted(
        df["country"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def get_years(df: pd.DataFrame) -> list:
    """
    Return sorted unique years when available.
    """

    if "year" not in df.columns:
        return []

    years = pd.to_numeric(
        df["year"],
        errors="coerce",
    ).dropna()

    return sorted(
        years.astype(int).unique().tolist()
    )


def get_diseases(df: pd.DataFrame) -> list:
    """
    Return sorted unique disease names when available.
    """

    if "disease_name" not in df.columns:
        return []

    return sorted(
        df["disease_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


# ---------------------------------------------------------
# Reset uploaded dataset
# ---------------------------------------------------------

def clear_uploaded_data() -> None:
    """
    Remove the uploaded dataset from the current session.
    """

    keys_to_remove = [
        "uploaded_data",
        "uploaded_filename",
        "uploaded_sheet",
        "uploaded_table",
    ]

    for key in keys_to_remove:

        if key in st.session_state:
            del st.session_state[key]