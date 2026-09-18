"""
End-to-end data cleaning pipeline for the Global Health
Dataset.

This module orchestrates text cleaning, numeric conversion,
missing-value treatment, data repairs, validation checks,
and validation summary generation into a single reusable
workflow.

Column-name normalization is handled upstream by extract.py.
"""

import pandas as pd
from pathlib import Path
from .extract import load_data, prepare_dataframe
from .text_cleaning import clean_text_columns
from .numeric_cleaning import convert_numeric_columns
from .missing_values import handle_missing_values
from .validation import clean_and_validate_data
from .validation_report import build_validation_report


def run_pipeline(
    data: str | Path | pd.DataFrame,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    
    """
    Execute the complete data-cleaning pipeline.

    Data loading and column-name normalization are performed
    by the extraction stage.

    The pipeline performs the following stages:

    1. Data loading and column-name normalization.
    2. Text standardization.
    3. Numeric conversion.
    4. Missing-value handling.
    5. Data repairs.
    6. Data validation.
    7. Validation summary generation.

    Parameters
    ----------
    file_path : str
        Path to the raw Global Health Dataset.

    Returns
    -------
    clean_df : pandas.DataFrame
        Fully cleaned and validated dataset.

    reports : dict
        Nested dictionary containing reports from every
        pipeline stage.

    summary_report : pandas.DataFrame
        High-level validation summary suitable for
        reporting and documentation.
    """

    # --------------------------------------------------
    # Loading data and normalizing column names
    # --------------------------------------------------
    if isinstance(data, pd.DataFrame):

        df, metadata = prepare_dataframe(
            data,
            source_file="Uploaded Dataset",
        )

    else:

        df, metadata = load_data(data)

    # --------------------------------------------------
    # Text cleaning
    # --------------------------------------------------
    df = clean_text_columns(df)

    # --------------------------------------------------
    # Numeric cleaning
    # --------------------------------------------------
    df, numeric_conversion_report = (
        convert_numeric_columns(df)
    )

    # --------------------------------------------------
    # Missing-value handling
    # --------------------------------------------------
    df, missing_value_reports = (
        handle_missing_values(df)
    )

    # --------------------------------------------------
    # Repairs and validations
    # --------------------------------------------------
    df, validation_results = (
        clean_and_validate_data(df)
    )

    # --------------------------------------------------
    # Combine all reports
    # --------------------------------------------------
    reports = {
        "pipeline": {
            "source_file": metadata.get(
                "source_file"
            ),
            "rows_after_pipeline": len(df),
            "columns_after_pipeline": len(df.columns),
        },
        "numeric_conversion": (
            numeric_conversion_report
        ),
        "missing_values": (
            missing_value_reports
        ),
        "repairs": (
            validation_results["repairs"]
        ),
        "validations": (
            validation_results["validations"]
        ),
    }

    # --------------------------------------------------
    # Build validation summary
    # --------------------------------------------------
    summary_report = build_validation_report(
        reports
    )

    return (
        df,
        reports,
        summary_report,
    )



























# """
# End-to-end data cleaning pipeline for the Global Health
# Dataset.

# This module orchestrates text cleaning, numeric conversion,
# missing-value treatment, data repairs, validation checks,
# and validation summary generation into a single reusable
# workflow.

# Column-name normalization is handled upstream by extract.py.
# """

# import pandas as pd

# from extract import load_data
# from text_cleaning import clean_text_columns
# from numeric_cleaning import convert_numeric_columns
# from missing_values import handle_missing_values
# from validation import clean_and_validate_data
# from validation_report import build_validation_report


# def run_pipeline(
#     file_path: str,
# ) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    
#     """
#     Execute the complete data-cleaning pipeline.

#     Data loading and column-name normalization are performed
#     by the extraction stage.

#     The pipeline performs the following stages:

#     1. Data loading and column-name normalization.
#     2. Text standardization.
#     3. Numeric conversion.
#     4. Missing-value handling.
#     5. Data repairs.
#     6. Data validation.
#     7. Validation summary generation.

#     Parameters
#     ----------
#     file_path : str
#         Path to the raw Global Health Dataset.

#     Returns
#     -------
#     clean_df : pandas.DataFrame
#         Fully cleaned and validated dataset.

#     reports : dict
#         Nested dictionary containing reports from every
#         pipeline stage.

#     summary_report : pandas.DataFrame
#         High-level validation summary suitable for
#         reporting and documentation.
#     """

#     # --------------------------------------------------
#     # Loading data and normalizing column names
#     # --------------------------------------------------
#     df, _ = load_data(file_path)

#     # --------------------------------------------------
#     # Text cleaning
#     # --------------------------------------------------
#     df = clean_text_columns(df)

#     # --------------------------------------------------
#     # Numeric cleaning
#     # --------------------------------------------------
#     df, numeric_conversion_report = (
#         convert_numeric_columns(df)
#     )

#     # --------------------------------------------------
#     # Missing-value handling
#     # --------------------------------------------------
#     df, missing_value_reports = (
#         handle_missing_values(df)
#     )

#     # --------------------------------------------------
#     # Repairs and validations
#     # --------------------------------------------------
#     df, validation_results = (
#         clean_and_validate_data(df)
#     )

#     # --------------------------------------------------
#     # Combine all reports
#     # --------------------------------------------------
#     reports = {
#         "pipeline": {
#             "rows_after_pipeline": len(df),
#             "columns_after_pipeline": len(df.columns),
#         },
#         "numeric_conversion": (
#             numeric_conversion_report
#         ),
#         "missing_values": (
#             missing_value_reports
#         ),
#         "repairs": (
#             validation_results["repairs"]
#         ),
#         "validations": (
#             validation_results["validations"]
#         ),
#     }

#     # --------------------------------------------------
#     # Build validation summary
#     # --------------------------------------------------
#     summary_report = build_validation_report(
#         reports
#     )

#     return (
#         df,
#         reports,
#         summary_report,
#     )

