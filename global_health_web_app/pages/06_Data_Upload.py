"""
Data Upload page for the Global Health Dashboard.

Uploaded datasets are passed through the project's existing
data-cleaning and validation pipeline before they are made
available to the analysis pages.
"""

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_pipeline

from utils.style import apply_global_styles


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Data Upload | Global Health Dashboard",
    page_icon="📁",
    layout="wide",
)

apply_global_styles()

# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------

st.title("📁 Data Upload")

st.markdown(
    """
    Upload a health dataset in CSV, Excel, or SQLite format.
    Uploaded data is automatically processed through the
    Global Health data-cleaning and validation pipeline before
    it is used for analysis.
    """
)

st.divider()


# ---------------------------------------------------------
# Upload section
# ---------------------------------------------------------

st.subheader("Upload Dataset")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=[
        "csv",
        "xlsx",
        "xls",
        "db",
        "sqlite",
        "sqlite3",
    ],
    help=(
        "Supported formats: CSV, Excel, and SQLite "
        "database files."
    ),
)


# ---------------------------------------------------------
# Stop if no file has been uploaded
# ---------------------------------------------------------

if uploaded_file is None:

    st.info(
        "Upload a CSV, Excel, or SQLite file to begin."
    )

    st.stop()


# ---------------------------------------------------------
# File information
# ---------------------------------------------------------

file_name = uploaded_file.name

file_extension = (
    Path(file_name)
    .suffix
    .lower()
    .replace(".", "")
)

st.success(
    f"Successfully uploaded: **{file_name}**"
)

st.caption(
    "The file has been uploaded successfully. "
    "The dataset structure will now be checked before "
    "processing."
)


# ---------------------------------------------------------
# Read uploaded file
# ---------------------------------------------------------

df = None
selected_sheet = None
selected_table = None


# ---------------------------------------------------------
# CSV processing
# ---------------------------------------------------------

if file_extension == "csv":

    try:

        uploaded_file.seek(0)

        try:

            df = pd.read_csv(
                uploaded_file,
                encoding="utf-8",
            )

        except UnicodeDecodeError:

            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                encoding="cp1252",
            )

    except Exception as exc:

        st.error(
            f"Unable to read the CSV file: {exc}"
        )

        st.stop()


# ---------------------------------------------------------
# Excel processing
# ---------------------------------------------------------

elif file_extension in ["xlsx", "xls"]:

    try:

        uploaded_file.seek(0)

        excel_file = pd.ExcelFile(
            uploaded_file
        )

        sheet_names = excel_file.sheet_names

        if not sheet_names:

            st.error(
                "No worksheets were found in the Excel file."
            )

            st.stop()

        selected_sheet = st.selectbox(
            "Select worksheet",
            sheet_names,
        )

        uploaded_file.seek(0)

        df = pd.read_excel(
            uploaded_file,
            sheet_name=selected_sheet,
        )

    except Exception as exc:

        st.error(
            f"Unable to read the Excel file: {exc}"
        )

        st.stop()


# ---------------------------------------------------------
# SQLite processing
# ---------------------------------------------------------

elif file_extension in [
    "db",
    "sqlite",
    "sqlite3",
]:

    try:

        database_bytes = uploaded_file.getvalue()

        # ---------------------------------------------
        # Create temporary database
        # ---------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".db",
            delete=False,
        ) as temp_database:

            temp_database.write(database_bytes)

            temporary_database_path = (
                Path(temp_database.name)
            )

        # ---------------------------------------------
        # Discover tables
        # ---------------------------------------------

        connection = sqlite3.connect(
            temporary_database_path
        )

        tables_query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
        """

        tables = pd.read_sql_query(
            tables_query,
            connection,
        )

        connection.close()

        if tables.empty:

            st.error(
                "No tables were found in the SQLite database."
            )

            temporary_database_path.unlink(
                missing_ok=True
            )

            st.stop()

        selected_table = st.selectbox(
            "Select table",
            tables["name"].tolist(),
        )

        # ---------------------------------------------
        # Load selected table
        # ---------------------------------------------

        connection = sqlite3.connect(
            temporary_database_path
        )

        df = pd.read_sql_query(
            f'SELECT * FROM "{selected_table}"',
            connection,
        )

        connection.close()

        temporary_database_path.unlink(
            missing_ok=True
        )

    except Exception as exc:

        st.error(
            f"Unable to read the SQLite database: {exc}"
        )

        st.stop()


# ---------------------------------------------------------
# Validate that data was loaded
# ---------------------------------------------------------

if df is None or df.empty:

    st.error(
        "The uploaded dataset is empty or could not be loaded."
    )

    st.stop()


# ---------------------------------------------------------
# Validate dataset structure
# ---------------------------------------------------------

if len(df.columns) == 0:

    st.error(
        "The uploaded dataset does not contain any columns."
    )

    st.stop()


if len(df) < 2:

    st.error(
        "The uploaded dataset must contain at least "
        "2 rows of data."
    )

    st.stop()


# ---------------------------------------------------------
# Validate column names
# ---------------------------------------------------------

clean_column_names = [
    str(column).strip()
    for column in df.columns
]

if not any(clean_column_names):

    st.error(
        "The uploaded dataset does not contain valid "
        "column names."
    )

    st.stop()


# ---------------------------------------------------------
# Validate required structural fields
# ---------------------------------------------------------

normalized_columns = {
    str(column)
    .strip()
    .lower()
    .replace(" ", "_")
    for column in df.columns
}

required_columns = {
    "country",
    "year",
    "disease_name",
}

missing_required_columns = (
    required_columns - normalized_columns
)

if missing_required_columns:

    missing_columns_display = ", ".join(
        sorted(missing_required_columns)
    )

    st.error(
        "The uploaded dataset is missing required "
        f"structural fields: {missing_columns_display}."
    )

    st.info(
        "The dataset should contain at least the "
        "country, year, and disease name fields "
        "before cleaning."
    )

    st.stop()


# ---------------------------------------------------------
# Raw dataset preview
# ---------------------------------------------------------

st.divider()

st.subheader("Uploaded Dataset Preview")

st.caption(
    "This is the dataset before the cleaning pipeline runs."
)

raw_col1, raw_col2, raw_col3, raw_col4 = st.columns(4)

with raw_col1:

    st.metric(
        "Rows",
        f"{len(df):,}",
    )

with raw_col2:

    st.metric(
        "Columns",
        len(df.columns),
    )

with raw_col3:

    st.metric(
        "Missing Values",
        f"{int(df.isna().sum().sum()):,}",
    )

with raw_col4:

    st.metric(
        "Duplicate Rows",
        f"{int(df.duplicated().sum()):,}",
    )


preview_rows = st.slider(
    "Rows to preview",
    min_value=5,
    max_value=min(50, len(df)),
    value=min(10, len(df)),
    key="raw_preview_rows",
)

st.dataframe(
    df.head(preview_rows),
    use_container_width=True,
)


# ---------------------------------------------------------
# Run cleaning pipeline
# ---------------------------------------------------------

st.divider()

st.subheader("Data Processing")

st.markdown(
    """
    The uploaded dataset will now pass through the project's
    automated cleaning and validation pipeline.
    """
)


process_button = st.button(
    "🧹 Clean and Validate Dataset",
    type="primary",
    use_container_width=True,
)


if process_button:

    temporary_input_path = None

    try:

        # ---------------------------------------------
        # Save uploaded DataFrame temporarily
        # ---------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".csv",
            delete=False,
        ) as temp_file:

            temporary_input_path = Path(
                temp_file.name
            )

        df.to_csv(
            temporary_input_path,
            index=False,
        )

        # ---------------------------------------------
        # Run existing project pipeline
        # ---------------------------------------------

        with st.spinner(
            "Running the data-cleaning and validation pipeline..."
        ):

            clean_df, reports, summary_report = (
                run_pipeline(
                    temporary_input_path
                )
            )

        # ---------------------------------------------
        # Store cleaned dataset
        # ---------------------------------------------

        st.session_state[
            "uploaded_data"
        ] = clean_df.copy()

        st.session_state[
            "uploaded_filename"
        ] = file_name

        st.session_state[
            "uploaded_sheet"
        ] = selected_sheet

        st.session_state[
            "uploaded_table"
        ] = selected_table

        st.session_state[
            "uploaded_reports"
        ] = reports

        st.session_state[
            "uploaded_validation_report"
        ] = summary_report

        st.session_state[
            "uploaded_data_processed"
        ] = True

        # ---------------------------------------------
        # Success message
        # ---------------------------------------------

        st.success(
            "✅ Dataset cleaned and validated successfully."
        )

        st.rerun()

    except Exception as exc:

        st.error(
            "The uploaded dataset could not be processed "
            "by the cleaning pipeline."
        )

        st.info(
            f"Processing error: {exc}"
        )

    finally:

        # ---------------------------------------------
        # Remove temporary input file
        # ---------------------------------------------

        if temporary_input_path is not None:

            temporary_input_path.unlink(
                missing_ok=True
            )


# ---------------------------------------------------------
# Display processing results
# ---------------------------------------------------------

if st.session_state.get(
    "uploaded_data_processed",
    False,
):

    clean_df = st.session_state[
        "uploaded_data"
    ]

    reports = st.session_state.get(
        "uploaded_reports",
        {},
    )

    summary_report = st.session_state.get(
        "uploaded_validation_report",
        pd.DataFrame(),
    )

    # -----------------------------------------------------
    # Cleaned dataset statistics
    # -----------------------------------------------------

    st.divider()

    st.subheader("Cleaned Dataset")

    st.caption(
        "This cleaned and validated dataset is now used "
        "throughout the analysis pages."
    )

    clean_col1, clean_col2, clean_col3, clean_col4 = (
        st.columns(4)
    )

    with clean_col1:

        st.metric(
            "Rows",
            f"{len(clean_df):,}",
        )

    with clean_col2:

        st.metric(
            "Columns",
            len(clean_df.columns),
        )

    with clean_col3:

        st.metric(
            "Missing Values",
            f"{int(clean_df.isna().sum().sum()):,}",
        )

    with clean_col4:

        st.metric(
            "Duplicate Rows",
            f"{int(clean_df.duplicated().sum()):,}",
        )

    # -----------------------------------------------------
    # Cleaned preview
    # -----------------------------------------------------

    st.subheader("Cleaned Data Preview")

    cleaned_preview_rows = st.slider(
        "Rows to preview",
        min_value=5,
        max_value=min(50, len(clean_df)),
        value=min(10, len(clean_df)),
        key="cleaned_preview_rows",
    )

    st.dataframe(
        clean_df.head(cleaned_preview_rows),
        use_container_width=True,
    )

    # -----------------------------------------------------
    # Pipeline summary
    # -----------------------------------------------------

    st.divider()

    st.subheader("Pipeline Summary")

    pipeline_report = reports.get(
        "pipeline",
        {},
    )

    pipeline_col1, pipeline_col2 = st.columns(2)

    with pipeline_col1:

        st.metric(
            "Final Rows",
            f"{pipeline_report.get(
                'rows_after_pipeline',
                len(clean_df)
            ):,}",
        )

    with pipeline_col2:

        st.metric(
            "Final Columns",
            pipeline_report.get(
                "columns_after_pipeline",
                len(clean_df.columns),
            ),
        )

    # -----------------------------------------------------
    # Validation report
    # -----------------------------------------------------

    if not summary_report.empty:

        st.divider()

        st.subheader(
            "Validation Summary"
        )

        st.dataframe(
            summary_report,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # Column listing
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "Cleaned Column Listing"
    )

    column_information = pd.DataFrame(
        {
            "Column": clean_df.columns,
            "Data Type": [
                str(dtype)
                for dtype in clean_df.dtypes
            ],
            "Non-Null Values": [
                int(
                    clean_df[column]
                    .notna()
                    .sum()
                )
                for column in clean_df.columns
            ],
            "Missing Values": [
                int(
                    clean_df[column]
                    .isna()
                    .sum()
                )
                for column in clean_df.columns
            ],
            "Unique Values": [
                int(
                    clean_df[column]
                    .nunique()
                )
                for column in clean_df.columns
            ],
        }
    )

    st.dataframe(
        column_information,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # Quick insights
    # -----------------------------------------------------

    st.divider()

    st.subheader("Quick Insights")

    numeric_columns = clean_df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = clean_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:

        st.markdown("**Numeric Columns**")

        if numeric_columns:

            st.write(
                f"{len(numeric_columns)} numeric "
                "columns detected."
            )

            st.write(
                ", ".join(numeric_columns)
            )

        else:

            st.write(
                "No numeric columns detected."
            )

    with insight_col2:

        st.markdown("**Categorical Columns**")

        if categorical_columns:

            st.write(
                f"{len(categorical_columns)} categorical "
                "columns detected."
            )

            st.write(
                ", ".join(categorical_columns)
            )

        else:

            st.write(
                "No categorical columns detected."
            )

    # -----------------------------------------------------
    # Numeric summary
    # -----------------------------------------------------

    if numeric_columns:

        st.divider()

        st.subheader(
            "Numeric Summary"
        )

        st.dataframe(
            clean_df[
                numeric_columns
            ].describe().T,
            use_container_width=True,
        )