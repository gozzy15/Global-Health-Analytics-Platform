"""
Main entry point for the Global Health Dataset
data-cleaning pipeline.

This module executes the complete workflow from raw-data
extraction through cleaning, validation, reporting, and
export of the final outputs.

All data-processing logic is delegated to the project's
specialized modules.
"""

from .config import (
    RAW_PATH,
    CLEANED_DATA_PATH_1,
    CLEANED_DATA_PATH_2,
    VALIDATION_REPORT_PATH,
)
from .pipeline import run_pipeline
from .export import (
    save_cleaned_data,
    save_validation_report,
)


def main() -> None:
    """
    Execute the complete Global Health Dataset workflow.

    The workflow:

    1. Loads and cleans the raw dataset.
    2. Performs data repairs and validation checks.
    3. Builds the validation summary report.
    4. Exports the cleaned dataset.
    5. Exports the validation summary report.

    Returns
    -------
    None
        The function performs the workflow and exports the
        resulting files without returning a value.
    """

    print("=" * 70)
    print("GLOBAL HEALTH DATA CLEANING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------
    # Run complete cleaning and validation pipeline
    # --------------------------------------------------
    print("\n[1] Running data-cleaning pipeline...")

    clean_df, reports, summary_report = run_pipeline(
        RAW_PATH
    )

    print(
        f"✓ Pipeline completed successfully: "
        f"{len(clean_df):,} rows × "
        f"{len(clean_df.columns)} columns."
    )

    # --------------------------------------------------
    # Export cleaned dataset
    # --------------------------------------------------
    print("\n[2] Exporting cleaned dataset...")

    cleaned_export_report_1 = save_cleaned_data(
        clean_df,
        CLEANED_DATA_PATH_1,
    )

    cleaned_export_report_2 = save_cleaned_data(
        clean_df,
        CLEANED_DATA_PATH_2,
    )

    print(
        "✓ Cleaned dataset exported successfully to both paths."
    )

    print(
        f"  Output 1: "
        f"{cleaned_export_report_1['Output Path']}"
    )

    print(
        f"  Output 2: "
        f"{cleaned_export_report_2['Output Path']}"
    )

    # --------------------------------------------------
    # Export validation summary
    # --------------------------------------------------
    print("\n[3] Exporting validation report...")

    validation_export_report = (
        save_validation_report(
            summary_report,
            VALIDATION_REPORT_PATH,
        )
    )

    print(
        "✓ Validation report exported successfully."
    )
    print(
        f"  Output: "
        f"{validation_export_report['Output Path']}"
    )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("✓ GLOBAL HEALTH DATA PIPELINE COMPLETED")
    print("=" * 70)

    print(
        f"\nFinal dataset: "
        f"{len(clean_df):,} rows × "
        f"{len(clean_df.columns)} columns"
    )

    print(
        f"Validation summary: "
        f"{len(summary_report)} rows"
    )

    print(
        "\nOutputs:"
        f"\n  • {cleaned_export_report_1['Output Path']}"
        f"\n  • {cleaned_export_report_2['Output Path']}"
        f"\n  • {validation_export_report['Output Path']}"
    )


if __name__ == "__main__":
    main()