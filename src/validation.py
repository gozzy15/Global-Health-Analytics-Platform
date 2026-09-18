"""
Utilities for repairing data quality issues and validating
business rules in the Global Health Dataset.

This module performs duplicate removal, consistency checks,
constraint validation, range validation, and data repairs
to ensure the cleaned dataset satisfies predefined quality
requirements.
"""

import pandas as pd

from .config import (
    COUNTRY_YEAR_COLUMNS,
    VALID_RANGES,
)


def check_duplicates(
    df: pd.DataFrame,
) -> dict:
    """
    Check for duplicate records in the dataset.

    Two types of duplicates are evaluated:

    1. Exact duplicate rows.
    2. Duplicate Country-Year-Disease combinations.

    The Country-Year-Disease combination is expected to
    uniquely identify each record in the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to inspect for duplicate records.

    Returns
    -------
    dict
        Dictionary containing:

        - ``Total Rows``: Total number of rows inspected.
        - ``Duplicate Rows``: Number of exact duplicate rows.
        - ``Duplicate Country-Year-Disease``: Number of
          duplicate Country-Year-Disease combinations.
    """

    df = df.copy()

    duplicate_rows = df.duplicated().sum()

    duplicate_keys = (
        df.duplicated(
            subset=[
                "country",
                "year",
                "disease_name",
            ]
        ).sum()
    )

    report = {
        "Total Rows": len(df),
        "Duplicate Rows": int(duplicate_rows),
        "Duplicate Country-Year-Disease": int(duplicate_keys),
    }

    return report


def remove_duplicate_records(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Remove duplicate records from the dataset.

    Duplicate detection ignores the ``row_num`` column so
    that otherwise identical records with different source
    row numbers are treated as duplicates.

    The first occurrence of each duplicate record is retained.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to clean.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        DataFrame with duplicate records removed and the
        index reset.

    report : dict
        Summary containing:

        - ``Rows Before``: Number of rows before removal.
        - ``Duplicates Removed``: Number of duplicate rows removed.
        - ``Rows After``: Number of rows remaining after removal.
    """

    df = df.copy()

    duplicate_mask = df.duplicated(
        subset=df.columns.drop("row_num"),
        keep="first",
    )

    duplicates_removed = duplicate_mask.sum()

    rows_before = len(df)

    df = df.loc[~duplicate_mask].reset_index(drop=True)

    report = {
        "Rows Before": int(rows_before),
        "Duplicates Removed": int(duplicates_removed),
        "Rows After": len(df),
    }

    return df, report


def validate_numeric_range(
    df: pd.DataFrame,
    column: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict:
    """
    Validate that values in a numeric column fall within
    an expected range.

    Values below ``minimum`` and values above ``maximum``
    are counted as violations. Either boundary may be
    omitted by passing ``None``.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the column to validate.

    column : str
        Name of the numeric column to validate.

    minimum : float or None, optional
        Minimum allowed value. Values below this threshold
        are treated as violations.

    maximum : float or None, optional
        Maximum allowed value. Values above this threshold
        are treated as violations.

    Returns
    -------
    dict
        Validation report containing the configured limits,
        violation counts, total violations, and a boolean
        ``Passed`` status.
    """

    series = df[column]

    below_min = 0
    above_max = 0

    if minimum is not None:
        below_min = (series < minimum).sum()

    if maximum is not None:
        above_max = (series > maximum).sum()

    total_violations = below_min + above_max

    report = {
        "Column": column,
        "Minimum": minimum,
        "Maximum": maximum,
        "Below Minimum": int(below_min),
        "Above Maximum": int(above_max),
        "Total Violations": int(total_violations),
        "Passed": bool(total_violations == 0),
    }

    return report


def show_range_violations(
    df: pd.DataFrame,
    column: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> pd.DataFrame:
    """
    Return rows containing values outside an expected range.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the column to inspect.

    column : str
        Name of the numeric column to inspect.

    minimum : float or None, optional
        Minimum allowed value.

    maximum : float or None, optional
        Maximum allowed value.

    Returns
    -------
    pandas.DataFrame
        Rows where the specified column falls below the
        minimum or above the maximum.
    """

    below = (
        df[column] < minimum
        if minimum is not None
        else False
    )

    above = (
        df[column] > maximum
        if maximum is not None
        else False
    )

    violations = df[below | above]

    return violations


def repair_age_distribution(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Repair rows containing one invalid or missing age-group percentage.

    The four age-group percentages are expected to represent the
    complete population distribution and therefore sum to 100%.

    Two repair cases are handled:

    1. If exactly one age-group value is missing and the other three
       values are valid percentages, the missing value is calculated as:

           100 - sum(other three age groups)

       The repair is only performed when the three known values sum
       to less than 100%.

    2. If exactly one age-group value is outside the valid percentage
       range of 0 to 100 and the other three values are valid, the
       invalid value is recalculated as:

           100 - sum(other three age groups)

       The calculated replacement must itself fall within the valid
       range of 0 to 100.

    Rows containing multiple missing values, multiple invalid values,
    or insufficient information for a reliable calculation are left
    unchanged.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the four age-group percentage columns.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        DataFrame with eligible invalid or missing age-group values
        repaired.

    report : dict
        Dictionary containing the number of rows repaired.
    """

    df = df.copy()

    age_columns = [
        "ages_0_18_pct",
        "ages_19_35_pct",
        "ages_36_60_pct",
        "ages_61_plus_pct",
    ]

    repaired = 0

    for index, row in df.iterrows():

        # ---------------------------------------------------------
        # Identify missing age-group values.
        # ---------------------------------------------------------
        missing_columns = [
            col
            for col in age_columns
            if pd.isna(row[col])
        ]

        # ---------------------------------------------------------
        # CASE 1: Exactly one value is missing.
        # ---------------------------------------------------------
        if len(missing_columns) == 1:

            missing_column = missing_columns[0]

            other_columns = [
                col
                for col in age_columns
                if col != missing_column
            ]

            # All three known values must be valid percentages.
            known_values_valid = all(
                pd.notna(row[col])
                and 0 <= row[col] <= 100
                for col in other_columns
            )

            if not known_values_valid:
                continue

            other_total = row[other_columns].sum()

            # The known values must leave room for the missing value.
            if other_total >= 100:
                continue

            corrected_value = 100 - other_total

            # Final safety check.
            if 0 <= corrected_value <= 100:
                df.at[index, missing_column] = corrected_value
                repaired += 1

            continue

        # ---------------------------------------------------------
        # CASE 2: More than one value is missing.
        # ---------------------------------------------------------
        if len(missing_columns) > 1:
            continue

        # ---------------------------------------------------------
        # CASE 3: No missing values - check for invalid values.
        # ---------------------------------------------------------
        invalid_columns = [
            col
            for col in age_columns
            if row[col] < 0 or row[col] > 100
        ]

        # Only repair rows containing exactly one invalid value.
        if len(invalid_columns) != 1:
            continue

        invalid_column = invalid_columns[0]

        other_columns = [
            col
            for col in age_columns
            if col != invalid_column
        ]

        # All other age-group values must be valid before
        # using them to calculate the replacement.
        other_values_valid = all(
            pd.notna(row[col])
            and 0 <= row[col] <= 100
            for col in other_columns
        )

        if not other_values_valid:
            continue

        other_total = row[other_columns].sum()

        corrected_value = 100 - other_total

        # Only use the calculated value if it is itself valid.
        if 0 <= corrected_value <= 100:
            df.at[index, invalid_column] = corrected_value
            repaired += 1

    return df, {
        "Rows Repaired": repaired,
    }


def repair_healthcare_access(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Repair obvious decimal-place errors in healthcare access.

    Values greater than 100 are repeatedly divided by 10
    until they fall within the valid percentage range.

    This is intended to correct apparent decimal-place errors
    rather than legitimate values.

    Examples
    --------
    ``850`` becomes ``85``.

    ``1200`` becomes ``12``.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the ``healthcare_access_pct``
        column.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        DataFrame with values above 100 corrected.

    report : dict
        Dictionary containing:

        - ``Column``: Name of the repaired column.
        - ``Rows Above 100 Before``: Number of values above
          100 before repair.
        - ``Rows Above 100 After``: Number remaining above
          100 after repair.
        - ``Rows Repaired``: Number of values processed.
    """

    df = df.copy()

    column = "healthcare_access_pct"

    mask = df[column] > 100

    repaired = int(mask.sum())

    def fix_value(x):
        while pd.notna(x) and x > 100:
            x /= 10
        return x

    df.loc[mask, column] = (
        df.loc[mask, column]
        .apply(fix_value)
    )

    return df, {
        "Column": column,
        "Rows Above 100 Before": repaired,
        "Rows Above 100 After": int(
            (df[column] > 100).sum()
        ),
        "Rows Repaired": repaired,
    }


def validate_gender_population(
    df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """
    Validate gender population consistency.

    The validation checks whether:

    ``population_affected = pop_affected_male + pop_affected_female``

    Rows with missing values in any of the three required
    columns are excluded from the validation.

    A tolerance of ±1 person is allowed to account for
    rounding differences.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing population affected and male/female
        population columns.

    Returns
    -------
    report : dict
        Validation summary containing rows checked, rows skipped,
        number of violations, and pass/fail status.

    violations : pandas.DataFrame
        Rows where the male and female populations differ from
        the total population affected by more than one person.
    """

    required_cols = [
        "population_affected",
        "pop_affected_male",
        "pop_affected_female",
    ]

    check_df = df.dropna(
        subset=required_cols
    ).copy()

    comparison = (
        check_df["pop_affected_male"]
        + check_df["pop_affected_female"]
    )

    difference = (
        comparison
        - check_df["population_affected"]
    ).abs()

    violations = check_df[
        difference > 1
    ]

    report = {
        "Rows Checked": int(len(check_df)),
        "Rows Skipped (Missing)": int(
            len(df) - len(check_df)
        ),
        "Violations": int(len(violations)),
        "Passed": len(violations) == 0,
    }

    return report, violations


def validate_urban_rural_distribution(
    df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """
    Validate the Urban and Rural population distribution.

    The validation checks whether:

    ``pop_affected_urban_pct + pop_affected_rural_pct = 100%``

    Rows containing missing Urban or Rural values are excluded
    from the validation.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing Urban and Rural population
        percentage columns.

    Returns
    -------
    report : dict
        Validation summary containing rows checked, rows skipped,
        number of violations, and pass/fail status.

    violations : pandas.DataFrame
        Rows where the Urban and Rural percentages do not
        sum to 100%.
    """

    columns = [
        "pop_affected_urban_pct",
        "pop_affected_rural_pct",
    ]

    complete = df[columns].notna().all(axis=1)

    totals = df.loc[
        complete,
        columns,
    ].sum(axis=1)

    violations = df.loc[
        complete & (totals != 100),
        [
            "country",
            "year",
            "disease_name",
            "pop_affected_urban_pct",
            "pop_affected_rural_pct",
        ],
    ]

    report = {
        "Rows Checked": int(complete.sum()),
        "Rows Skipped (Missing)": int(
            (~complete).sum()
        ),
        "Violations": int(len(violations)),
        "Passed": len(violations) == 0,
    }

    return report, violations


def repair_urban_rural_distribution(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Repair small rounding errors in Urban and Rural
    population percentages.

    Only totals of 99% or 101% are corrected. The larger
    percentage is adjusted by one percentage point so that
    the two values sum to 100%.

    Larger discrepancies are preserved for manual review.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing Urban and Rural population
        percentage columns.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        DataFrame with eligible rounding errors repaired.

    report : dict
        Dictionary containing the number of rows repaired.
    """

    df = df.copy()

    urban = "pop_affected_urban_pct"
    rural = "pop_affected_rural_pct"

    repaired = 0

    complete = (
        df[[urban, rural]]
        .notna()
        .all(axis=1)
    )

    totals = df.loc[
        complete,
        [urban, rural],
    ].sum(axis=1)

    for idx in totals[
        totals == 101
    ].index:

        if df.at[idx, urban] >= df.at[idx, rural]:
            df.at[idx, urban] -= 1
        else:
            df.at[idx, rural] -= 1

        repaired += 1

    for idx in totals[
        totals == 99
    ].index:

        if df.at[idx, urban] >= df.at[idx, rural]:
            df.at[idx, urban] += 1
        else:
            df.at[idx, rural] += 1

        repaired += 1

    report = {
        "Rows Repaired": repaired,
    }

    return df, report


def validate_population_affected(
    df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """
    Validate the population affected constraint.

    The validation checks that population affected does not
    exceed the corresponding country population.

    Rows containing missing values in either required column
    are excluded from the validation.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing population affected and country
        population values.

    Returns
    -------
    report : dict
        Validation summary containing rows checked, rows skipped,
        number of violations, and pass/fail status.

    violations : pandas.DataFrame
        Rows where population affected exceeds country population.
    """

    complete = (
        df[
            [
                "population_affected",
                "country_pop",
            ]
        ]
        .notna()
        .all(axis=1)
    )

    violations = df.loc[
        complete
        & (
            df["population_affected"]
            > df["country_pop"]
        ),
        [
            "country",
            "year",
            "disease_name",
            "population_affected",
            "country_pop",
        ],
    ]

    report = {
        "Rows Checked": int(complete.sum()),
        "Rows Skipped (Missing)": int(
            (~complete).sum()
        ),
        "Violations": int(len(violations)),
        "Passed": len(violations) == 0,
    }

    return report, violations


def validate_chi(
    df: pd.DataFrame,
) -> dict:
    """
    Validate Composite Health Index values.

    Composite Health Index values are expected to fall
    between 0 and 100 inclusive.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the composite health index column.

    Returns
    -------
    dict
        Numeric range validation report for the
        ``composite_health_index`` column.
    """

    return validate_numeric_range(
        df=df,
        column="composite_health_index",
        minimum=0,
        maximum=100,
    )


def validate_country_year_consistency(
    df: pd.DataFrame,
    column: str,
) -> tuple[dict, pd.DataFrame]:
    """
    Validate country-level consistency within Country-Year groups.

    A country-level indicator is expected to have one consistent
    value for every disease recorded within the same Country-Year
    group.

    This validation is appropriate for indicators such as
    country population, per capita income, Education Index,
    and Urbanization Rate.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing country, year, and the indicator
        column to validate.

    column : str
        Country-level indicator whose consistency should
        be checked.

    Returns
    -------
    report : dict
        Validation summary containing:

        - ``Column``
        - ``Country-Year Groups Checked``
        - ``Inconsistent Groups``
        - ``Passed``

    violations : pandas.DataFrame
        Rows belonging to Country-Year groups containing
        more than one unique value for the specified column.

    Notes
    -----
    Some indicators, such as healthcare access, doctors per
    1000, hospital beds per 1000, and composite health index,
    may legitimately vary by disease. Such variation is
    reported as informational rather than automatically treated
    as a data error by the reporting layer.
    """

    grouped = (
        df.groupby(
            ["country", "year"]
        )[column]
        .nunique(dropna=True)
        .reset_index(
            name="Unique Values"
        )
    )

    inconsistent_groups = grouped[
        grouped["Unique Values"] > 1
    ]

    if inconsistent_groups.empty:

        report = {
            "Column": column,
            "Country-Year Groups Checked": len(grouped),
            "Inconsistent Groups": 0,
            "Passed": True,
        }

        violations = df.iloc[0:0].copy()

    else:

        violations = df.merge(
            inconsistent_groups[
                [
                    "country",
                    "year",
                ]
            ],
            on=[
                "country",
                "year",
            ],
            how="inner",
        ).sort_values(
            [
                "country",
                "year",
                "disease_name",
            ]
        )

        report = {
            "Column": column,
            "Country-Year Groups Checked": len(grouped),
            "Inconsistent Groups": len(
                inconsistent_groups
            ),
            "Passed": False,
        }

    return report, violations


def clean_and_validate_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Execute all data repairs and validation checks.

    The workflow removes duplicate records, repairs known
    percentage errors, and then performs the configured
    validation checks.

    Repair stages include:

    1. Duplicate record removal.
    2. Age distribution repair.
    3. Healthcare Access repair.
    4. Urban/Rural percentage repair.

    Validation stages include:

    1. Duplicate record validation.
    2. Composite Health Index range validation.
    3. Gender population consistency.
    4. Urban/Rural distribution consistency.
    5. Population affected constraint.
    6. Country-Year consistency checks for configured
       country-level indicators.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned DataFrame entering the repair and validation
        stage of the pipeline.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        DataFrame after all configured repairs have been
        applied.

    reports : dict
        Nested dictionary containing separate ``repairs`` and
        ``validations`` sections. Each validation contains its
        report and, where applicable, a DataFrame of violations.
    """

    reports = {
        "repairs": {},
        "validations": {},
    }

    # --------------------------------------------------
    # Remove duplicate records
    # --------------------------------------------------
    df, reports["repairs"]["duplicate_records"] = (
        remove_duplicate_records(df)
    )

    # --------------------------------------------------
    # Repair age distribution
    # --------------------------------------------------
    df, reports["repairs"]["age_distribution"] = (
        repair_age_distribution(df)
    )

    # --------------------------------------------------
    # Repair Healthcare Access
    # --------------------------------------------------
    df, reports["repairs"]["healthcare_access"] = (
        repair_healthcare_access(df)
    )

    # --------------------------------------------------
    # Repair Urban/Rural percentages
    # --------------------------------------------------
    df, reports["repairs"]["urban_rural"] = (
        repair_urban_rural_distribution(df)
    )

    # --------------------------------------------------
    # Duplicate validation
    # --------------------------------------------------
    reports["validations"]["duplicate"] = {
        "report": check_duplicates(df),
        "violations": None,
    }

    # --------------------------------------------------
    # Composite Health Index range
    # --------------------------------------------------
    reports["validations"]["chi"] = {
        "report": validate_chi(df),
        "violations": None,
    }

    # --------------------------------------------------
    # Gender population consistency
    # --------------------------------------------------
    report, violations = (
        validate_gender_population(df)
    )

    reports["validations"]["gender_population"] = {
        "report": report,
        "violations": violations,
    }

    # --------------------------------------------------
    # Urban/Rural distribution
    # --------------------------------------------------
    report, violations = (
        validate_urban_rural_distribution(df)
    )

    reports["validations"]["urban_rural"] = {
        "report": report,
        "violations": violations,
    }

    # --------------------------------------------------
    # Population affected constraint
    # --------------------------------------------------
    report, violations = (
        validate_population_affected(df)
    )

    reports["validations"]["population_constraint"] = {
        "report": report,
        "violations": violations,
    }

    # --------------------------------------------------
    # Country-Year consistency checks
    # --------------------------------------------------
    reports["validations"][
        "country_year_consistency"
    ] = {}

    for column in COUNTRY_YEAR_COLUMNS:

        report, violations = (
            validate_country_year_consistency(
                df,
                column,
            )
        )

        reports["validations"][
            "country_year_consistency"
        ][column] = {
            "report": report,
            "violations": violations,
        }

    return df, reports