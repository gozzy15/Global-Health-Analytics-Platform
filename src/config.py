"""
Configuration constants used throughout the Global Health
Dataset cleaning project.

This module stores file paths, column definitions,
correction mappings, validation settings, and other shared
configuration values.
"""


# -------------------------------------------------------------------
# File paths
# -------------------------------------------------------------------

RAW_PATH = "data/raw/Global Health Dataset.csv"

PROCESSED_DIR = "data/processed"

CLEANED_DATA_PATH_1 = (
    f"{PROCESSED_DIR}/Global Health Dataset_cleaned.csv"
)

CLEANED_DATA_PATH_2 = "global_health_web_app/data/Global Health Dataset_cleaned.csv"

VALIDATION_REPORT_PATH = (
    f"{PROCESSED_DIR}/validation_report.csv"
)

# -------------------------------------------------------------------
# Values that should be interpreted as missing
# -------------------------------------------------------------------

NA_VALUES = [
    "",
    " ",
    "NA",
    "N/A",
    "NULL",
    "null",
    "NaN",
    "nan",
]

# -------------------------------------------------------------------
# Column renaming map
# -------------------------------------------------------------------
COLUMN_RENAME_MAP = {
    "Row_num": "row_num",
    "Country": "country",
    "Year": "year",
    "Disease Name": "disease_name",
    "Country_pop": "country_pop",
    "Incidence Rate mn (%)": "incidence_rate_pct",
    "Prevalence rate (%)": "prevalence_rate_pct",
    "Mortality Rate per 100 people (%)": "mortality_rate_per_100_people_pct",
    "Population affected": "population_affected",
    "Pop_affected(Male)": "pop_affected_male",
    "Pop_affected(Female)": "pop_affected_female",
    "Ages 0-18 (%)": "ages_0_18_pct",
    "Ages 19-35 (%)": "ages_19_35_pct",
    "Ages 36-60 (%)": "ages_36_60_pct",
    "Ages 61+ (%)": "ages_61_plus_pct",
    "Pop_affected_U (%)": "pop_affected_urban_pct",
    "Pop_affected_R (%)": "pop_affected_rural_pct",
    "Healthcare Access (%)": "healthcare_access_pct",
    "Doctors per 1000": "doctors_per_1000",
    "Hospital Beds per 1000": "hospital_beds_per_1000",
    "Treatment type": "treatment_type",
    "Recovery Rate (%)": "recovery_rate_pct",
    "DALYs": "dalys",
    "Improvement in 5 Years (%)": "improvement_in_5_years_pct",
    "Average Annual Treatment Cost (USD)": "average_annual_treatment_cost_usd",
    "Availability of Vaccines/Treatment": "availability_of_vaccines_treatment",
    "Composite Health Index (CHI)": "composite_health_index",
    "Per Capita Income (USD)": "per_capita_income_usd",
    "Education Index": "education_index",
    "Urbanization Rate (%)": "urbanization_rate_pct",
}

# -------------------------------------------------------------------
# Expected dataset structure
# -------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "Row_num",
    "Country",
    "Year",
    "Disease Name",
    "Country_pop",
    "Incidence Rate mn (%)",
    "Prevalence rate (%)",
    "Mortality Rate per 100 people (%)",
    "Population affected",
    "Pop_affected(Male)",
    "Pop_affected(Female)",
    "Ages 0-18 (%)",
    "Ages 19-35 (%)",
    "Ages 36-60 (%)",
    "Ages 61+ (%)",
    "Pop_affected_U (%)",
    "Pop_affected_R (%)",
    "Healthcare Access (%)",
    "Doctors per 1000",
    "Hospital Beds per 1000",
    "Treatment type",
    "Recovery Rate (%)",
    "DALYs",
    "Improvement in 5 Years (%)",
    "Average Annual Treatment Cost (USD)",
    "Availability of Vaccines/Treatment",
    "Composite Health Index (CHI)",
    "Per Capita Income (USD)",
    "Education Index",
    "Urbanization Rate (%)",
]


EXPECTED_DTYPES = {
    "Row_num": "numeric",
    "Country": "object",
    "Year": "numeric",
    "Disease Name": "object",
    "Country_pop": "numeric",
    "Incidence Rate mn (%)": "numeric",
    "Prevalence rate (%)": "numeric",
    "Mortality Rate per 100 people (%)": "numeric",
    "Population affected": "numeric",
    "Pop_affected(Male)": "numeric",
    "Pop_affected(Female)": "numeric",
    "Ages 0-18 (%)": "numeric",
    "Ages 19-35 (%)": "numeric",
    "Ages 36-60 (%)": "numeric",
    "Ages 61+ (%)": "numeric",
    "Pop_affected_U (%)": "numeric",
    "Pop_affected_R (%)": "numeric",
    "Healthcare Access (%)": "numeric",
    "Doctors per 1000": "numeric",
    "Hospital Beds per 1000": "numeric",
    "Treatment type": "object",
    "Recovery Rate (%)": "numeric",
    "DALYs": "numeric",
    "Improvement in 5 Years (%)": "numeric",
    "Average Annual Treatment Cost (USD)": "numeric",
    "Availability of Vaccines/Treatment": "object",
    "Composite Health Index (CHI)": "numeric",
    "Per Capita Income (USD)": "numeric",
    "Education Index": "numeric",
    "Urbanization Rate (%)": "numeric",
}


# -------------------------------------------------------------------
# Expected categorical values
# -------------------------------------------------------------------

EXPECTED_COUNTRIES = [
    "Italy",
    "France",
    "Turkey",
    "Indonesia",
    "Saudi Arabia",
    "USA",
    "Nigeria",
    "Australia",
    "Canada",
    "Mexico",
    "China",
    "South Africa",
    "Japan",
    "United Kingdom",
    "Russia",
    "Brazil",
    "Germany",
    "India",
    "Argentina",
    "South Korea",
]


EXPECTED_DISEASES = [
    "Malaria",
    "Ebola",
    "COVID-19",
    "Parkinson's Disease",
    "Tuberculosis",
    "Dengue",
    "Rabies",
    "Cholera",
    "Leprosy",
    "Cancer",
    "Diabetes",
    "Measles",
    "Zika",
    "Alzheimer's Disease",
    "Polio",
    "Hypertension",
    "Asthma",
    "HIV/AIDS",
    "Influenza",
    "Hepatitis",
]


# -------------------------------------------------------------------
# Numeric columns
# -------------------------------------------------------------------

NUMERIC_COLUMNS = [
    "row_num",
    "year",
    "country_pop",

    "incidence_rate_pct",
    "prevalence_rate_pct",
    "mortality_rate_pct",

    "population_affected",
    "pop_affected_male",
    "pop_affected_female",

    "ages_0_18_pct",
    "ages_19_35_pct",
    "ages_36_60_pct",
    "ages_61_plus_pct",

    "pop_affected_urban_pct",
    "pop_affected_rural_pct",

    "healthcare_access_pct",
    "doctors_per_1000",
    "hospital_beds_per_1000",

    "recovery_rate_pct",

    "dalys",

    "improvement_in_5_years_pct",

    "average_annual_treatment_cost_usd",

    "composite_health_index",

    "per_capita_income_usd",

    "education_index",

    "urbanization_rate_pct",
]


# -------------------------------------------------------------------
# Column categories
# -------------------------------------------------------------------

PERCENTAGE_COLUMNS = [
    "incidence_rate_pct",
    "prevalence_rate_pct",
    "mortality_rate_pct",
    "ages_0_18_pct",
    "ages_19_35_pct",
    "ages_36_60_pct",
    "ages_61_plus_pct",
    "pop_affected_urban_pct",
    "pop_affected_rural_pct",
    "healthcare_access_pct",
    "recovery_rate_pct",
    "improvement_in_5_years_pct",
    "urbanization_rate_pct",
]


CURRENCY_COLUMNS = [
    "average_annual_treatment_cost_usd",
    "per_capita_income_usd",
]


# -------------------------------------------------------------------
# Text correction mappings
# -------------------------------------------------------------------

COUNTRY_CORRECTIONS = {
    "It@l¥": "Italy",
    "It@lĄ": "Italy",
    "It@lÂ¥": "Italy",
    "T?u?r?k?e?y?": "Turkey",
    "Can@da": "Canada",
    "Mex!co": "Mexico",
    "G%rmany": "Germany",
    "?r?zil": "Brazil",
    "Ind!a": "India",
}


DISEASE_CORRECTIONS = {
    "Tub?rculosis": "Tuberculosis",
    "Influen&za": "Influenza",
    "Pol!o": "Polio",
    "HIV/A!DS": "HIV/AIDS",
}


AVAILABILITY_MAPPING = {
    "high": "High",
    "medium": "Medium",
    "m?dium": "Medium",
    "low": "Low",
    "none": "None",
    "~none~": "None",
}


TREATMENT_MAP = {
    "medication": "Medication",
    "therapy": "Therapy",
    "surgery": "Surgery",
    "vaccination": "Vaccination",
}


# -------------------------------------------------------------------
# Valid numeric ranges
# -------------------------------------------------------------------

VALID_RANGES = {
    "year": (2000, 2024),

    "healthcare_access_pct": (0, 100),

    "recovery_rate_pct": (0, 100),

    "education_index": (0, 1),

    "composite_health_index": (0, 100),

    "doctors_per_1000": (0, 10),

    "hospital_beds_per_1000": (0, 20),

    "urbanization_rate_pct": (0, 100),
}


# -------------------------------------------------------------------
# Identifier columns
# -------------------------------------------------------------------

IDENTIFIER_COLUMNS = [
    "country",
    "year",
    "disease_name",
]


# -------------------------------------------------------------------
# Structural missing columns
# -------------------------------------------------------------------

STRUCTURAL_MISSING_COLUMNS = [
    "incidence_rate_pct",
    "prevalence_rate_pct",
    "population_affected",
    "pop_affected_male",
    "pop_affected_female",
    "ages_0_18_pct",
    "ages_19_35_pct",
    "ages_36_60_pct",
    "ages_61_plus_pct",
    "pop_affected_urban_pct",
    "pop_affected_rural_pct",
]


# -------------------------------------------------------------------
# Peer-country definitions
# -------------------------------------------------------------------

PEER_COUNTRIES = {
    "United Kingdom": [
        "Canada",
        "France",
        "Germany",
        "Australia",
        "Japan",
    ]
}


# -------------------------------------------------------------------
# Columns handled by peer-country median imputation
# -------------------------------------------------------------------

PEER_IMPUTATION_COLUMNS = [
    "healthcare_access_pct",
    "doctors_per_1000",
    "hospital_beds_per_1000",
]


# -------------------------------------------------------------------
# Country-Year consistency validation columns
# -------------------------------------------------------------------

COUNTRY_YEAR_COLUMNS = [
    "country_pop",
    "per_capita_income_usd",
    "education_index",
    "urbanization_rate_pct",
    "healthcare_access_pct",
    "doctors_per_1000",
    "hospital_beds_per_1000",
    "composite_health_index",
]