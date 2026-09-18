"""
Machine learning utilities for the Global Health Dashboard.

This module handles:
- ML dataset preparation
- Multi-target regression
- Time-aware train/test splitting
- Model training
- Model evaluation
- Historical predictions
- Manual future scenario projections
- Autonomous future feature forecasting
- Autonomous future outcome projections
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================================================
# Prediction targets
# =========================================================

TARGETS = {
    "Incidence Rate": "incidence_rate_pct",
    "Mortality Rate": "mortality_rate_per_100_people_pct",
    "Recovery Rate": "recovery_rate_pct",
}


TARGET_DISPLAY_NAMES = {
    "incidence_rate_pct": "Incidence Rate (%)",
    "mortality_rate_per_100_people_pct": (
        "Mortality Rate per 100 People (%)"
    ),
    "recovery_rate_pct": "Recovery Rate (%)",
}


# =========================================================
# Predictor variables
# =========================================================

NUMERIC_FEATURES = [
    "year",
    "healthcare_access_pct",
    "doctors_per_1000",
    "hospital_beds_per_1000",
    "recovery_rate_pct",
    "per_capita_income_usd",
    "education_index",
    "urbanization_rate_pct",
]


CATEGORICAL_FEATURES = [
    "country",
    "disease_name",
]


# =========================================================
# Backward-compatible default target
# =========================================================

TARGET_COLUMN = "incidence_rate_pct"


# =========================================================
# Target-specific feature selection
# =========================================================

def get_feature_columns(
    target_column: str,
) -> tuple[list[str], list[str]]:
    """
    Return numeric and categorical predictors for a target.

    The selected target is removed from the predictors to
    prevent direct target leakage.
    """

    numeric_features = [
        feature
        for feature in NUMERIC_FEATURES
        if feature != target_column
    ]

    categorical_features = list(
        CATEGORICAL_FEATURES
    )

    return numeric_features, categorical_features


# =========================================================
# ML dataset preparation
# =========================================================

def prepare_ml_dataset(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Prepare a dataset for one prediction target.
    """

    numeric_features, categorical_features = (
        get_feature_columns(target_column)
    )

    required_columns = [
        target_column,
        *numeric_features,
        *categorical_features,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required ML columns are missing "
            f"from the dataset: {missing_columns}"
        )

    ml_df = df[
        required_columns
    ].copy()

    for column in numeric_features + [
        target_column
    ]:

        ml_df[column] = pd.to_numeric(
            ml_df[column],
            errors="coerce",
        )

    for column in categorical_features:

        ml_df[column] = (
            ml_df[column]
            .astype("string")
            .str.strip()
        )

    ml_df = ml_df.dropna(
        subset=[target_column]
    )

    ml_df = (
        ml_df
        .sort_values("year")
        .reset_index(drop=True)
    )

    return ml_df


# =========================================================
# Time-based split
# =========================================================

def time_based_split(
    ml_df: pd.DataFrame,
    test_years: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if "year" not in ml_df.columns:
        raise ValueError(
            "The ML dataset must contain a 'year' column."
        )

    available_years = sorted(
        ml_df["year"]
        .dropna()
        .unique()
    )

    if len(available_years) <= test_years:

        raise ValueError(
            "There are not enough years available to create "
            "the requested time-based train/test split."
        )

    test_year_values = (
        available_years[-test_years:]
    )

    test_start_year = test_year_values[0]

    train_df = ml_df[
        ml_df["year"] < test_start_year
    ].copy()

    test_df = ml_df[
        ml_df["year"] >= test_start_year
    ].copy()

    return train_df, test_df


# =========================================================
# Preprocessor
# =========================================================

def create_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:

    if numeric_features is None:
        numeric_features = list(
            NUMERIC_FEATURES
        )

    if categorical_features is None:
        categorical_features = list(
            CATEGORICAL_FEATURES
        )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor


# =========================================================
# Model definitions
# =========================================================

def build_models(
    random_state: int = 42,
) -> dict[str, Any]:

    models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state,
            ),
    }

    return models


# =========================================================
# Train models
# =========================================================

def train_models(
    train_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> dict[str, Pipeline]:

    numeric_features, categorical_features = (
        get_feature_columns(
            target_column
        )
    )

    feature_columns = (
        numeric_features
        + categorical_features
    )

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        target_column
    ]

    models = build_models()

    trained_models = {}

    for model_name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(
                        numeric_features,
                        categorical_features,
                    ),
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        trained_models[
            model_name
        ] = pipeline

    return trained_models


# =========================================================
# Train all prediction systems
# =========================================================

def train_all_target_models(
    train_data_by_target: dict[
        str,
        pd.DataFrame,
    ],
) -> dict[
    str,
    dict[str, Pipeline],
]:

    all_trained_models = {}

    for target_column, train_df in (
        train_data_by_target.items()
    ):

        all_trained_models[
            target_column
        ] = train_models(
            train_df,
            target_column=target_column,
        )

    return all_trained_models


# =========================================================
# Evaluate models
# =========================================================

def evaluate_models(
    trained_models: dict[str, Pipeline],
    test_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:

    numeric_features, categorical_features = (
        get_feature_columns(
            target_column
        )
    )

    feature_columns = (
        numeric_features
        + categorical_features
    )

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        target_column
    ]

    results = []

    for model_name, model in (
        trained_models.items()
    ):

        predictions = model.predict(
            X_test
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions,
            )
        )

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        results.append(
            {
                "Model": model_name,
                "R²": r2,
                "RMSE": rmse,
                "MAE": mae,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "RMSE",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    return results_df


# =========================================================
# Generate historical predictions
# =========================================================

def generate_predictions(
    model: Pipeline,
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:

    numeric_features, categorical_features = (
        get_feature_columns(
            target_column
        )
    )

    feature_columns = (
        numeric_features
        + categorical_features
    )

    prediction_df = data.copy()

    prediction_df[
        "predicted_value"
    ] = model.predict(
        prediction_df[
            feature_columns
        ]
    )

    if target_column in prediction_df.columns:

        prediction_df[
            "prediction_error"
        ] = (
            prediction_df[
                target_column
            ]
            - prediction_df[
                "predicted_value"
            ]
        )

    return prediction_df


# =========================================================
# Forecast future predictor variables
# =========================================================

def forecast_future_features(
    historical_data: pd.DataFrame,
    future_year: int,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Forecast future numeric predictor variables for a
    selected country-disease historical series.

    Each predictor is forecast independently using a simple
    linear time trend fitted to the available historical
    observations.

    Parameters
    ----------
    historical_data:
        Historical observations for one country-disease pair.

    future_year:
        Future year for which predictor values should be
        estimated.

    feature_columns:
        Numeric predictor columns to forecast. The `year`
        column is handled separately and should not be
        included here.

    Returns
    -------
    pd.DataFrame
        A one-row dataframe containing the forecasted
        predictor variables.
    """

    if historical_data is None or historical_data.empty:

        raise ValueError(
            "Historical data is required to forecast "
            "future predictor variables."
        )

    if "year" not in historical_data.columns:

        raise ValueError(
            "Historical data must contain a 'year' column."
        )

    if not isinstance(future_year, (int, np.integer)):

        raise TypeError(
            "future_year must be an integer."
        )

    if future_year <= int(
        historical_data["year"].max()
    ):

        raise ValueError(
            "The future year must be later than the "
            "latest historical year."
        )

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in historical_data.columns
    ]

    if missing_features:

        raise ValueError(
            "The following predictor columns are missing "
            f"from the historical data: {missing_features}"
        )

    historical_data = (
        historical_data
        .copy()
        .sort_values("year")
    )

    historical_data["year"] = pd.to_numeric(
        historical_data["year"],
        errors="coerce",
    )

    forecast_values = {}

    for feature in feature_columns:

        feature_data = historical_data[
            [
                "year",
                feature,
            ]
        ].copy()

        feature_data[feature] = pd.to_numeric(
            feature_data[feature],
            errors="coerce",
        )

        feature_data = (
            feature_data
            .dropna(
                subset=[
                    "year",
                    feature,
                ]
            )
            .sort_values("year")
        )

        if feature_data.empty:

            raise ValueError(
                f"No valid historical values are available "
                f"to forecast '{feature}'."
            )

        if len(feature_data) == 1:

            forecast_value = float(
                feature_data[feature].iloc[0]
            )

        else:

            x = feature_data[
                "year"
            ].to_numpy(
                dtype=float
            )

            y = feature_data[
                feature
            ].to_numpy(
                dtype=float
            )

            slope, intercept = np.polyfit(
                x,
                y,
                1,
            )

            forecast_value = (
                slope * float(future_year)
                + intercept
            )

        forecast_values[
            feature
        ] = float(
            forecast_value
        )

    forecast_values[
        "year"
    ] = int(
        future_year
    )

    return pd.DataFrame(
        [forecast_values]
    )


# =========================================================
# Generate autonomous future projection
# =========================================================

def generate_autonomous_projection(
    model: Pipeline,
    historical_data: pd.DataFrame,
    future_year: int,
    target_column: str = TARGET_COLUMN,
    country: str | None = None,
    disease_name: str | None = None,
) -> pd.DataFrame:
    """
    Generate an autonomous future projection.

    Historical predictor variables for one country-disease
    series are forecast to the requested future year. The
    resulting future feature vector is then passed through
    the existing trained outcome model.

    Parameters
    ----------
    model:
        Trained prediction pipeline.

    historical_data:
        Historical observations for one country-disease pair.

    future_year:
        Future year for which the projection is required.

    target_column:
        Prediction target used by the trained model.

    country:
        Optional country label for the resulting projection.

    disease_name:
        Optional disease label for the resulting projection.

    Returns
    -------
    pd.DataFrame
        A one-row dataframe containing the forecasted
        predictors and projected target.
    """

    if model is None:

        raise ValueError(
            "A trained prediction model is required."
        )

    if historical_data is None or historical_data.empty:

        raise ValueError(
            "Historical data is required to generate "
            "an autonomous projection."
        )

    numeric_features, categorical_features = (
        get_feature_columns(
            target_column
        )
    )

    feature_columns = (
        numeric_features
        + categorical_features
    )

    forecast_features = [
        feature
        for feature in numeric_features
        if feature != "year"
    ]

    if "country" not in historical_data.columns:

        raise ValueError(
            "Historical data must contain a 'country' column."
        )

    if "disease_name" not in historical_data.columns:

        raise ValueError(
            "Historical data must contain a "
            "'disease_name' column."
        )

    latest_row = (
        historical_data
        .sort_values("year")
        .iloc[-1]
    )

    if country is None:

        country = latest_row[
            "country"
        ]

    if disease_name is None:

        disease_name = latest_row[
            "disease_name"
        ]

    forecasted_features = forecast_future_features(
        historical_data=historical_data,
        future_year=future_year,
        feature_columns=forecast_features,
    )

    forecasted_features[
        "country"
    ] = country

    forecasted_features[
        "disease_name"
    ] = disease_name

    projection_data = forecasted_features[
        feature_columns
    ].copy()

    projected_result = model.predict(
        projection_data
    )

    projection_data[
        "projected_value"
    ] = projected_result

    return projection_data


# =========================================================
# Generate future scenario projection
# =========================================================

def generate_future_projection(
    model: Pipeline,
    scenario_data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Generate a future projection from a user-defined
    scenario.

    This function is intentionally kept separate from the
    autonomous forecasting system. It accepts predictor
    values supplied directly by the user and passes them
    through the existing trained prediction model.

    Parameters
    ----------
    model:
        Trained prediction pipeline.

    scenario_data:
        One or more rows containing the predictor values
        required by the selected target model.

    target_column:
        Prediction target used by the trained model.

    Returns
    -------
    pd.DataFrame
        The supplied scenario data with a
        `projected_value` column.
    """

    if model is None:

        raise ValueError(
            "A trained prediction model is required."
        )

    if scenario_data is None or scenario_data.empty:

        raise ValueError(
            "Scenario data is required to generate "
            "a future projection."
        )

    numeric_features, categorical_features = (
        get_feature_columns(
            target_column
        )
    )

    feature_columns = (
        numeric_features
        + categorical_features
    )

    missing_columns = [
        column
        for column in feature_columns
        if column not in scenario_data.columns
    ]

    if missing_columns:

        raise ValueError(
            "The following required prediction columns are "
            f"missing from the scenario data: {missing_columns}"
        )

    projection_df = scenario_data.copy()

    projection_df[
        "projected_value"
    ] = model.predict(
        projection_df[
            feature_columns
        ]
    )

    return projection_df



















# """
# Machine learning utilities for the Global Health Dashboard.

# This module handles:
# - ML dataset preparation
# - Multi-target regression
# - Time-aware train/test splitting
# - Model training
# - Model evaluation
# - Historical predictions
# - Future scenario-based projections
# """

# from __future__ import annotations

# from typing import Any

# import numpy as np
# import pandas as pd

# from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import (
#     GradientBoostingRegressor,
#     RandomForestRegressor,
# )
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_squared_error,
#     r2_score,
# )
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder


# # =========================================================
# # Prediction targets
# # =========================================================

# TARGETS = {
#     "Incidence Rate": "incidence_rate_pct",
#     "Mortality Rate": "mortality_rate_per_100_people_pct",
#     "Recovery Rate": "recovery_rate_pct",
# }


# TARGET_DISPLAY_NAMES = {
#     "incidence_rate_pct": "Incidence Rate (%)",
#     "mortality_rate_per_100_people_pct": (
#         "Mortality Rate per 100 People (%)"
#     ),
#     "recovery_rate_pct": "Recovery Rate (%)",
# }


# # =========================================================
# # Predictor variables
# # =========================================================

# NUMERIC_FEATURES = [
#     "year",
#     "healthcare_access_pct",
#     "doctors_per_1000",
#     "hospital_beds_per_1000",
#     "recovery_rate_pct",
#     "per_capita_income_usd",
#     "education_index",
#     "urbanization_rate_pct",
# ]


# CATEGORICAL_FEATURES = [
#     "country",
#     "disease_name",
# ]


# # =========================================================
# # Backward-compatible default target
# # =========================================================

# TARGET_COLUMN = "incidence_rate_pct"


# # =========================================================
# # Target-specific feature selection
# # =========================================================

# def get_feature_columns(
#     target_column: str,
# ) -> tuple[list[str], list[str]]:
#     """
#     Return numeric and categorical predictors for a target.

#     The selected target is removed from the predictors to
#     prevent direct target leakage.
#     """

#     numeric_features = [
#         feature
#         for feature in NUMERIC_FEATURES
#         if feature != target_column
#     ]

#     categorical_features = list(
#         CATEGORICAL_FEATURES
#     )

#     return numeric_features, categorical_features


# # =========================================================
# # ML dataset preparation
# # =========================================================

# def prepare_ml_dataset(
#     df: pd.DataFrame,
#     target_column: str = TARGET_COLUMN,
# ) -> pd.DataFrame:
#     """
#     Prepare a dataset for one prediction target.
#     """

#     numeric_features, categorical_features = (
#         get_feature_columns(target_column)
#     )

#     required_columns = [
#         target_column,
#         *numeric_features,
#         *categorical_features,
#     ]

#     missing_columns = [
#         column
#         for column in required_columns
#         if column not in df.columns
#     ]

#     if missing_columns:
#         raise ValueError(
#             "The following required ML columns are missing "
#             f"from the dataset: {missing_columns}"
#         )

#     ml_df = df[
#         required_columns
#     ].copy()

#     for column in numeric_features + [
#         target_column
#     ]:

#         ml_df[column] = pd.to_numeric(
#             ml_df[column],
#             errors="coerce",
#         )

#     for column in categorical_features:

#         ml_df[column] = (
#             ml_df[column]
#             .astype("string")
#             .str.strip()
#         )

#     ml_df = ml_df.dropna(
#         subset=[target_column]
#     )

#     ml_df = (
#         ml_df
#         .sort_values("year")
#         .reset_index(drop=True)
#     )

#     return ml_df


# # =========================================================
# # Time-based split
# # =========================================================

# def time_based_split(
#     ml_df: pd.DataFrame,
#     test_years: int = 5,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:

#     if "year" not in ml_df.columns:
#         raise ValueError(
#             "The ML dataset must contain a 'year' column."
#         )

#     available_years = sorted(
#         ml_df["year"]
#         .dropna()
#         .unique()
#     )

#     if len(available_years) <= test_years:

#         raise ValueError(
#             "There are not enough years available to create "
#             "the requested time-based train/test split."
#         )

#     test_year_values = (
#         available_years[-test_years:]
#     )

#     test_start_year = test_year_values[0]

#     train_df = ml_df[
#         ml_df["year"] < test_start_year
#     ].copy()

#     test_df = ml_df[
#         ml_df["year"] >= test_start_year
#     ].copy()

#     return train_df, test_df


# # =========================================================
# # Preprocessor
# # =========================================================

# def create_preprocessor(
#     numeric_features: list[str] | None = None,
#     categorical_features: list[str] | None = None,
# ) -> ColumnTransformer:

#     if numeric_features is None:
#         numeric_features = list(
#             NUMERIC_FEATURES
#         )

#     if categorical_features is None:
#         categorical_features = list(
#             CATEGORICAL_FEATURES
#         )

#     numeric_pipeline = Pipeline(
#         steps=[
#             (
#                 "imputer",
#                 SimpleImputer(
#                     strategy="median"
#                 ),
#             ),
#         ]
#     )

#     categorical_pipeline = Pipeline(
#         steps=[
#             (
#                 "imputer",
#                 SimpleImputer(
#                     strategy="most_frequent"
#                 ),
#             ),
#             (
#                 "onehot",
#                 OneHotEncoder(
#                     handle_unknown="ignore",
#                     sparse_output=False,
#                 ),
#             ),
#         ]
#     )

#     preprocessor = ColumnTransformer(
#         transformers=[
#             (
#                 "numeric",
#                 numeric_pipeline,
#                 numeric_features,
#             ),
#             (
#                 "categorical",
#                 categorical_pipeline,
#                 categorical_features,
#             ),
#         ]
#     )

#     return preprocessor


# # =========================================================
# # Model definitions
# # =========================================================

# def build_models(
#     random_state: int = 42,
# ) -> dict[str, Any]:

#     models = {

#         "Linear Regression":
#             LinearRegression(),

#         "Random Forest":
#             RandomForestRegressor(
#                 n_estimators=300,
#                 random_state=random_state,
#                 n_jobs=-1,
#             ),

#         "Gradient Boosting":
#             GradientBoostingRegressor(
#                 n_estimators=200,
#                 learning_rate=0.05,
#                 max_depth=3,
#                 random_state=random_state,
#             ),
#     }

#     return models


# # =========================================================
# # Train models
# # =========================================================

# def train_models(
#     train_df: pd.DataFrame,
#     target_column: str = TARGET_COLUMN,
# ) -> dict[str, Pipeline]:

#     numeric_features, categorical_features = (
#         get_feature_columns(
#             target_column
#         )
#     )

#     feature_columns = (
#         numeric_features
#         + categorical_features
#     )

#     X_train = train_df[
#         feature_columns
#     ]

#     y_train = train_df[
#         target_column
#     ]

#     models = build_models()

#     trained_models = {}

#     for model_name, model in models.items():

#         pipeline = Pipeline(
#             steps=[
#                 (
#                     "preprocessor",
#                     create_preprocessor(
#                         numeric_features,
#                         categorical_features,
#                     ),
#                 ),
#                 (
#                     "model",
#                     model,
#                 ),
#             ]
#         )

#         pipeline.fit(
#             X_train,
#             y_train,
#         )

#         trained_models[
#             model_name
#         ] = pipeline

#     return trained_models


# # =========================================================
# # Train all prediction systems
# # =========================================================

# def train_all_target_models(
#     train_data_by_target: dict[
#         str,
#         pd.DataFrame,
#     ],
# ) -> dict[
#     str,
#     dict[str, Pipeline],
# ]:

#     all_trained_models = {}

#     for target_column, train_df in (
#         train_data_by_target.items()
#     ):

#         all_trained_models[
#             target_column
#         ] = train_models(
#             train_df,
#             target_column=target_column,
#         )

#     return all_trained_models


# # =========================================================
# # Evaluate models
# # =========================================================

# def evaluate_models(
#     trained_models: dict[str, Pipeline],
#     test_df: pd.DataFrame,
#     target_column: str = TARGET_COLUMN,
# ) -> pd.DataFrame:

#     numeric_features, categorical_features = (
#         get_feature_columns(
#             target_column
#         )
#     )

#     feature_columns = (
#         numeric_features
#         + categorical_features
#     )

#     X_test = test_df[
#         feature_columns
#     ]

#     y_test = test_df[
#         target_column
#     ]

#     results = []

#     for model_name, model in (
#         trained_models.items()
#     ):

#         predictions = model.predict(
#             X_test
#         )

#         rmse = np.sqrt(
#             mean_squared_error(
#                 y_test,
#                 predictions,
#             )
#         )

#         mae = mean_absolute_error(
#             y_test,
#             predictions,
#         )

#         r2 = r2_score(
#             y_test,
#             predictions,
#         )

#         results.append(
#             {
#                 "Model": model_name,
#                 "R²": r2,
#                 "RMSE": rmse,
#                 "MAE": mae,
#             }
#         )

#     results_df = pd.DataFrame(
#         results
#     )

#     results_df = (
#         results_df
#         .sort_values(
#             "RMSE",
#             ascending=True,
#         )
#         .reset_index(drop=True)
#     )

#     return results_df


# # =========================================================
# # Generate historical predictions
# # =========================================================

# def generate_predictions(
#     model: Pipeline,
#     data: pd.DataFrame,
#     target_column: str = TARGET_COLUMN,
# ) -> pd.DataFrame:

#     numeric_features, categorical_features = (
#         get_feature_columns(
#             target_column
#         )
#     )

#     feature_columns = (
#         numeric_features
#         + categorical_features
#     )

#     prediction_df = data.copy()

#     prediction_df[
#         "predicted_value"
#     ] = model.predict(
#         prediction_df[
#             feature_columns
#         ]
#     )

#     if target_column in prediction_df.columns:

#         prediction_df[
#             "prediction_error"
#         ] = (
#             prediction_df[
#                 target_column
#             ]
#             - prediction_df[
#                 "predicted_value"
#             ]
#         )

#     return prediction_df


# # =========================================================
# # Generate future scenario projection
# # =========================================================

# def generate_future_projection(
#     model: Pipeline,
#     scenario_data: pd.DataFrame,
#     target_column: str = TARGET_COLUMN,
# ) -> pd.DataFrame:

#     numeric_features, categorical_features = (
#         get_feature_columns(
#             target_column
#         )
#     )

#     feature_columns = (
#         numeric_features
#         + categorical_features
#     )

#     projection_df = scenario_data.copy()

#     projection_df[
#         "projected_value"
#     ] = model.predict(
#         projection_df[
#             feature_columns
#         ]
#     )

#     return projection_df