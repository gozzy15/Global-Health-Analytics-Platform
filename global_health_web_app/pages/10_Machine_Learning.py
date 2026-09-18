"""
Machine Learning page for the Global Health Dashboard.
"""

import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_health_data
from utils.ml_models import (
    prepare_ml_dataset,
    time_based_split,
    train_models,
    train_all_target_models,
    evaluate_models,
    generate_predictions,
    generate_future_projection,
    generate_autonomous_projection,
    get_feature_columns,
    TARGET_COLUMN,
    TARGETS,
    TARGET_DISPLAY_NAMES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

from utils.style import apply_global_styles


# =========================================================
# HELPERS
# =========================================================

def make_export_name(value: str) -> str:
    """
    Convert a value into a safe filename component.
    """
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


@st.cache_data
def calculate_permutation_importance(
    _model,
    X_test,
    y_test,
    scoring_metric,
    model_name,
):
    return permutation_importance(
        _model,
        X_test,
        y_test,
        scoring=scoring_metric,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Machine Learning | Global Health Dashboard",
    page_icon="🤖",
    layout="wide",
)

apply_global_styles()


# =========================================================
# LOAD DATA
# =========================================================

df = load_health_data()


# =========================================================
# PAGE HEADER
# =========================================================

st.title("🤖 Machine Learning")

st.markdown(
    """
    This page applies machine learning techniques to the Global
    Health Dataset to investigate whether health, demographic,
    and socioeconomic indicators can be used to predict future
    health outcomes.
    """
)


# =========================================================
# SHARED STATE INITIALIZATION
#
# These variables are initialized before the tabs so that
# downstream tabs can safely reference them even when an
# earlier prediction or analysis is unavailable.
# =========================================================

ml_df = pd.DataFrame()

train_df = pd.DataFrame()
test_df = pd.DataFrame()

target_columns = [
    "incidence_rate_pct",
    "mortality_rate_per_100_people_pct",
    "recovery_rate_pct",
]

target_ml_data = {}
train_data_by_target = {}
failed_targets = []

all_trained_models = {}
successful_targets = []

evaluation_results = {}
evaluation_rows = []
results_df = pd.DataFrame()

best_models_df = pd.DataFrame()

# Prediction state
prediction_system = None
prediction_target = None
target_data = {}
target_ml_df = pd.DataFrame()
target_trained_models = {}
target_display_name = None

prediction_mode = "Historical Prediction"

# Historical prediction state
prediction_country = None
prediction_disease = None
prediction_year = None
prediction_model_name = None

selected_prediction_data = pd.DataFrame()
prediction_result = pd.DataFrame()

predicted_value = np.nan
actual_value = np.nan
prediction_error = np.nan

# Future projection state
future_projection_mode = "Autonomous Forecast"

projection_country = None
projection_disease = None
baseline_year = None
latest_country_disease_data = pd.Series(dtype="object")

projection_year = None
projection_model_name = None
manual_model_name = None

country_disease_data = pd.DataFrame()
projection_result = pd.DataFrame()
projection_feature_data = pd.DataFrame()
scenario_data = pd.DataFrame()
scenario_result = pd.DataFrame()
scenario_display = pd.DataFrame()

projected_value = np.nan
scenario_projected_value = np.nan

# Feature importance state
feature_importance_df = pd.DataFrame()
feature_importance_export = pd.DataFrame()
target_test_df = pd.DataFrame()
feature_columns = []

# Clustering state
country_cluster_df = pd.DataFrame()
cluster_summary = pd.DataFrame()

# Export state
model_evaluation_export = pd.DataFrame()
prediction_export = pd.DataFrame()
country_clustering_export = pd.DataFrame()
cluster_summary_export = pd.DataFrame()


# =========================================================
# PHASE 1 — PREPARE ML DATASET
#
# Shared preparation is performed before tabs because the
# training, evaluation, prediction, feature importance, and
# clustering tabs all depend on it.
# =========================================================

try:

    ml_df = prepare_ml_dataset(
        df
    )

except ValueError as error:

    st.error(
        str(error)
    )

    st.stop()


# =========================================================
# PHASE 1 — TIME SPLIT
# =========================================================

try:

    train_df, test_df = time_based_split(
        ml_df,
        test_years=5,
    )

except ValueError as error:

    st.error(
        str(error)
    )

    st.stop()


# =========================================================
# PHASE 2 — TRAIN ALL PREDICTION SYSTEMS
#
# This remains shared because several tabs depend on the
# trained models.
# =========================================================

with st.spinner(
    "Preparing training datasets..."
):

    for target_column in target_columns:

        try:

            target_df = prepare_ml_dataset(
                df,
                target_column=target_column,
            )

            target_train_df, target_test_df = (
                time_based_split(
                    target_df,
                    test_years=5,
                )
            )

            target_ml_data[
                target_column
            ] = {
                "data": target_df,
                "train": target_train_df,
                "test": target_test_df,
            }

            train_data_by_target[
                target_column
            ] = target_train_df

        except ValueError as exc:

            failed_targets.append(
                (
                    target_column,
                    str(exc),
                )
            )


@st.cache_resource(
    show_spinner="Training machine learning models..."
)
def get_cached_trained_models(
    training_data_by_target: dict[str, pd.DataFrame],
):
    return train_all_target_models(
        training_data_by_target
    )


try:

    all_trained_models = (
        get_cached_trained_models(
            train_data_by_target
        )
    )

except (
    ValueError,
    KeyError,
    TypeError,
) as exc:

    st.error(
        "Machine learning model training could not be completed. "
        f"Details: {exc}"
    )

    all_trained_models = {}


for target_column, models in (
    all_trained_models.items()
):

    if target_column in target_ml_data:

        target_ml_data[
            target_column
        ][
            "models"
        ] = models


successful_targets = list(
    all_trained_models.keys()
)


# =========================================================
# PHASE 3 — EVALUATE ALL PREDICTION SYSTEMS
#
# Shared because evaluation is used by both the evaluation
# and prediction-interpretation tabs.
# =========================================================

for target_column, target_info in (
    target_ml_data.items()
):

    models = target_info.get(
        "models"
    )

    test_df_target = target_info.get(
        "test"
    )

    if not models or test_df_target is None:

        continue

    try:

        target_results = evaluate_models(
            models,
            test_df_target,
            target_column=target_column,
        )

        evaluation_results[
            target_column
        ] = target_results

        display_name = TARGET_DISPLAY_NAMES.get(
            target_column,
            target_column,
        )

        for _, row in target_results.iterrows():

            evaluation_rows.append(
                {
                    "Prediction System": display_name,
                    "Model": row["Model"],
                    "R²": row["R²"],
                    "RMSE": row["RMSE"],
                    "MAE": row["MAE"],
                }
            )

    except ValueError:
        continue


results_df = pd.DataFrame(
    evaluation_rows
)


# =========================================================
# MAIN TABS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       STREAMLIT TABS — 4 TABS PER ROW
       ===================================================== */

    /* Main tab list */
    .stTabs [role="tablist"],
    .stTabs [data-baseweb="tab-list"] {
        display: grid !important;

        grid-template-columns:
            repeat(4, minmax(0, 1fr)) !important;

        width: 100% !important;
        max-width: 100% !important;

        height: auto !important;
        min-height: 0 !important;

        overflow: visible !important;

        gap: 0.25rem !important;

        white-space: normal !important;

        scrollbar-width: none !important;
    }

    /* Remove the intermediate Streamlit/BaseWeb wrapper
       from the layout so the actual tabs become grid items */
    .stTabs [role="tablist"] > div,
    .stTabs [data-baseweb="tab-list"] > div {
        display: contents !important;
    }

    /* Individual tabs */
    .stTabs [role="tab"],
    .stTabs [data-baseweb="tab"] {
        width: 100% !important;
        max-width: none !important;

        min-width: 0 !important;

        box-sizing: border-box !important;

        white-space: nowrap !important;

        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* Hide scrollbar */
    .stTabs [role="tablist"]::-webkit-scrollbar,
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
    }

    /* Keep the tab underline/border within each grid cell */
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        max-width: none !important;
    }

    /* =====================================================
       RESPONSIVE BEHAVIOUR
       ===================================================== */

    /* Medium screens: 2 tabs per row */
    @media (max-width: 900px) {

        .stTabs [role="tablist"],
        .stTabs [data-baseweb="tab-list"] {
            grid-template-columns:
                repeat(2, minmax(0, 1fr)) !important;
        }

    }

    /* Small screens: 1 tab per row */
    @media (max-width: 600px) {

        .stTabs [role="tablist"],
        .stTabs [data-baseweb="tab-list"] {
            grid-template-columns:
                1fr !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


(
    tab_data,
    tab_training,
    tab_evaluation,
    tab_prediction,
    tab_interpretation,
    tab_importance,
    tab_clustering,
    tab_exports,
) = st.tabs(
    [
        "🧹 Data Preparation",
        "🤖 Model Training",
        "📊 Model Evaluation",
        "🔮 Prediction",
        "🔎 Interpretation",
        "📌 Feature Importance",
        "🌍 Clustering",
        "📥 ML Exports",
    ]
)


# =========================================================
# TAB 1 — ML DATA PREPARATION
# =========================================================

with tab_data:

    st.header(
        "Phase 1 — ML Data Preparation"
    )

    st.markdown(
        """
        This phase prepares the cleaned health dataset for machine
        learning and establishes a time-based training and testing
        framework.
        """
    )

    # -----------------------------------------------------
    # Dataset overview
    # -----------------------------------------------------

    st.subheader(
        "ML Dataset Overview"
    )

    data_col1, data_col2, data_col3, data_col4 = (
        st.columns(4)
    )

    with data_col1:

        st.metric(
            "ML Observations",
            f"{len(ml_df):,}",
        )

    with data_col2:

        st.metric(
            "Features",
            len(
                ml_df.columns
            ) - 1,
        )

    with data_col3:

        st.metric(
            "Countries",
            ml_df[
                "country"
            ].nunique(),
        )

    with data_col4:

        st.metric(
            "Diseases",
            ml_df[
                "disease_name"
            ].nunique(),
        )

    # -----------------------------------------------------
    # Prediction target
    # -----------------------------------------------------

    st.subheader(
        "Prediction Target"
    )

    st.info(
        """
        **Prediction targets: Incidence Rate, Mortality Rate, and Recovery Rate**

        The machine learning models predict three health outcomes:
        incidence rate, mortality rate, and recovery rate. The models use
        historical year information together with country, disease,
        health-system, and socioeconomic indicators to generate predictions.
        """
    )

    # -----------------------------------------------------
    # Feature list
    # -----------------------------------------------------

    st.subheader(
        "Predictor Variables"
    )

    feature_display_names = {
        "year": "Year",
        "healthcare_access_pct": "Healthcare Access (%)",
        "doctors_per_1000": "Doctors per 1,000",
        "hospital_beds_per_1000": "Hospital Beds per 1,000",
        "recovery_rate_pct": "Recovery Rate (%)",
        "per_capita_income_usd": "Per Capita Income (USD)",
        "education_index": "Education Index",
        "urbanization_rate_pct": "Urbanization Rate (%)",
        "country": "Country",
        "disease_name": "Disease",
    }

    feature_table = [
        {
            "Variable": display_name,
            "Type": (
                "Categorical"
                if column in [
                    "country",
                    "disease_name",
                ]
                else "Numeric"
            ),
        }
        for column, display_name
        in feature_display_names.items()
    ]

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------
    # Time-based split
    # -----------------------------------------------------

    st.subheader(
        "Time-Based Train/Test Split"
    )

    split_col1, split_col2, split_col3, split_col4 = (
        st.columns(4)
    )

    with split_col1:

        st.metric(
            "Training Observations",
            f"{len(train_df):,}",
        )

    with split_col2:

        st.metric(
            "Testing Observations",
            f"{len(test_df):,}",
        )

    with split_col3:

        st.metric(
            "Training Period",
            (
                f"{int(train_df['year'].min())}"
                f"–"
                f"{int(train_df['year'].max())}"
            ),
        )

    with split_col4:

        st.metric(
            "Testing Period",
            (
                f"{int(test_df['year'].min())}"
                f"–"
                f"{int(test_df['year'].max())}"
            ),
        )

    st.caption(
        """
        The most recent five years are reserved for testing.
        This prevents future observations from being used to train
        the models.
        """
    )


# =========================================================
# TAB 2 — MODEL TRAINING
# =========================================================

with tab_training:

    st.header(
        "Phase 2 — Model Training"
    )

    st.markdown(
        """
        Three regression algorithms are trained for each
        available prediction system:
        Linear Regression, Random Forest, and Gradient Boosting.
        """
    )

    # -----------------------------------------------------
    # Training status
    # -----------------------------------------------------

    if successful_targets:

        st.success(
            "Machine learning models were trained successfully "
            f"for {len(successful_targets)} prediction system(s)."
        )

    if failed_targets:

        st.warning(
            "Some prediction systems could not be trained."
        )

        for target_column, error_message in (
            failed_targets
        ):

            st.caption(
                f"{TARGET_DISPLAY_NAMES.get(target_column, target_column)}: "
                f"{error_message}"
            )

    # -----------------------------------------------------
    # Model list
    # -----------------------------------------------------

    st.subheader(
        "Trained Models"
    )

    model_rows = []

    for target_column, models in (
        all_trained_models.items()
    ):

        display_name = TARGET_DISPLAY_NAMES.get(
            target_column,
            target_column,
        )

        for model_name in models:

            model_rows.append(
                {
                    "Prediction System": display_name,
                    "Model": model_name,
                    "Status": "Trained",
                }
            )

    model_table = pd.DataFrame(
        model_rows
    )

    st.dataframe(
        model_table,
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------
    # Training summary
    # -----------------------------------------------------

    training_summary = pd.DataFrame(
        [
            {
                "Prediction System":
                    TARGET_DISPLAY_NAMES.get(
                        target_column,
                        target_column,
                    ),
                "Training Observations":
                    len(
                        target_ml_data[
                            target_column
                        ]["train"]
                    )
                    if target_column in target_ml_data
                    else 0,
                "Testing Observations":
                    len(
                        target_ml_data[
                            target_column
                        ]["test"]
                    )
                    if target_column in target_ml_data
                    else 0,
                "Models Trained":
                    len(
                        all_trained_models.get(
                            target_column,
                            {},
                        )
                    ),
            }
            for target_column in target_columns
        ]
    )

    st.subheader(
        "Training Summary"
    )

    st.dataframe(
        training_summary,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# TAB 3 — MODEL EVALUATION
# =========================================================

with tab_evaluation:

    st.header(
        "Phase 3 — Model Evaluation"
    )

    st.markdown(
        """
        Each prediction system is evaluated independently on its
        unseen five-year test period using R², RMSE, and MAE.
        """
    )

    if not evaluation_results:

        st.error(
            "No prediction systems were successfully evaluated."
        )

    else:

        # -------------------------------------------------
        # Best models
        # -------------------------------------------------

        st.subheader(
            "Best Model by Prediction System"
        )

        best_model_rows = []

        for target_column, results_target_df in (
            evaluation_results.items()
        ):

            best_r2_row = results_target_df.loc[
                results_target_df["R²"].idxmax()
            ]

            best_rmse_row = results_target_df.loc[
                results_target_df["RMSE"].idxmin()
            ]

            best_mae_row = results_target_df.loc[
                results_target_df["MAE"].idxmin()
            ]

            display_name = TARGET_DISPLAY_NAMES.get(
                target_column,
                target_column,
            )

            best_model_rows.append(
                {
                    "Prediction System": display_name,
                    "Best R² Model": (
                        best_r2_row["Model"]
                    ),
                    "R²": best_r2_row["R²"],
                    "Best RMSE Model": (
                        best_rmse_row["Model"]
                    ),
                    "RMSE": best_rmse_row["RMSE"],
                    "Best MAE Model": (
                        best_mae_row["Model"]
                    ),
                    "MAE": best_mae_row["MAE"],
                }
            )

        best_models_df = pd.DataFrame(
            best_model_rows
        )

        for column in [
            "R²",
            "RMSE",
            "MAE",
        ]:

            best_models_df[column] = (
                best_models_df[column].round(4)
            )

        st.dataframe(
            best_models_df,
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------------------------------
        # Overall performance
        # -------------------------------------------------

        st.subheader(
            "Model Performance Comparison"
        )

        display_results = results_df.copy()

        for column in [
            "R²",
            "RMSE",
            "MAE",
        ]:

            if column in display_results.columns:

                display_results[column] = (
                    display_results[column].round(4)
                )

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------------------------------
        # Interactive performance
        # -------------------------------------------------

        st.subheader(
            "Model Performance"
        )

        available_targets = list(
            evaluation_results.keys()
        )

        target_labels = {
            target_column:
                TARGET_DISPLAY_NAMES.get(
                    target_column,
                    target_column,
                )
            for target_column in available_targets
        }

        selected_target = st.selectbox(
            "Select prediction system",
            available_targets,
            format_func=lambda target: (
                target_labels[target]
            ),
            key="evaluation_target",
        )

        metric_choice = st.selectbox(
            "Select evaluation metric",
            [
                "R²",
                "RMSE",
                "MAE",
            ],
            key="evaluation_metric",
        )

        selected_results = (
            evaluation_results[
                selected_target
            ]
            .copy()
        )

        fig_performance = px.bar(
            selected_results,
            x="Model",
            y=metric_choice,
            title=(
                f"{target_labels[selected_target]} "
                f"— Model Comparison — {metric_choice}"
            ),
            labels={
                "Model": "Model",
                metric_choice: metric_choice,
            },
        )

        fig_performance.update_layout(
            height=450,
        )

        st.plotly_chart(
            fig_performance,
            use_container_width=True,
        )

        # -------------------------------------------------
        # Interpretation
        # -------------------------------------------------

        st.subheader(
            "Model Evaluation Interpretation"
        )

        selected_best_r2 = selected_results.loc[
            selected_results["R²"].idxmax()
        ]

        selected_best_rmse = selected_results.loc[
            selected_results["RMSE"].idxmin()
        ]

        selected_best_mae = selected_results.loc[
            selected_results["MAE"].idxmin()
        ]

        selected_test_df = target_ml_data[
            selected_target
        ]["test"]

        test_start_year = int(
            selected_test_df["year"].min()
        )

        test_end_year = int(
            selected_test_df["year"].max()
        )

        st.markdown(
            f"""
            **Prediction System:** {target_labels[selected_target]}

            **Best R²:** {selected_best_r2["Model"]} with **{selected_best_r2["R²"]:.4f}**

            **Best RMSE:** {selected_best_rmse["Model"]} with **{selected_best_rmse["RMSE"]:.4f}**

            **Best MAE:** {selected_best_mae["Model"]} with **{selected_best_mae["MAE"]:.4f}**

            The models were evaluated on the unseen
            **{test_start_year}–{test_end_year}**
            test period rather than on the observations used
            for training.
            """
        )

        st.success(
            """
            **Machine learning baseline successfully established.**

            All available prediction systems have been trained
            and evaluated using a time-aware testing strategy.
            """
        )


# =========================================================
# TAB 4 — PREDICTION INTERFACE
# =========================================================

with tab_prediction:

    st.header(
        "Phase 4 — Prediction Interface"
    )

    st.markdown(
        """
        This interface supports multiple health-outcome prediction
        systems and two prediction modes.

        **Historical Prediction** evaluates a selected recorded
        observation against its actual outcome.

        **Future Projection** estimates a future outcome using
        user-defined scenario assumptions for the predictor variables.
        """
    )

    # =====================================================
    # Prediction System
    # =====================================================

    st.subheader(
        "Prediction System"
    )

    available_prediction_systems = []

    for display_name, target_column in TARGETS.items():

        if target_column in target_ml_data:

            available_prediction_systems.append(
                display_name
            )

    if not available_prediction_systems:

        st.error(
            "No prediction systems are currently available."
        )

    else:

        prediction_system = st.selectbox(
            "Select prediction system",
            available_prediction_systems,
            key="prediction_system",
        )

        prediction_target = TARGETS[
            prediction_system
        ]

        target_data = target_ml_data[
            prediction_target
        ]

        target_ml_df = target_data[
            "data"
        ]

        target_trained_models = target_data[
            "models"
        ]

        target_display_name = TARGET_DISPLAY_NAMES.get(
            prediction_target,
            prediction_system,
        )

        # =================================================
        # Prediction Mode
        # =================================================

        prediction_mode = st.radio(
            "Prediction mode",
            [
                "Historical Prediction",
                "Future Projection",
            ],
            horizontal=True,
            key="prediction_mode",
        )

        st.divider()

        # =================================================
        # HISTORICAL PREDICTION
        # =================================================

        if prediction_mode == "Historical Prediction":

            st.subheader(
                "Historical Prediction"
            )

            st.caption(
                """
                Select an observation that already exists in the dataset.
                The selected model prediction is compared with the
                recorded outcome.
                """
            )

            historical_col1, historical_col2 = (
                st.columns(2)
            )

            with historical_col1:

                prediction_country = st.selectbox(
                    "Select country",
                    sorted(
                        target_ml_df[
                            "country"
                        ]
                        .dropna()
                        .unique()
                        .tolist()
                    ),
                    key="prediction_country",
                )

            with historical_col2:

                prediction_disease = st.selectbox(
                    "Select disease",
                    sorted(
                        target_ml_df[
                            "disease_name"
                        ]
                        .dropna()
                        .unique()
                        .tolist()
                    ),
                    key="prediction_disease",
                )

            historical_selection_df = target_ml_df[
                (
                    target_ml_df[
                        "country"
                    ]
                    == prediction_country
                )
                &
                (
                    target_ml_df[
                        "disease_name"
                    ]
                    == prediction_disease
                )
            ].copy()

            selected_prediction_data = (
                historical_selection_df
            )

            if historical_selection_df.empty:

                st.warning(
                    """
                    No observations are available for the selected
                    country and disease combination.
                    """
                )

            else:

                historical_years = sorted(
                    historical_selection_df[
                        "year"
                    ]
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                )

                historical_col3, historical_col4 = (
                    st.columns(2)
                )

                with historical_col3:

                    prediction_year = st.selectbox(
                        "Select year",
                        historical_years,
                        index=len(
                            historical_years
                        ) - 1,
                        key="prediction_year",
                    )

                with historical_col4:

                    prediction_model_name = st.selectbox(
                        "Select model",
                        list(
                            target_trained_models.keys()
                        ),
                        key="prediction_model",
                    )

                selected_prediction_data = (
                    historical_selection_df[
                        historical_selection_df[
                            "year"
                        ]
                        == prediction_year
                    ]
                    .copy()
                )

                if selected_prediction_data.empty:

                    st.warning(
                        """
                        No observation is available for the selected
                        country, disease, and year combination.
                        """
                    )

                else:

                    try:

                        selected_model = target_trained_models[
                            prediction_model_name
                        ]

                        prediction_result = generate_predictions(
                            selected_model,
                            selected_prediction_data,
                            target_column=prediction_target,
                        )

                    except (
                        KeyError,
                        ValueError,
                        TypeError,
                    ) as exc:

                        st.error(
                            "The selected historical prediction could not "
                            f"be generated. Details: {exc}"
                        )

                        prediction_result = pd.DataFrame()

                    required_prediction_columns = {
                        "predicted_value",
                        prediction_target,
                        "prediction_error",
                    }

                    if (
                        prediction_result is None
                        or prediction_result.empty
                        or not required_prediction_columns.issubset(
                            prediction_result.columns
                        )
                    ):

                        st.error(
                            "The prediction engine returned an invalid "
                            "or incomplete prediction result."
                        )

                    else:

                        try:

                            predicted_value = float(
                                prediction_result[
                                    "predicted_value"
                                ].iloc[0]
                            )

                            actual_value = float(
                                prediction_result[
                                    prediction_target
                                ].iloc[0]
                            )

                            prediction_error = float(
                                prediction_result[
                                    "prediction_error"
                                ].iloc[0]
                            )

                        except (
                            IndexError,
                            KeyError,
                            TypeError,
                            ValueError,
                        ) as exc:

                            st.error(
                                "The prediction result could not be "
                                f"interpreted correctly. Details: {exc}"
                            )

                        else:

                            st.subheader(
                                "Prediction Result"
                            )

                            result_col1, result_col2, result_col3 = (
                                st.columns(3)
                            )

                            with result_col1:

                                st.metric(
                                    f"Predicted {prediction_system}",
                                    f"{predicted_value:.2f}%",
                                )

                            with result_col2:

                                st.metric(
                                    f"Actual {prediction_system}",
                                    f"{actual_value:.2f}%",
                                )

                            with result_col3:

                                st.metric(
                                    "Prediction Error",
                                    f"{prediction_error:+.2f} pp",
                                )

                            st.subheader(
                                "Prediction vs Actual"
                            )

                            comparison_df = pd.DataFrame(
                                {
                                    "Measure": [
                                        f"Actual {prediction_system}",
                                        f"Predicted {prediction_system}",
                                    ],
                                    target_display_name: [
                                        actual_value,
                                        predicted_value,
                                    ],
                                }
                            )

                            comparison_fig = px.bar(
                                comparison_df,
                                x="Measure",
                                y=target_display_name,
                                title=(
                                    f"{prediction_model_name} — "
                                    f"Actual vs Predicted "
                                    f"{prediction_system}"
                                ),
                                labels={
                                    "Measure": "",
                                    target_display_name:
                                        target_display_name,
                                },
                            )

                            comparison_fig.update_layout(
                                height=450,
                            )

                            st.plotly_chart(
                                comparison_fig,
                                use_container_width=True,
                            )

                            st.subheader(
                                "Selected Observation"
                            )

                            numeric_features, categorical_features = (
                                get_feature_columns(
                                    prediction_target
                                )
                            )

                            observation_columns = [
                                "country",
                                "disease_name",
                                *numeric_features,
                            ]

                            observation_display = (
                                selected_prediction_data[
                                    observation_columns
                                ]
                                .copy()
                            )

                            observation_display = (
                                observation_display.rename(
                                    columns={
                                        "country":
                                            "Country",

                                        "disease_name":
                                            "Disease",

                                        "year":
                                            "Year",

                                        "healthcare_access_pct":
                                            "Healthcare Access (%)",

                                        "doctors_per_1000":
                                            "Doctors per 1,000",

                                        "hospital_beds_per_1000":
                                            "Hospital Beds per 1,000",

                                        "recovery_rate_pct":
                                            "Recovery Rate (%)",

                                        "per_capita_income_usd":
                                            "Per Capita Income (USD)",

                                        "education_index":
                                            "Education Index",

                                        "urbanization_rate_pct":
                                            "Urbanization Rate (%)",
                                    }
                                )
                            )

                            st.dataframe(
                                observation_display,
                                use_container_width=True,
                                hide_index=True,
                            )

                            st.subheader(
                                "Prediction Summary"
                            )

                            if prediction_error > 0:

                                error_direction = (
                                    "The model underestimated the "
                                    "recorded value."
                                )

                            elif prediction_error < 0:

                                error_direction = (
                                    "The model overestimated the "
                                    "recorded value."
                                )

                            else:

                                error_direction = (
                                    "The model exactly matched the "
                                    "recorded value."
                                )

                            st.info(
                                f"""
                                **Prediction System:** {prediction_system}

                                **Model:** {prediction_model_name}

                                **Country:** {prediction_country}

                                **Disease:** {prediction_disease}

                                **Year:** {int(prediction_year)}

                                The model estimated a
                                **{predicted_value:.2f}%**
                                {prediction_system.lower()}, compared with a recorded
                                value of **{actual_value:.2f}%**.

                                The prediction error was
                                **{prediction_error:+.2f} percentage points**.

                                {error_direction}
                                """
                            )

        # =================================================
        # FUTURE PROJECTION
        # =================================================

        else:

            st.subheader(
                "Future Projection"
            )

            st.warning(
                """
                Future projections are estimates based on historical
                relationships in the dataset. Autonomous forecasts
                estimate future predictor values from historical trends,
                while manual scenarios allow you to define the predictor
                assumptions yourself.
                """
            )

            future_projection_mode = st.radio(
                "Select projection mode",
                [
                    "Autonomous Forecast",
                    "Manual Scenario",
                ],
                horizontal=True,
                key="future_projection_mode",
            )

            future_col1, future_col2 = (
                st.columns(2)
            )

            with future_col1:

                projection_country = st.selectbox(
                    "Select country",
                    sorted(
                        target_ml_df[
                            "country"
                        ]
                        .dropna()
                        .unique()
                        .tolist()
                    ),
                    key="projection_country",
                )

            with future_col2:

                projection_disease = st.selectbox(
                    "Select disease",
                    sorted(
                        target_ml_df[
                            "disease_name"
                        ]
                        .dropna()
                        .unique()
                        .tolist()
                    ),
                    key="projection_disease",
                )

            country_disease_data = target_ml_df[
                (
                    target_ml_df[
                        "country"
                    ]
                    == projection_country
                )
                &
                (
                    target_ml_df[
                        "disease_name"
                    ]
                    == projection_disease
                )
            ].copy()

            if country_disease_data.empty:

                st.warning(
                    """
                    No historical observations are available for the
                    selected country and disease combination, so a future
                    projection cannot be generated.
                    """
                )

            else:

                country_disease_data = (
                    country_disease_data
                    .sort_values(
                        "year"
                    )
                    .reset_index(
                        drop=True
                    )
                )

                baseline_year = int(
                    country_disease_data[
                        "year"
                    ].max()
                )

                latest_country_disease_data = (
                    country_disease_data[
                        country_disease_data[
                            "year"
                        ]
                        == baseline_year
                    ]
                    .iloc[0]
                )

                # =============================================
                # AUTONOMOUS FORECAST
                # =============================================

                if (
                    future_projection_mode
                    == "Autonomous Forecast"
                ):

                    st.subheader(
                        "Autonomous Forecast"
                    )

                    st.caption(
                        """
                        The system forecasts the selected predictor
                        variables from the historical country-disease
                        trend and uses those projected values as inputs
                        to the selected machine learning model.
                        """
                    )

                    autonomous_col1, autonomous_col2 = (
                        st.columns(2)
                    )

                    with autonomous_col1:

                        projection_year = st.number_input(
                            "Projection year",
                            min_value=baseline_year + 1,
                            max_value=baseline_year + 10,
                            value=baseline_year + 1,
                            step=1,
                            key="autonomous_projection_year",
                        )

                    with autonomous_col2:

                        projection_model_name = st.selectbox(
                            "Select model",
                            list(
                                target_trained_models.keys()
                            ),
                            key="autonomous_projection_model",
                        )

                    try:

                        projection_model = (
                            target_trained_models[
                                projection_model_name
                            ]
                        )

                        autonomous_result = (
                            generate_autonomous_projection(
                                model=projection_model,
                                historical_data=country_disease_data,
                                future_year=int(
                                    projection_year
                                ),
                                target_column=prediction_target,
                                country=projection_country,
                                disease_name=projection_disease,
                            )
                        )

                        if (
                            autonomous_result is None
                            or autonomous_result.empty
                            or "projected_value"
                            not in autonomous_result.columns
                        ):

                            st.error(
                                """
                                The autonomous forecasting engine returned
                                an invalid or incomplete projection.
                                """
                            )

                        else:

                            projection_feature_data = (
                                autonomous_result
                                .drop(
                                    columns=["projected_value"],
                                    errors="ignore",
                                )
                                .copy()
                            )

                            projection_result = (
                                autonomous_result.copy()
                            )

                            try:

                                projected_value = float(
                                    autonomous_result[
                                        "projected_value"
                                    ].iloc[0]
                                )

                            except (
                                IndexError,
                                KeyError,
                                TypeError,
                                ValueError,
                            ) as exc:

                                st.error(
                                    "The projected value could not be "
                                    f"interpreted correctly. Details: {exc}"
                                )

                            else:

                                st.subheader(
                                    "Projected Outcome"
                                )

                                autonomous_result_col1, autonomous_result_col2 = (
                                    st.columns(2)
                                )

                                with autonomous_result_col1:

                                    st.metric(
                                        f"Projected {prediction_system}",
                                        f"{projected_value:.2f}%",
                                    )

                                with autonomous_result_col2:

                                    st.metric(
                                        "Projection Year",
                                        int(projection_year),
                                    )

                                st.info(
                                    f"""
                                    The **{projection_model_name}** model projects a
                                    **{projected_value:.2f}%**
                                    {prediction_system.lower()} for
                                    **{projection_country}**
                                    and **{projection_disease}**
                                    in **{int(projection_year)}**.

                                    The predictor variables were automatically
                                    forecast from historical observations for the
                                    selected country and disease.

                                    This is an autonomous model-based forecast and
                                    should not be interpreted as a validated real-world
                                    forecast.
                                    """
                                )

                                st.subheader(
                                    "Forecasted Predictor Values"
                                )

                                st.caption(
                                    f"""
                                    These are the predictor values estimated by the
                                    forecasting layer for {int(projection_year)}.
                                    """
                                )

                                autonomous_display = (
                                    autonomous_result.rename(
                                        columns={
                                            "year":
                                                "Year",

                                            "healthcare_access_pct":
                                                "Healthcare Access (%)",

                                            "doctors_per_1000":
                                                "Doctors per 1,000",

                                            "hospital_beds_per_1000":
                                                "Hospital Beds per 1,000",

                                            "recovery_rate_pct":
                                                "Recovery Rate (%)",

                                            "per_capita_income_usd":
                                                "Per Capita Income (USD)",

                                            "education_index":
                                                "Education Index",

                                            "urbanization_rate_pct":
                                                "Urbanization Rate (%)",

                                            "country":
                                                "Country",

                                            "disease_name":
                                                "Disease",

                                            "projected_value":
                                                "Projected Outcome",
                                        }
                                    )
                                )

                                st.dataframe(
                                    autonomous_display,
                                    use_container_width=True,
                                    hide_index=True,
                                )

                                scenario_display = (
                                    autonomous_display.copy()
                                )

                                st.subheader(
                                    "Forecast Context"
                                )

                                context_col1, context_col2 = (
                                    st.columns(2)
                                )

                                with context_col1:

                                    st.metric(
                                        "Historical Baseline Year",
                                        baseline_year,
                                    )

                                with context_col2:

                                    years_of_history = (
                                        country_disease_data[
                                            "year"
                                        ]
                                        .dropna()
                                        .nunique()
                                    )

                                    st.metric(
                                        "Historical Years Used",
                                        int(years_of_history),
                                    )

                                st.subheader(
                                    "Forecast Trend Diagnostic"
                                )

                                st.caption(
                                    """
                                    These are the predictor values generated for the selected
                                    future year. They are forecast from the historical
                                    country-disease trend and then supplied to the trained
                                    prediction model.
                                    """
                                )

                                forecast_diagnostic_columns = [
                                    column
                                    for column in projection_feature_data.columns
                                    if column not in [
                                        "country",
                                        "disease_name",
                                    ]
                                ]

                                forecast_diagnostic = (
                                    projection_feature_data[
                                        forecast_diagnostic_columns
                                    ]
                                    .copy()
                                    .T
                                    .reset_index()
                                )

                                forecast_diagnostic.columns = [
                                    "Feature",
                                    "Forecasted Value",
                                ]

                                st.dataframe(
                                    forecast_diagnostic,
                                    use_container_width=True,
                                    hide_index=True,
                                )

                    except (
                        KeyError,
                        ValueError,
                        TypeError,
                    ) as exc:

                        st.error(
                            "The autonomous forecast could not be generated. "
                            f"Details: {exc}"
                        )

                # =============================================
                # MANUAL SCENARIO
                # =============================================

                else:

                    st.subheader(
                        "Manual Scenario"
                    )

                    st.caption(
                        """
                        Define your own predictor assumptions and see how
                        the selected machine learning model responds.
                        The model uses these values directly; no future
                        predictor forecasting is performed.
                        """
                    )

                    manual_model_name = st.selectbox(
                        "Select model",
                        list(
                            target_trained_models.keys()
                        ),
                        key="manual_projection_model",
                    )

                    numeric_features, categorical_features = (
                        get_feature_columns(
                            prediction_target
                        )
                    )

                    scenario_numeric_features = [
                        feature
                        for feature in numeric_features
                        if feature != "year"
                    ]

                    st.subheader(
                        "Scenario Assumptions"
                    )

                    st.caption(
                        f"""
                        The default values below are taken from the latest
                        available {baseline_year} observation for
                        {projection_country} and {projection_disease}.
                        Adjust them to create your own scenario.
                        """
                    )

                    scenario_values = {}

                    scenario_col1, scenario_col2 = (
                        st.columns(2)
                    )

                    def get_baseline_value(
                        column_name: str,
                    ) -> float:

                        value = (
                            latest_country_disease_data[
                                column_name
                            ]
                        )

                        if pd.isna(value):

                            column_values = (
                                target_ml_df[
                                    column_name
                                ]
                                .dropna()
                            )

                            if column_values.empty:

                                return 0.0

                            return float(
                                column_values.median()
                            )

                        return float(value)

                    left_features = [
                        feature
                        for feature in scenario_numeric_features
                        if feature in [
                            "healthcare_access_pct",
                            "doctors_per_1000",
                            "hospital_beds_per_1000",
                            "recovery_rate_pct",
                        ]
                    ]

                    right_features = [
                        feature
                        for feature in scenario_numeric_features
                        if feature not in left_features
                    ]

                    with scenario_col1:

                        for feature in left_features:

                            baseline_value = (
                                get_baseline_value(
                                    feature
                                )
                            )

                            if feature == (
                                "healthcare_access_pct"
                            ):

                                scenario_values[
                                    feature
                                ] = st.number_input(
                                    "Healthcare Access (%)",
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=baseline_value,
                                    step=0.5,
                                    key=(
                                        "manual_scenario_"
                                        + feature
                                    ),
                                )

                            elif feature == (
                                "doctors_per_1000"
                            ):

                                scenario_values[
                                    feature
                                ] = st.number_input(
                                    "Doctors per 1,000",
                                    min_value=0.0,
                                    value=baseline_value,
                                    step=0.1,
                                    key=(
                                        "manual_scenario_"
                                        + feature
                                    ),
                                )

                            elif feature == (
                                "hospital_beds_per_1000"
                            ):

                                scenario_values[
                                    feature
                                ] = st.number_input(
                                    "Hospital Beds per 1,000",
                                    min_value=0.0,
                                    value=baseline_value,
                                    step=0.1,
                                    key=(
                                        "manual_scenario_"
                                        + feature
                                    ),
                                )

                            elif feature == (
                                "recovery_rate_pct"
                            ):

                                scenario_values[
                                    feature
                                ] = st.number_input(
                                    "Recovery Rate (%)",
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=baseline_value,
                                    step=0.5,
                                    key=(
                                        "manual_scenario_"
                                        + feature
                                    ),
                                )

                    with scenario_col2:

                        for feature in right_features:

                            baseline_value = (
                                get_baseline_value(
                                    feature
                                )
                            )

                            if feature == (
                                "per_capita_income_usd"
                            ):

                                scenario_values[
                                    feature
                                ] = st.number_input(
                                    "Per Capita Income (USD)",
                                    min_value=0.0,
                                    value=baseline_value,
                                    step=100.0,
                                    key=(
                                        "manual_scenario_"
                                        + feature
                                    ),
                                )

                            elif feature == (
                                "education_index"
                            ):

                                scenario_values[
                                    feature
                                ] = st.number_input(
                                    "Education Index",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=baseline_value,
                                    step=0.01,
                                    key=(
                                        "manual_scenario_"
                                        + feature
                                    ),
                                )

                            elif feature == (
                                "urbanization_rate_pct"
                            ):

                                scenario_values[
                                    feature
                                ] = st.number_input(
                                    "Urbanization Rate (%)",
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=baseline_value,
                                    step=0.5,
                                    key=(
                                        "manual_scenario_"
                                        + feature
                                    ),
                                )

                    scenario_record = {
                        "year": baseline_year,
                        "country": projection_country,
                        "disease_name": projection_disease,
                    }

                    scenario_record.update(
                        scenario_values
                    )

                    scenario_data = pd.DataFrame(
                        [scenario_record]
                    )

                    try:

                        scenario_model = (
                            target_trained_models[
                                manual_model_name
                            ]
                        )

                        scenario_result = (
                            generate_future_projection(
                                scenario_model,
                                scenario_data,
                                target_column=prediction_target,
                            )
                        )

                        if (
                            scenario_result is None
                            or scenario_result.empty
                            or "projected_value"
                            not in scenario_result.columns
                        ):

                            st.error(
                                """
                                The manual scenario engine returned an invalid
                                or incomplete projection.
                                """
                            )

                        else:

                            projection_result = (
                                scenario_result.copy()
                            )

                            projection_feature_data = (
                                scenario_result
                                .drop(
                                    columns=["projected_value"],
                                    errors="ignore",
                                )
                                .copy()
                            )

                            try:

                                scenario_projected_value = float(
                                    scenario_result[
                                        "projected_value"
                                    ].iloc[0]
                                )

                            except (
                                IndexError,
                                KeyError,
                                TypeError,
                                ValueError,
                            ) as exc:

                                st.error(
                                    "The scenario projected value could not "
                                    f"be interpreted correctly. Details: {exc}"
                                )

                            else:

                                projected_value = (
                                    scenario_projected_value
                                )

                                st.subheader(
                                    "Scenario Outcome"
                                )

                                scenario_result_col1, scenario_result_col2 = (
                                    st.columns(2)
                                )

                                with scenario_result_col1:

                                    st.metric(
                                        f"Projected {prediction_system}",
                                        f"{scenario_projected_value:.2f}%",
                                    )

                                with scenario_result_col2:

                                    st.metric(
                                        "Scenario Baseline",
                                        baseline_year,
                                    )

                                st.info(
                                    f"""
                                    The **{manual_model_name}** model produces a
                                    **{scenario_projected_value:.2f}%**
                                    {prediction_system.lower()} for
                                    **{projection_country}**
                                    and **{projection_disease}**
                                    under your manually defined predictor assumptions.

                                    The values are supplied directly by the user and
                                    are not automatically forecast from historical trends.
                                    """
                                )

                                st.subheader(
                                    "Scenario Inputs"
                                )

                                scenario_display = (
                                    scenario_result.rename(
                                        columns={
                                            "year":
                                                "Reference Year",

                                            "healthcare_access_pct":
                                                "Healthcare Access (%)",

                                            "doctors_per_1000":
                                                "Doctors per 1,000",

                                            "hospital_beds_per_1000":
                                                "Hospital Beds per 1,000",

                                            "recovery_rate_pct":
                                                "Recovery Rate (%)",

                                            "per_capita_income_usd":
                                                "Per Capita Income (USD)",

                                            "education_index":
                                                "Education Index",

                                            "urbanization_rate_pct":
                                                "Urbanization Rate (%)",

                                            "country":
                                                "Country",

                                            "disease_name":
                                                "Disease",

                                            "projected_value":
                                                "Projected Outcome",
                                        }
                                    )
                                )

                                st.dataframe(
                                    scenario_display,
                                    use_container_width=True,
                                    hide_index=True,
                                )

                    except (
                        KeyError,
                        ValueError,
                        TypeError,
                    ) as exc:

                        st.error(
                            "The manual scenario projection could not be "
                            f"generated. Details: {exc}"
                        )


# =========================================================
# TAB 5 — PREDICTION INTERPRETATION
# =========================================================

with tab_interpretation:

    st.header(
        "🔎 Phase 5 — Prediction Interpretation"
    )

    if prediction_system is None:

        st.info(
            "Select a prediction system in the Prediction Interface tab "
            "to view prediction interpretation."
        )

    elif prediction_mode == "Historical Prediction":

        st.markdown(
            f"""
            This section interprets the selected
            **{prediction_system}** using the selected historical
            observation.
            """
        )

        if (
            selected_prediction_data.empty
            or pd.isna(predicted_value)
            or pd.isna(actual_value)
            or pd.isna(prediction_error)
            or prediction_model_name is None
        ):

            st.warning(
                "Prediction interpretation requires a valid "
                "selected historical prediction."
            )

        else:

            # =================================================
            # Prediction Assessment
            # =================================================

            st.subheader(
                "Prediction Assessment"
            )

            absolute_error = abs(
                prediction_error
            )

            if actual_value != 0:

                relative_prediction_difference = (
                    absolute_error
                    / abs(actual_value)
                ) * 100

            else:

                relative_prediction_difference = np.nan

            if absolute_error < 0.10:

                assessment = (
                    "Essentially identical"
                )

                assessment_message = (
                    "The model's prediction is extremely close to "
                    f"the actual {prediction_system.lower()}, with "
                    "only a negligible difference."
                )

            elif absolute_error < 1:

                assessment = (
                    "Very close prediction"
                )

                assessment_message = (
                    "The model's prediction is very close to the "
                    f"actual {prediction_system.lower()}."
                )

            elif absolute_error < 3:

                assessment = (
                    "Small prediction error"
                )

                assessment_message = (
                    "The model produced a relatively small prediction "
                    f"error for this {prediction_system.lower()} observation."
                )

            elif absolute_error < 5:

                assessment = (
                    "Moderate prediction error"
                )

                assessment_message = (
                    "The model produced a moderate prediction error "
                    f"for this {prediction_system.lower()} observation."
                )

            else:

                assessment = (
                    "Large prediction error"
                )

                assessment_message = (
                    "The model's prediction differs substantially from "
                    f"the actual {prediction_system.lower()}."
                )

            if prediction_error > 0:

                prediction_direction = (
                    "Underestimated"
                )

            elif prediction_error < 0:

                prediction_direction = (
                    "Overestimated"
                )

            else:

                prediction_direction = (
                    "Exact match"
                )

            if absolute_error < 0.10:

                direction_text = (
                    "essentially the same as"
                )

            elif prediction_error > 0:

                if actual_value != 0:

                    if relative_prediction_difference < 10:

                        direction_text = (
                            "slightly higher than"
                        )

                    elif relative_prediction_difference < 30:

                        direction_text = (
                            "moderately higher than"
                        )

                    else:

                        direction_text = (
                            "substantially higher than"
                        )

                else:

                    if absolute_error < 1:

                        direction_text = (
                            "slightly higher than"
                        )

                    elif absolute_error < 3:

                        direction_text = (
                            "moderately higher than"
                        )

                    else:

                        direction_text = (
                            "substantially higher than"
                        )

            else:

                if actual_value != 0:

                    if relative_prediction_difference < 10:

                        direction_text = (
                            "slightly lower than"
                        )

                    elif relative_prediction_difference < 30:

                        direction_text = (
                            "moderately lower than"
                        )

                    else:

                        direction_text = (
                            "substantially lower than"
                        )

                else:

                    if absolute_error < 1:

                        direction_text = (
                            "slightly lower than"
                        )

                    elif absolute_error < 3:

                        direction_text = (
                            "moderately lower than"
                        )

                    else:

                        direction_text = (
                            "substantially lower than"
                        )

            assessment_col1, assessment_col2, assessment_col3 = (
                st.columns(3)
            )

            with assessment_col1:

                st.markdown(
                    """
                    <div style="
                        font-size: 1.05rem;
                        font-weight: 600;
                        margin-bottom: 0.35rem;
                    ">
                        Prediction Assessment
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div style="
                        font-size: 1.5rem;
                        font-weight: 600;
                        line-height: 1.3;
                        white-space: normal;
                        overflow-wrap: break-word;
                        word-break: normal;
                        margin-bottom: 1.5rem;
                    ">
                        {assessment}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with assessment_col2:

                st.markdown(
                    """
                    <div style="
                        font-size: 1.05rem;
                        font-weight: 600;
                        margin-bottom: 0.35rem;
                    ">
                        Prediction Direction
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div style="
                        font-size: 1.5rem;
                        font-weight: 600;
                        line-height: 1.3;
                        white-space: normal;
                        overflow-wrap: break-word;
                        word-break: normal;
                    ">
                        {prediction_direction}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with assessment_col3:

                st.markdown(
                    """
                    <div style="
                        font-size: 1.05rem;
                        font-weight: 600;
                        margin-bottom: 0.35rem;
                    ">
                        Relative Difference
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if pd.notna(
                    relative_prediction_difference
                ):

                    relative_difference_display = (
                        f"{relative_prediction_difference:.1f}%"
                    )

                else:

                    relative_difference_display = (
                        "N/A"
                    )

                st.markdown(
                    f"""
                    <div style="
                        font-size: 1.5rem;
                        font-weight: 600;
                        line-height: 1.3;
                        white-space: normal;
                        overflow-wrap: break-word;
                        word-break: normal;
                    ">
                        {relative_difference_display}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.info(
                assessment_message
            )

            # =================================================
            # Prediction Reliability
            # =================================================

            st.subheader(
                "Prediction Reliability"
            )

            selected_target_results = (
                evaluation_results.get(
                    prediction_target,
                    pd.DataFrame(),
                )
            )

            selected_model_rmse_series = (
                selected_target_results.loc[
                    selected_target_results["Model"]
                    == prediction_model_name,
                    "RMSE",
                ]
                if not selected_target_results.empty
                else pd.Series(dtype=float)
            )

            selected_model_mae_series = (
                selected_target_results.loc[
                    selected_target_results["Model"]
                    == prediction_model_name,
                    "MAE",
                ]
                if not selected_target_results.empty
                else pd.Series(dtype=float)
            )

            if not selected_model_rmse_series.empty:

                selected_model_rmse = float(
                    selected_model_rmse_series.iloc[0]
                )

            else:

                selected_model_rmse = np.nan

            if not selected_model_mae_series.empty:

                selected_model_mae = float(
                    selected_model_mae_series.iloc[0]
                )

            else:

                selected_model_mae = np.nan

            if (
                pd.notna(selected_model_rmse)
                and selected_model_rmse > 0
            ):

                error_to_rmse_ratio = (
                    absolute_error
                    / selected_model_rmse
                )

            else:

                error_to_rmse_ratio = np.nan

            reliability_col1, reliability_col2, reliability_col3 = (
                st.columns(3)
            )

            with reliability_col1:

                st.metric(
                    "Model Test RMSE",
                    (
                        f"{selected_model_rmse:.2f}"
                        if pd.notna(selected_model_rmse)
                        else "N/A"
                    ),
                )

            with reliability_col2:

                st.metric(
                    "Model Test MAE",
                    (
                        f"{selected_model_mae:.2f}"
                        if pd.notna(selected_model_mae)
                        else "N/A"
                    ),
                )

            with reliability_col3:

                st.metric(
                    "Error / RMSE",
                    (
                        f"{error_to_rmse_ratio:.2f}×"
                        if pd.notna(error_to_rmse_ratio)
                        else "N/A"
                    ),
                )

            if pd.notna(
                error_to_rmse_ratio
            ):

                if error_to_rmse_ratio <= 1:

                    reliability_message = (
                        "This prediction error is within the model's "
                        "overall RMSE range and is consistent with the "
                        "model's typical test performance."
                    )

                elif error_to_rmse_ratio <= 2:

                    reliability_message = (
                        "This prediction error is larger than the "
                        "model's RMSE but remains within a reasonable "
                        "range of the model's overall test performance."
                    )

                elif error_to_rmse_ratio <= 3:

                    reliability_message = (
                        "This prediction error is substantially larger "
                        "than the model's RMSE. The selected observation "
                        "appears more difficult for the model to estimate."
                    )

                else:

                    reliability_message = (
                        "This prediction error is considerably larger "
                        "than the model's RMSE. The selected observation "
                        "appears unusually difficult for the model to estimate."
                    )

                st.info(
                    reliability_message
                )

            # =================================================
            # Model Agreement
            # =================================================

            st.subheader(
                "Model Agreement"
            )

            numeric_features, categorical_features = (
                get_feature_columns(
                    prediction_target
                )
            )

            feature_columns = (
                numeric_features
                + categorical_features
            )

            model_agreement_results = []

            for model_name, model in (
                target_trained_models.items()
            ):

                try:

                    model_prediction = model.predict(
                        selected_prediction_data[
                            feature_columns
                        ]
                    )[0]

                    model_agreement_results.append(
                        {
                            "Model": model_name,
                            (
                                f"Predicted {prediction_system} (%)"
                            ): model_prediction,
                        }
                    )

                except (
                    KeyError,
                    ValueError,
                    TypeError,
                ) as exc:

                    st.warning(
                        f"Could not generate a prediction from "
                        f"{model_name} for the selected observation: "
                        f"{exc}"
                    )

            model_agreement_df = pd.DataFrame(
                model_agreement_results
            )

            if model_agreement_df.empty:

                st.warning(
                    "No model-agreement predictions could be generated "
                    "for the selected observation."
                )

            else:

                agreement_value_column = (
                    f"Predicted {prediction_system} (%)"
                )

                model_agreement_df[
                    agreement_value_column
                ] = (
                    model_agreement_df[
                        agreement_value_column
                    ].round(2)
                )

                st.dataframe(
                    model_agreement_df,
                    use_container_width=True,
                    hide_index=True,
                )

                fig_model_agreement = px.bar(
                    model_agreement_df,
                    x="Model",
                    y=agreement_value_column,
                    title=(
                        f"Model Predictions for Selected "
                        f"{prediction_system}"
                    ),
                    labels={
                        "Model": "Model",
                        agreement_value_column:
                            f"{prediction_system} (%)",
                    },
                )

                fig_model_agreement.add_hline(
                    y=actual_value,
                    line_dash="dash",
                    annotation_text=(
                        f"Actual {prediction_system}"
                    ),
                )

                fig_model_agreement.update_layout(
                    height=450,
                )

                st.plotly_chart(
                    fig_model_agreement,
                    use_container_width=True,
                )

                predictions = model_agreement_df[
                    agreement_value_column
                ]

                prediction_range = (
                    predictions.max()
                    - predictions.min()
                )

                if prediction_range < 1:

                    agreement_level = (
                        "Strong model agreement"
                    )

                    agreement_message = (
                        "The models produce very similar predictions "
                        f"for this {prediction_system.lower()}. This "
                        "indicates strong agreement across the different "
                        "algorithms."
                    )

                elif prediction_range < 3:

                    agreement_level = (
                        "Moderate model agreement"
                    )

                    agreement_message = (
                        "The models show some variation in their "
                        f"{prediction_system.lower()} estimates, but "
                        "their predictions remain reasonably close."
                    )

                elif prediction_range < 5:

                    agreement_level = (
                        "Noticeable model disagreement"
                    )

                    agreement_message = (
                        "The models produce noticeably different "
                        f"{prediction_system.lower()} estimates. The "
                        "prediction should therefore be interpreted "
                        "with some caution."
                    )

                else:

                    agreement_level = (
                        "Strong model disagreement"
                    )

                    agreement_message = (
                        "The models produce substantially different "
                        f"{prediction_system.lower()} estimates. This "
                        "indicates meaningful model disagreement and "
                        "suggests greater caution when interpreting "
                        "the selected prediction."
                    )

                st.markdown(
                    """
                    <style>

                    /* Allow long metric values to wrap */
                    [data-testid="stMetricValue"] {
                        white-space: normal !important;
                        overflow-wrap: anywhere !important;
                        word-break: normal !important;
                        line-height: 1.25 !important;
                        font-size: 1.6rem !important;
                        font-weight: 700 !important;
                    }

                    /* Keep metric labels slightly prominent */
                    [data-testid="stMetricLabel"] {
                        font-size: 0.9rem !important;
                        font-weight: 500 !important;
                    }

                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                agreement_col1, agreement_col2 = st.columns(2)

                with agreement_col1:

                    st.metric(
                        "Model Agreement",
                        agreement_level,
                    )

                with agreement_col2:

                    st.metric(
                        "Prediction Range",
                        f"{prediction_range:.2f} percentage points",
                    )

                st.info(
                    agreement_message
                )

                # =================================================
                # Overall Prediction Interpretation
                # =================================================

                st.subheader(
                    "Prediction Interpretation"
                )

                st.markdown(
                    f"""
                    The **{prediction_model_name}** model predicted a
                    **{predicted_value:.2f}%**
                    {prediction_system.lower()} for
                    **{prediction_country} — {prediction_disease}**
                    in **{int(prediction_year)}**.

                    The prediction error was **{prediction_error:.2f} percentage points**,
                    compared with the recorded outcome of
                    **{actual_value:.2f}%**.
                    """
                )

                if pd.notna(
                    relative_prediction_difference
                ):

                    st.markdown(
                        f"""
                        Relative to the actual
                        {prediction_system.lower()}, the prediction
                        differed by approximately
                        **{relative_prediction_difference:.1f}%**.
                        """
                    )

                else:

                    st.markdown(
                        f"""
                        Because the actual
                        {prediction_system.lower()} was zero, a relative
                        percentage difference is not meaningful for this
                        observation. The prediction difference is therefore
                        interpreted using the absolute error in percentage
                        points.
                        """
                    )

                st.markdown(
                    f"""
                    The individual prediction is classified as
                    **{assessment.lower()}**.

                    The selected model has an overall test RMSE of
                    **{selected_model_rmse:.2f} percentage points** and
                    an overall test MAE of
                    **{selected_model_mae:.2f} percentage points**.
                    """
                )

                if pd.notna(
                    error_to_rmse_ratio
                ):

                    if error_to_rmse_ratio <= 1:

                        overall_reliability = (
                            "The individual prediction error is within "
                            "the model's typical RMSE range."
                        )

                    elif error_to_rmse_ratio <= 2:

                        overall_reliability = (
                            "The individual prediction error is above "
                            "the model's RMSE but remains within a "
                            "reasonable range of its typical performance."
                        )

                    elif error_to_rmse_ratio <= 3:

                        overall_reliability = (
                            "The individual prediction error is considerably "
                            "larger than the model's typical RMSE, indicating "
                            "that this observation was relatively difficult "
                            "for the model to predict."
                        )

                    else:

                        overall_reliability = (
                            "The individual prediction error is much larger "
                            "than the model's typical RMSE, indicating that "
                            "this observation was unusually difficult for "
                            "the model to predict."
                        )

                    st.markdown(
                        f"""
                        **Reliability assessment:** {overall_reliability}
                        """
                    )

                st.markdown(
                    f"""
                    **Model agreement:** The three models have a prediction
                    range of **{prediction_range:.2f} percentage points**.
                    {agreement_message}
                    """
                )

                if (
                    absolute_error < 1
                    and prediction_range < 1
                ):

                    final_interpretation = (
                        f"Overall, this is a highly consistent "
                        f"{prediction_system.lower()} prediction. The "
                        "selected model is very close to the observed "
                        "value, and the different models also produce "
                        "similar estimates."
                    )

                elif (
                    absolute_error < 3
                    and prediction_range < 3
                ):

                    final_interpretation = (
                        f"Overall, this is a reasonably consistent "
                        f"{prediction_system.lower()} prediction. The "
                        "selected model has a relatively small error, "
                        "while the different models remain reasonably "
                        "close to one another."
                    )

                elif (
                    absolute_error >= 5
                    or prediction_range >= 5
                ):

                    final_interpretation = (
                        f"Overall, this {prediction_system.lower()} "
                        "prediction should be interpreted with greater "
                        "caution. The individual prediction error or the "
                        "disagreement between models indicates that this "
                        "observation is relatively challenging for the "
                        "current models to estimate accurately."
                    )

                else:

                    final_interpretation = (
                        f"Overall, the {prediction_system.lower()} "
                        "prediction provides a useful estimate, but some "
                        "uncertainty remains. The individual error and "
                        "the level of agreement between models should "
                        "be considered when interpreting the result."
                    )

                st.success(
                    final_interpretation
                )

    # =====================================================
    # FUTURE PROJECTION INTERPRETATION
    # =====================================================

    elif prediction_mode == "Future Projection":

        st.markdown(
            f"""
            This section interprets the projected
            **{prediction_system}** using the selected future
            scenario and the predictions produced by all trained
            models.

            Because the projected year has no recorded outcome yet,
            this section does not calculate prediction error or
            historical reliability.
            """
        )

        if (
            projection_result.empty
            or "projected_value" not in projection_result.columns
            or pd.isna(projected_value)
            or prediction_target is None
            or not target_trained_models
            or baseline_year is None
        ):

            st.warning(
                "Projection interpretation requires a valid "
                "future projection."
            )

        else:

            st.subheader(
                "Projection Interpretation"
            )

            projection_col1, projection_col2, projection_col3 = (
                st.columns(3)
            )

            with projection_col1:

                st.metric(
                    f"Projected {prediction_system}",
                    f"{projected_value:.2f}%",
                )

            if future_projection_mode == "Autonomous Forecast":

                with projection_col2:

                    st.metric(
                        "Projection Year",
                        int(projection_year),
                    )

                with projection_col3:

                    st.metric(
                        "Baseline Year",
                        int(baseline_year),
                    )

            else:

                with projection_col2:

                    st.metric(
                        "Scenario Reference Year",
                        int(baseline_year),
                    )

                with projection_col3:

                    st.metric(
                        "Scenario Type",
                        "Manual",
                    )

            # =================================================
            # Model Agreement
            # =================================================

            st.subheader(
                "Model Agreement"
            )

            numeric_features, categorical_features = (
                get_feature_columns(
                    prediction_target
                )
            )

            feature_columns = (
                numeric_features
                + categorical_features
            )

            future_model_predictions = []

            if future_projection_mode == "Autonomous Forecast":

                future_prediction_data = (
                    projection_feature_data[
                        feature_columns
                    ]
                    .copy()
                )

            else:

                future_prediction_data = (
                    scenario_data[
                        feature_columns
                    ]
                    .copy()
                )

            for model_name, model in (
                target_trained_models.items()
            ):

                try:

                    model_projection = model.predict(
                        future_prediction_data
                    )[0]

                    future_model_predictions.append(
                        {
                            "Model": model_name,
                            (
                                f"Projected {prediction_system} (%)"
                            ): float(
                                model_projection
                            ),
                        }
                    )

                except (
                    ValueError,
                    KeyError,
                    TypeError,
                ) as exc:

                    st.warning(
                        f"{model_name} could not generate a future "
                        f"projection. Details: {exc}"
                    )

            future_model_agreement_df = pd.DataFrame(
                future_model_predictions
            )

            if future_model_agreement_df.empty:

                st.warning(
                    "No trained model was able to generate a future "
                    "projection for the selected scenario."
                )

            else:

                future_value_column = (
                    f"Projected {prediction_system} (%)"
                )

                future_model_agreement_df[
                    future_value_column
                ] = (
                    future_model_agreement_df[
                        future_value_column
                    ].round(2)
                )

                st.dataframe(
                    future_model_agreement_df,
                    use_container_width=True,
                    hide_index=True,
                )

                fig_future_model_agreement = px.bar(
                    future_model_agreement_df,
                    x="Model",
                    y=future_value_column,
                    title=(
                        f"Model Projections for Future "
                        f"{prediction_system}"
                    ),
                    labels={
                        "Model": "Model",
                        future_value_column:
                            f"Projected {prediction_system} (%)",
                    },
                )

                fig_future_model_agreement.update_layout(
                    height=450,
                )

                st.plotly_chart(
                    fig_future_model_agreement,
                    use_container_width=True,
                )

                projection_predictions = (
                    future_model_agreement_df[
                        future_value_column
                    ]
                )

                projection_min = (
                    projection_predictions.min()
                )

                projection_max = (
                    projection_predictions.max()
                )

                prediction_range = (
                    projection_max
                    - projection_min
                )

                projection_range = prediction_range

                projection_mean = (
                    projection_predictions.mean()
                )

                if projection_range < 1:

                    agreement_level = (
                        "Strong model agreement"
                    )

                    agreement_message = (
                        "The models produce very similar projections "
                        f"for the future {prediction_system.lower()}."
                    )

                elif projection_range < 3:

                    agreement_level = (
                        "Moderate model agreement"
                    )

                    agreement_message = (
                        "The models show some variation in their future "
                        f"{prediction_system.lower()} projections, but "
                        "their estimates remain reasonably close."
                    )

                elif projection_range < 5:

                    agreement_level = (
                        "Noticeable model disagreement"
                    )

                    agreement_message = (
                        "The models produce noticeably different future "
                        f"{prediction_system.lower()} projections. The "
                        "scenario should therefore be interpreted with "
                        "some caution."
                    )

                else:

                    agreement_level = (
                        "Strong model disagreement"
                    )

                    agreement_message = (
                        "The models produce substantially different future "
                        f"{prediction_system.lower()} projections. This "
                        "indicates greater uncertainty around the scenario."
                    )

                projection_agreement_col1, projection_agreement_col2 = (
                    st.columns(2)
                )

                with projection_agreement_col1:

                    st.metric(
                        "Model Agreement",
                        agreement_level,
                    )

                with projection_agreement_col2:

                    st.metric(
                        "Projection Range",
                        f"{projection_range:.2f} percentage points",
                    )

                st.info(
                    agreement_message
                )

                # =================================================
                # Scenario Interpretation
                # =================================================

                st.subheader(
                    "Scenario Interpretation"
                )

                if future_projection_mode == "Autonomous Forecast":

                    st.markdown(
                        f"""
                        Under the selected scenario, the models project an
                        average {prediction_system.lower()} of approximately
                        **{projection_mean:.2f}%** for
                        **{projection_country}**
                        and **{projection_disease}**
                        in **{int(projection_year)}**.

                        Across the three models, the projected
                        {prediction_system.lower()} ranges from
                        **{projection_min:.2f}%**
                        to **{projection_max:.2f}%**,
                        giving a model projection range of
                        **{projection_range:.2f} percentage points**.
                        """
                    )

                else:

                    st.markdown(
                        f"""
                        Under the selected manual scenario, the models project an
                        average {prediction_system.lower()} of approximately
                        **{projection_mean:.2f}%** for
                        **{projection_country}**
                        and **{projection_disease}**.

                        The scenario uses the latest available historical
                        observation from **{int(baseline_year)}** as its
                        reference point.

                        Across the models, the projected
                        {prediction_system.lower()} ranges from
                        **{projection_min:.2f}%**
                        to **{projection_max:.2f}%**,
                        giving a model projection range of
                        **{projection_range:.2f} percentage points**.
                        """
                    )

                # =================================================
                # Scenario Assumptions
                # =================================================

                if future_projection_mode == "Autonomous Forecast":

                    st.subheader(
                        "Forecasted Predictor Values"
                    )

                    scenario_interpretation = (
                        projection_feature_data
                        .copy()
                    )

                else:

                    st.subheader(
                        "Scenario Assumptions"
                    )

                    scenario_interpretation = (
                        scenario_display
                        .copy()
                    )

                st.dataframe(
                    scenario_interpretation,
                    use_container_width=True,
                    hide_index=True,
                )

                if future_projection_mode == "Autonomous Forecast":

                    st.caption(
                        f"""
                        The autonomous forecast uses historical
                        **{projection_country} — {projection_disease}**
                        observations through **{int(baseline_year)}**.

                        Predictor variables are forecast independently to
                        **{int(projection_year)}** and then supplied to the
                        selected machine learning model.
                        """
                    )

                else:

                    st.caption(
                        f"""
                        The manual scenario starts from the latest available
                        **{int(baseline_year)}**
                        country-disease observation and uses the predictor
                        assumptions supplied directly by the user.
                        """
                    )

                # =================================================
                # Overall Projection Interpretation
                # =================================================

                st.subheader(
                    "Overall Projection Interpretation"
                )

                if projection_range < 1:

                    projection_conclusion = (
                        f"The future {prediction_system.lower()} projection "
                        "is relatively stable across the three models. "
                        "Because the models produce closely aligned estimates, "
                        "the projected value is reasonably consistent under "
                        "the selected scenario assumptions."
                    )

                elif projection_range < 3:

                    projection_conclusion = (
                        f"The future {prediction_system.lower()} projection "
                        "shows moderate consistency across the models. "
                        "Some variation exists, but the estimates remain "
                        "reasonably close under the selected scenario."
                    )

                elif projection_range < 5:

                    projection_conclusion = (
                        f"The future {prediction_system.lower()} projection "
                        "shows noticeable variation across the models. "
                        "The scenario should therefore be interpreted with "
                        "some caution."
                    )

                else:

                    projection_conclusion = (
                        f"The future {prediction_system.lower()} projection "
                        "shows substantial variation across the models. "
                        "This indicates considerable model uncertainty, so "
                        "the projected value should not be treated as a "
                        "precise future forecast."
                    )

                st.success(
                    projection_conclusion
                )

                st.warning(
                    """
                    Future projections are scenario-based estimates rather
                    than validated forecasts. Changes in the assumed
                    predictor values can materially change the projected
                    outcome.
                    """
                )


# =========================================================
# TAB 6 — FEATURE IMPORTANCE
# =========================================================

with tab_importance:

    st.header(
        "📌 Phase 6 — Feature Importance"
    )

    if (
        prediction_system is None
        or prediction_target is None
        or not target_trained_models
    ):

        st.info(
            "Select a prediction system in the Prediction Interface "
            "tab to calculate feature importance."
        )

    else:

        st.markdown(
            f"""
            This section examines which health, socioeconomic,
            geographic, and temporal characteristics contribute most
            to the **{prediction_system}** model's performance.

            Feature importance is estimated using permutation importance
            on the unseen historical test set. A feature is considered
            influential when randomly shuffling its values causes a
            meaningful deterioration in model performance.

            Feature importance measures model dependence on the
            available predictors; it does not establish causation.
            """
        )

        st.subheader(
            "Feature Importance Analysis"
        )

        importance_col1, importance_col2 = (
            st.columns(2)
        )

        model_options = list(
            target_trained_models.keys()
        )

        if prediction_mode == "Historical Prediction":

            default_model = prediction_model_name

        elif future_projection_mode == "Autonomous Forecast":

            default_model = projection_model_name

        else:

            default_model = manual_model_name

        if default_model in model_options:

            default_model_index = (
                model_options.index(
                    default_model
                )
            )

        else:

            default_model_index = 0

        with importance_col1:

            importance_model_name = st.selectbox(
                "Select model",
                model_options,
                index=default_model_index,
                key="feature_importance_model",
            )

        with importance_col2:

            importance_metric = st.selectbox(
                "Importance metric",
                [
                    "RMSE",
                    "MAE",
                    "R²",
                ],
                key="feature_importance_metric",
            )

        target_test_df = target_ml_data[
            prediction_target
        ]["test"].copy()

        numeric_features, categorical_features = (
            get_feature_columns(
                prediction_target
            )
        )

        feature_columns = (
            numeric_features
            + categorical_features
        )

        X_test_importance = (
            target_test_df[
                feature_columns
            ]
            .copy()
        )

        y_test_importance = (
            target_test_df[
                prediction_target
            ]
            .copy()
        )

        selected_importance_model = (
            target_trained_models[
                importance_model_name
            ]
        )

        if importance_metric == "RMSE":

            scoring_metric = (
                "neg_root_mean_squared_error"
            )

        elif importance_metric == "MAE":

            scoring_metric = (
                "neg_mean_absolute_error"
            )

        else:

            scoring_metric = "r2"

        if X_test_importance.empty:

            st.warning(
                "Feature importance cannot be calculated because "
                "the historical test dataset is empty."
            )

        elif y_test_importance.empty:

            st.warning(
                "Feature importance cannot be calculated because "
                "the test target data is empty."
            )

        elif not set(feature_columns).issubset(
            X_test_importance.columns
        ):

            st.error(
                "Feature importance could not be calculated because "
                "one or more required predictor columns are missing."
            )

        else:

            try:

                permutation_result = (
                    calculate_permutation_importance(
                        selected_importance_model,
                        X_test_importance,
                        y_test_importance,
                        scoring_metric,
                        importance_model_name,
                    )
                )

            except (
                ValueError,
                KeyError,
                TypeError,
                AttributeError,
            ) as exc:

                st.error(
                    "Feature importance could not be calculated. "
                    f"Details: {exc}"
                )

                permutation_result = None

            if permutation_result is not None:

                feature_importance_df = pd.DataFrame(
                    {
                        "Feature": feature_columns,
                        "Importance": (
                            permutation_result.importances_mean
                        ),
                        "Importance Std": (
                            permutation_result.importances_std
                        ),
                    }
                )

                feature_importance_df[
                    "Absolute Importance"
                ] = (
                    feature_importance_df[
                        "Importance"
                    ].abs()
                )

                feature_importance_df = (
                    feature_importance_df
                    .sort_values(
                        "Absolute Importance",
                        ascending=False,
                    )
                    .reset_index(
                        drop=True
                    )
                )

                st.subheader(
                    "Feature Importance Ranking"
                )

                st.caption(
                    f"""
                    Importance is calculated for the
                    **{target_display_name}** target using the
                    **{importance_model_name}** model and the
                    **{importance_metric}** scoring metric.
                    """
                )

                feature_importance_display = (
                    feature_importance_df[
                        [
                            "Feature",
                            "Importance",
                            "Importance Std",
                        ]
                    ]
                    .copy()
                    .rename(
                        columns={
                            "Feature":
                                "Feature",

                            "Importance":
                                "Mean Importance",

                            "Importance Std":
                                "Importance Std.",
                        }
                    )
                )

                st.dataframe(
                    feature_importance_display.style.format(
                        {
                            "Mean Importance": "{:.4f}",
                            "Importance Std.": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader(
                    "Feature Importance Chart"
                )

                top_features = (
                    feature_importance_df
                    .head(10)
                    .sort_values(
                        "Importance",
                        ascending=True,
                    )
                )

                fig_feature_importance = px.bar(
                    top_features,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title=(
                        f"Top 10 Features — "
                        f"{target_display_name} — "
                        f"{importance_model_name}"
                    ),
                    labels={
                        "Importance": (
                            f"Permutation Importance ({importance_metric})"
                        ),
                        "Feature": "Feature",
                    },
                )

                fig_feature_importance.update_layout(
                    height=500,
                )

                st.plotly_chart(
                    fig_feature_importance,
                    use_container_width=True,
                )

                st.subheader(
                    "Feature Importance Interpretation"
                )

                if not feature_importance_df.empty:

                    top_feature_row = (
                        feature_importance_df
                        .iloc[0]
                    )

                    top_feature = (
                        top_feature_row[
                            "Feature"
                        ]
                    )

                    top_importance = float(
                        top_feature_row[
                            "Importance"
                        ]
                    )

                    positive_importance_count = int(
                        (
                            feature_importance_df[
                                "Importance"
                            ]
                            > 0
                        )
                        .sum()
                    )

                    negative_importance_count = int(
                        (
                            feature_importance_df[
                                "Importance"
                            ]
                            < 0
                        )
                        .sum()
                    )

                    if top_importance > 0:

                        importance_message = (
                            f"For the **{importance_model_name}** model, "
                            f"**{top_feature}** has the largest permutation "
                            f"importance by absolute magnitude for predicting "
                            f"the **{target_display_name}** target. Its mean "
                            f"permutation importance is **{top_importance:.4f}**, "
                            f"indicating that shuffling this feature causes a "
                            f"measurable deterioration in model performance."
                        )

                    elif top_importance < 0:

                        importance_message = (
                            f"For the **{importance_model_name}** model, "
                            f"**{top_feature}** has the largest permutation "
                            f"importance by absolute magnitude, but its mean "
                            f"importance is negative (**{top_importance:.4f}**). "
                            f"This means that shuffling the feature did not "
                            f"deteriorate test performance and may have slightly "
                            f"improved it under this evaluation."
                        )

                    else:

                        importance_message = (
                            f"For the **{importance_model_name}** model, "
                            f"**{top_feature}** has the largest permutation "
                            f"importance by absolute magnitude, but its mean "
                            f"importance is approximately zero. This indicates "
                            f"little measurable dependence on this feature under "
                            f"the current test-set evaluation."
                        )

                    st.info(
                        importance_message
                    )

                    st.markdown(
                        f"""
                        Among the **{len(feature_importance_df)}**
                        predictors evaluated,
                        **{positive_importance_count}** produced positive
                        permutation importance values, while
                        **{negative_importance_count}** produced negative
                        values.

                        Positive permutation importance means that shuffling a
                        feature worsened model performance. Negative importance
                        means that shuffling the feature improved performance
                        slightly under the test-set evaluation.

                        These results describe how the selected model uses the
                        available predictors. They should not be interpreted as
                        evidence that a feature causes changes in
                        **{prediction_system.lower()}**.
                        """
                    )

                st.subheader(
                    "Feature Importance Context"
                )

                st.markdown(
                    f"""
                    **Prediction System:** {target_display_name}

                    **Model:** {importance_model_name}

                    **Evaluation Metric:** {importance_metric}

                    **Evaluation Dataset:** Historical test set
                    ({int(target_test_df["year"].min())}–{int(target_test_df["year"].max())})

                    **Number of Predictors:** {len(feature_columns)}

                    Feature importance is evaluated on the historical test data
                    because the purpose is to understand how the trained model
                    behaves when estimating unseen observations. Future scenario
                    assumptions are therefore not used to calculate this metric.
                    """
                )


# =========================================================
# TAB 7 — COUNTRY CLUSTERING
# =========================================================

with tab_clustering:

    st.header(
        "🌍 Phase 7 — Country Clustering"
    )

    st.markdown(
        """
        This section groups countries according to similarities in
        their health and socioeconomic characteristics.

        Countries are clustered using standardized health, healthcare,
        economic, education, and urbanization indicators. Countries
        within the same cluster therefore share broadly similar
        characteristics across the selected variables.
        """
    )

    clustering_features = [
        "healthcare_access_pct",
        "doctors_per_1000",
        "hospital_beds_per_1000",
        "recovery_rate_pct",
        "per_capita_income_usd",
        "education_index",
        "urbanization_rate_pct",
    ]

    country_cluster_df = (
        ml_df[
            [
                "country",
                *clustering_features,
            ]
        ]
        .groupby("country", as_index=False)
        .mean(numeric_only=True)
    )

    country_cluster_df = country_cluster_df.dropna(
        subset=clustering_features
    ).reset_index(
        drop=True
    )

    if len(country_cluster_df) < 3:

        st.warning(
            "There are not enough countries with complete data "
            "to perform country clustering."
        )

    else:

        st.subheader(
            "Clustering Controls"
        )

        cluster_col1, cluster_col2 = (
            st.columns(2)
        )

        with cluster_col1:

            n_clusters = st.slider(
                "Number of clusters",
                min_value=2,
                max_value=min(
                    8,
                    len(country_cluster_df) - 1,
                ),
                value=3,
                step=1,
                key="country_cluster_count",
            )

        with cluster_col2:

            st.metric(
                "Countries included",
                len(country_cluster_df),
            )

        try:

            scaler = StandardScaler()

            X_cluster = scaler.fit_transform(
                country_cluster_df[
                    clustering_features
                ]
            )

        except (
            ValueError,
            KeyError,
            TypeError,
        ) as exc:

            st.error(
                "Country clustering could not be prepared because "
                "the clustering data is invalid. "
                f"Details: {exc}"
            )

            X_cluster = None

        if X_cluster is not None:

            try:

                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10,
                )

                country_cluster_df[
                    "Cluster"
                ] = kmeans.fit_predict(
                    X_cluster
                )

            except (
                ValueError,
                TypeError,
            ) as exc:

                st.error(
                    "The K-Means clustering model could not be fitted. "
                    f"Details: {exc}"
                )

                country_cluster_df = pd.DataFrame()

            if not country_cluster_df.empty:

                country_cluster_df[
                    "Cluster"
                ] = (
                    country_cluster_df[
                        "Cluster"
                    ]
                    + 1
                )

                st.subheader(
                    "Cluster Summary"
                )

                cluster_summary = (
                    country_cluster_df
                    .groupby("Cluster")
                    .agg(
                        Countries=(
                            "country",
                            "count",
                        ),
                        Healthcare_Access=(
                            "healthcare_access_pct",
                            "mean",
                        ),
                        Doctors_per_1000=(
                            "doctors_per_1000",
                            "mean",
                        ),
                        Hospital_Beds_per_1000=(
                            "hospital_beds_per_1000",
                            "mean",
                        ),
                        Recovery_Rate=(
                            "recovery_rate_pct",
                            "mean",
                        ),
                        Per_Capita_Income=(
                            "per_capita_income_usd",
                            "mean",
                        ),
                        Education_Index=(
                            "education_index",
                            "mean",
                        ),
                        Urbanization_Rate=(
                            "urbanization_rate_pct",
                            "mean",
                        ),
                    )
                    .reset_index()
                )

                cluster_summary = cluster_summary.rename(
                    columns={
                        "Healthcare_Access":
                            "Healthcare Access (%)",

                        "Doctors_per_1000":
                            "Doctors per 1,000",

                        "Hospital_Beds_per_1000":
                            "Hospital Beds per 1,000",

                        "Recovery_Rate":
                            "Recovery Rate (%)",

                        "Per_Capita_Income":
                            "Per Capita Income (USD)",

                        "Education_Index":
                            "Education Index",

                        "Urbanization_Rate":
                            "Urbanization Rate (%)",
                    }
                )

                numeric_summary_columns = [
                    column
                    for column in cluster_summary.columns
                    if column != "Cluster"
                    and column != "Countries"
                ]

                cluster_summary[
                    numeric_summary_columns
                ] = (
                    cluster_summary[
                        numeric_summary_columns
                    ]
                    .round(2)
                )

                st.dataframe(
                    cluster_summary,
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader(
                    "Countries by Cluster"
                )

                selected_cluster = st.selectbox(
                    "Select cluster",
                    sorted(
                        country_cluster_df[
                            "Cluster"
                        ]
                        .unique()
                        .tolist()
                    ),
                    key="selected_country_cluster",
                )

                countries_in_cluster = (
                    country_cluster_df[
                        country_cluster_df[
                            "Cluster"
                        ]
                        == selected_cluster
                    ]
                    [
                        [
                            "country",
                            "Cluster",
                            *clustering_features,
                        ]
                    ]
                    .copy()
                )

                countries_in_cluster = (
                    countries_in_cluster
                    .rename(
                        columns={
                            "country": "Country",

                            "Cluster": "Cluster",

                            "healthcare_access_pct":
                                "Healthcare Access (%)",

                            "doctors_per_1000":
                                "Doctors per 1,000",

                            "hospital_beds_per_1000":
                                "Hospital Beds per 1,000",

                            "recovery_rate_pct":
                                "Recovery Rate (%)",

                            "per_capita_income_usd":
                                "Per Capita Income (USD)",

                            "education_index":
                                "Education Index",

                            "urbanization_rate_pct":
                                "Urbanization Rate (%)",
                        }
                    )
                )

                countries_in_cluster[
                    countries_in_cluster.columns[
                        2:
                    ]
                ] = (
                    countries_in_cluster[
                        countries_in_cluster.columns[
                            2:
                        ]
                    ]
                    .round(2)
                )

                st.dataframe(
                    countries_in_cluster,
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader(
                    "Cluster Visualization"
                )

                try:

                    pca = PCA(
                        n_components=2,
                        random_state=42,
                    )

                    cluster_coordinates = pca.fit_transform(
                        X_cluster
                    )

                    country_cluster_df[
                        "PC1"
                    ] = cluster_coordinates[:, 0]

                    country_cluster_df[
                        "PC2"
                    ] = cluster_coordinates[:, 1]

                    fig_clusters = px.scatter(
                        country_cluster_df,
                        x="PC1",
                        y="PC2",
                        color="Cluster",
                        hover_name="country",
                        title=(
                            "Countries Grouped by Health and "
                            "Socioeconomic Characteristics"
                        ),
                        labels={
                            "PC1": "Principal Component 1",
                            "PC2": "Principal Component 2",
                            "Cluster": "Cluster",
                        },
                    )

                    fig_clusters.update_layout(
                        height=550,
                    )

                    st.plotly_chart(
                        fig_clusters,
                        use_container_width=True,
                    )

                except (
                    ValueError,
                    TypeError,
                ) as exc:

                    st.warning(
                        "The cluster visualization could not be generated. "
                        f"Details: {exc}"
                    )

                st.subheader(
                    "Cluster Interpretation"
                )

                overall_means = (
                    country_cluster_df[
                        clustering_features
                    ]
                    .mean()
                )

                cluster_means = (
                    country_cluster_df
                    .groupby("Cluster")[
                        clustering_features
                    ]
                    .mean()
                )

                for cluster_number in sorted(
                    cluster_means.index
                ):

                    cluster_countries = (
                        country_cluster_df[
                            country_cluster_df[
                                "Cluster"
                            ]
                            == cluster_number
                        ][
                            "country"
                        ]
                        .tolist()
                    )

                    cluster_values = cluster_means.loc[
                        cluster_number
                    ]

                    higher_features = []
                    lower_features = []

                    for feature in clustering_features:

                        cluster_value = (
                            cluster_values[
                                feature
                            ]
                        )

                        overall_value = (
                            overall_means[
                                feature
                            ]
                        )

                        if overall_value == 0:
                            continue

                        percentage_difference = (
                            (
                                cluster_value
                                - overall_value
                            )
                            / abs(overall_value)
                        ) * 100

                        if percentage_difference >= 10:

                            higher_features.append(
                                feature
                            )

                        elif percentage_difference <= -10:

                            lower_features.append(
                                feature
                            )

                    feature_labels = {

                        "healthcare_access_pct":
                            "healthcare access",

                        "doctors_per_1000":
                            "doctor availability",

                        "hospital_beds_per_1000":
                            "hospital bed availability",

                        "recovery_rate_pct":
                            "recovery rates",

                        "per_capita_income_usd":
                            "per-capita income",

                        "education_index":
                            "education levels",

                        "urbanization_rate_pct":
                            "urbanization",
                    }

                    higher_text = [
                        feature_labels[feature]
                        for feature in higher_features
                    ]

                    lower_text = [
                        feature_labels[feature]
                        for feature in lower_features
                    ]

                    if higher_text:

                        higher_description = (
                            ", ".join(
                                higher_text
                            )
                        )

                    else:

                        higher_description = (
                            "no major indicators above "
                            "the overall country average"
                        )

                    if lower_text:

                        lower_description = (
                            ", ".join(
                                lower_text
                            )
                        )

                    else:

                        lower_description = (
                            "no major indicators below "
                            "the overall country average"
                        )

                    country_count = len(
                        cluster_countries
                    )

                    st.markdown(
                        f"""
                        **Cluster {cluster_number}**

                        This cluster contains **{country_count}**
                        countries.

                        Countries in this cluster tend to show
                        **{higher_description}** compared with the
                        overall country-level averages, while showing
                        **{lower_description}** on other indicators.

                        **Countries:** {", ".join(cluster_countries)}
                        """
                    )


# =========================================================
# TAB 8 — MACHINE LEARNING EXPORTS
# =========================================================

with tab_exports:

    st.header(
        "📥 Machine Learning Exports"
    )

    st.markdown(
        """
        Export the machine learning analysis results generated on
        this page, including model evaluation, predictions,
        feature importance, and country clustering.
        """
    )

    # -----------------------------------------------------
    # Prepare Model Evaluation export
    # -----------------------------------------------------

    model_evaluation_export = (
        results_df.copy()
    )

    # -----------------------------------------------------
    # Prepare Prediction export
    # -----------------------------------------------------

    if prediction_mode == "Historical Prediction":

        prediction_export = (
            prediction_result.copy()
            if isinstance(
                prediction_result,
                pd.DataFrame,
            )
            else pd.DataFrame()
        )

    else:

        prediction_export = (
            projection_result.copy()
            if isinstance(
                projection_result,
                pd.DataFrame,
            )
            else pd.DataFrame()
        )

    # -----------------------------------------------------
    # Prepare Feature Importance export
    # -----------------------------------------------------

    feature_importance_export = (
        feature_importance_df.copy()
        if isinstance(
            feature_importance_df,
            pd.DataFrame,
        )
        else pd.DataFrame()
    )

    # -----------------------------------------------------
    # Prepare Country Clustering export
    # -----------------------------------------------------

    country_clustering_export = (
        country_cluster_df.copy()
        if isinstance(
            country_cluster_df,
            pd.DataFrame,
        )
        else pd.DataFrame()
    )

    cluster_summary_export = (
        cluster_summary.copy()
        if isinstance(
            cluster_summary,
            pd.DataFrame,
        )
        else pd.DataFrame()
    )

    # =====================================================
    # MODEL EVALUATION EXPORT
    # =====================================================

    st.subheader(
        "Model Evaluation"
    )

    model_evaluation_csv = (
        model_evaluation_export
        .to_csv(index=False)
        .encode("utf-8")
    )

    model_eval_col1, model_eval_col2 = (
        st.columns(2)
    )

    with model_eval_col1:

        st.download_button(
            label="⬇️ Download Model Evaluation (CSV)",
            data=model_evaluation_csv,
            file_name=(
                f"global_health_model_evaluation_"
                f"{make_export_name(prediction_system or 'all')}.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    with model_eval_col2:

        model_eval_buffer = io.BytesIO()

        try:

            with pd.ExcelWriter(
                model_eval_buffer,
                engine="openpyxl",
            ) as writer:

                model_evaluation_export.to_excel(
                    writer,
                    index=False,
                    sheet_name="Model Evaluation",
                )

        except (
            OSError,
            ValueError,
            ImportError,
        ) as exc:

            st.warning(
                "Model Evaluation Excel export could not be created. "
                f"Details: {exc}"
            )

            model_eval_buffer = None

        if model_eval_buffer is not None:

            st.download_button(
                label="📊 Download Model Evaluation (Excel)",
                data=model_eval_buffer.getvalue(),
                file_name=(
                    f"global_health_model_evaluation_"
                    f"{make_export_name(prediction_system or 'all')}.xlsx"
                ),
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
            )

    # =====================================================
    # PREDICTION EXPORT
    # =====================================================

    st.subheader(
        "Predictions"
    )

    if prediction_export.empty:

        st.info(
            "No prediction is currently available for export."
        )

    else:

        prediction_csv = (
            prediction_export
            .to_csv(index=False)
            .encode("utf-8")
        )

        prediction_export_target = make_export_name(
            prediction_system
        )

        if prediction_mode == "Historical Prediction":

            prediction_export_country = make_export_name(
                prediction_country or "unknown_country"
            )

            prediction_export_disease = make_export_name(
                prediction_disease or "unknown_disease"
            )

            prediction_export_mode = "historical"

        else:

            prediction_export_country = make_export_name(
                projection_country or "unknown_country"
            )

            prediction_export_disease = make_export_name(
                projection_disease or "unknown_disease"
            )

            if (
                future_projection_mode
                == "Autonomous Forecast"
            ):

                prediction_export_mode = (
                    "future_projection"
                )

            else:

                prediction_export_mode = (
                    "manual_scenario"
                )

        prediction_export_filename = (
            f"global_health_predictions_"
            f"{prediction_export_target}_"
            f"{prediction_export_country}_"
            f"{prediction_export_disease}_"
            f"{prediction_export_mode}"
        )

        prediction_col1, prediction_col2 = (
            st.columns(2)
        )

        with prediction_col1:

            st.download_button(
                label="⬇️ Download Predictions (CSV)",
                data=prediction_csv,
                file_name=(
                    f"{prediction_export_filename}.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        with prediction_col2:

            prediction_buffer = io.BytesIO()

            try:

                with pd.ExcelWriter(
                    prediction_buffer,
                    engine="openpyxl",
                ) as writer:

                    prediction_export.to_excel(
                        writer,
                        index=False,
                        sheet_name="Predictions",
                    )

                st.download_button(
                    label="📊 Download Predictions (Excel)",
                    data=prediction_buffer.getvalue(),
                    file_name=(
                        f"{prediction_export_filename}.xlsx"
                    ),
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    use_container_width=True,
                )

            except (
                OSError,
                ValueError,
                ImportError,
            ) as exc:

                st.warning(
                    "Predictions Excel export could not be created. "
                    f"Details: {exc}"
                )

    # =====================================================
    # FEATURE IMPORTANCE EXPORT
    # =====================================================

    st.subheader(
        "Feature Importance"
    )

    feature_importance_csv = (
        feature_importance_export
        .to_csv(index=False)
        .encode("utf-8")
    )

    feature_col1, feature_col2 = (
        st.columns(2)
    )

    with feature_col1:

        st.download_button(
            label="⬇️ Download Feature Importance (CSV)",
            data=feature_importance_csv,
            file_name=(
                f"global_health_feature_importance_"
                f"{make_export_name(prediction_system or 'all')}.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    with feature_col2:

        feature_buffer = io.BytesIO()

        try:

            with pd.ExcelWriter(
                feature_buffer,
                engine="openpyxl",
            ) as writer:

                feature_importance_export.to_excel(
                    writer,
                    index=False,
                    sheet_name="Feature Importance",
                )

            st.download_button(
                label="📊 Download Feature Importance (Excel)",
                data=feature_buffer.getvalue(),
                file_name=(
                    f"global_health_feature_importance_"
                    f"{make_export_name(prediction_system or 'all')}.xlsx"
                ),
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
            )

        except (
            OSError,
            ValueError,
            ImportError,
        ) as exc:

            st.warning(
                "Feature Importance Excel export could not be created. "
                f"Details: {exc}"
            )

    # =====================================================
    # COUNTRY CLUSTERING EXPORT
    # =====================================================

    st.subheader(
        "Country Clustering"
    )

    clustering_buffer = io.BytesIO()

    try:

        with pd.ExcelWriter(
            clustering_buffer,
            engine="openpyxl",
        ) as writer:

            country_clustering_export.to_excel(
                writer,
                index=False,
                sheet_name="Country Clusters",
            )

            if not cluster_summary_export.empty:

                cluster_summary_export.to_excel(
                    writer,
                    index=False,
                    sheet_name="Cluster Summary",
                )

        clustering_excel_data = (
            clustering_buffer.getvalue()
        )

    except (
        OSError,
        ValueError,
        ImportError,
    ) as exc:

        st.warning(
            "Country Clustering Excel export could not be created. "
            f"Details: {exc}"
        )

        clustering_excel_data = None

    clustering_csv = (
        country_clustering_export
        .to_csv(index=False)
        .encode("utf-8")
    )

    cluster_col1, cluster_col2 = (
        st.columns(2)
    )

    with cluster_col1:

        st.download_button(
            label="⬇️ Download Country Clustering (CSV)",
            data=clustering_csv,
            file_name="global_health_country_clustering.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with cluster_col2:

        if clustering_excel_data is not None:

            st.download_button(
                label="📊 Download Country Clustering (Excel)",
                data=clustering_excel_data,
                file_name="global_health_country_clustering.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
            )