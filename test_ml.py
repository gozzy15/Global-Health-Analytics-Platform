import pandas as pd

from global_health_web_app.utils.data_loader import load_health_data
from global_health_web_app.utils.ml_models import (
    prepare_ml_dataset,
    time_based_split,
    train_models,
    evaluate_models,
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = load_health_data()

print(
    f"Original dataset shape: {df.shape}"
)


# ---------------------------------------------------------
# Prepare ML dataset
# ---------------------------------------------------------

ml_df = prepare_ml_dataset(
    df
)

print(
    f"ML dataset shape: {ml_df.shape}"
)


# ---------------------------------------------------------
# Display available years
# ---------------------------------------------------------

print(
    "\nAvailable years:"
)

print(
    sorted(
        ml_df["year"]
        .dropna()
        .unique()
    )
)


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------

train_df, test_df = time_based_split(
    ml_df,
    test_years=5,
)


print(
    "\nTraining period:"
)

print(
    f"{int(train_df['year'].min())}"
    f" - "
    f"{int(train_df['year'].max())}"
)


print(
    "\nTesting period:"
)

print(
    f"{int(test_df['year'].min())}"
    f" - "
    f"{int(test_df['year'].max())}"
)


print(
    f"\nTraining observations: {len(train_df):,}"
)

print(
    f"Testing observations: {len(test_df):,}"
)


# ---------------------------------------------------------
# Train models
# ---------------------------------------------------------

trained_models = train_models(
    train_df
)


print(
    "\nModels trained:"
)

for model_name in trained_models:

    print(
        f"✓ {model_name}"
    )


# ---------------------------------------------------------
# Evaluate models
# ---------------------------------------------------------

results = evaluate_models(
    trained_models,
    test_df,
)


print(
    "\nModel Performance:"
)

print(
    results.to_string(
        index=False
    )
)