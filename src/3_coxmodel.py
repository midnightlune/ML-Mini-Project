import pandas as pd
from pathlib import Path

from lifelines import CoxPHFitter

INPUT_PATH = Path(
    "data/processed/clinical_survival_clean.csv"
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(INPUT_PATH)

# SELECT VARIABLES
model_df = df[
    [
        "survival_time_days",
        "survival_event",
        "Age_years",
        "Sex",
        "Tumor Grade",
    ]
].copy()

# CLEAN CATEGORICAL VARIABLES
model_df = pd.get_dummies(
    model_df,
    columns=["Sex", "Tumor Grade"],
    drop_first=True,
    dtype=int
)

# REMOVE MISSING VALUES
model_df = model_df.dropna()

print("Patients used in Cox model:")
print(len(model_df))

print("\nModel columns:")
print(model_df.columns.tolist())

# BUILD COX MODEL
cph = CoxPHFitter()

cph.fit(
    model_df,
    duration_col="survival_time_days",
    event_col="survival_event"
)

# DISPLAY RESULTS
print("\nCox Proportional Hazards Model")
print("=" * 60)

cph.print_summary()

# SAVE RESULTS
summary = cph.summary

summary.to_csv(
    RESULTS_DIR / "cox_model_results.csv"
)

print("\nC-index:")
print(cph.concordance_index_)

print("\nResults saved to:")
print(
    RESULTS_DIR / "cox_model_results.csv"
)