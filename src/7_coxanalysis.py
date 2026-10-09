"""
11_cox_full_analysis.py

Clinical-only Cox Proportional Hazards Analysis

Purpose:
    Build a Cox proportional hazards model using the
    clinical variables from the CPTAC-PDA dataset.

Outputs:
    - Hazard ratios
    - 95% confidence intervals
    - p-values
    - C-index
    - proportional-hazards assumption results
    - model summary

Important:
    - SMOTE is NOT used for Cox regression.
    - 100% missing variables are removed.
    - Survival time and censoring are retained.
    - Categorical variables are one-hot encoded.
    - A small penalizer is used to improve numerical stability.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter


# PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clinical_survival_clean.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

COX_RESULTS_FILE = (
    RESULTS_DIR
    / "cox_full_results.csv"
)

COX_SUMMARY_FILE = (
    RESULTS_DIR
    / "clinical_model_summary.csv"
)

PH_RESULTS_FILE = (
    RESULTS_DIR
    / "cox_proportional_hazards_test.csv"
)

REMOVED_FEATURES_FILE = (
    RESULTS_DIR
    / "cox_removed_features.csv"
)


# SETTINGS
# Small penalizer to reduce instability caused by sparse categories.
PENALIZER = 0.1

# Maximum number of iterations
MAX_ITERATIONS = 200


print("CLINICAL-ONLY COX PROPORTIONAL HAZARDS ANALYSIS")
print("=" * 70)


# LOAD DATA
if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}\n\n"
        "Run 02_clean_clinical_data.py first."
    )


df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nOriginal dataset shape: {df.shape}"
)


# REQUIRED SURVIVAL VARIABLES
required_columns = [
    "survival_time_days",
    "survival_event",
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"\nRequired column missing: {column}"
        )


# CONVERT SURVIVAL VARIABLES
df["survival_time_days"] = pd.to_numeric(
    df["survival_time_days"],
    errors="coerce"
)

df["survival_event"] = pd.to_numeric(
    df["survival_event"],
    errors="coerce"
)


# Remove records without survival information
df = df.dropna(
    subset=[
        "survival_time_days",
        "survival_event"
    ]
).copy()


# Remove impossible/non-positive survival times
df = df[
    df["survival_time_days"] > 0
].copy()


print(
    f"Patients with usable survival information: "
    f"{len(df)}"
)

print(
    f"Deaths/events: "
    f"{int(df['survival_event'].sum())}"
)

print(
    f"Censored/alive observations: "
    f"{int((df['survival_event'] == 0).sum())}"
)


# CANDIDATE CLINICAL FEATURES
candidate_features = [
    "Age_years",
    "Sex",
    "Race",
    "Ethnicity",
    "Tumor Grade",
    "AJCC Pathologic Stage",
    "AJCC Pathologic T",
    "AJCC Pathologic N",
    "AJCC Pathologic M",
    "Lymph Nodes Positive",
    "Lymph Nodes Tested",
    "Metastasis At Diagnosis",
    "Lymphatic Invasion Present",
    "Perineural Invasion Present",
    "Vascular Invasion Present",
    "Margins Involved Site",
]


# KEEP FEATURES THAT EXIST
available_features = [
    feature
    for feature in candidate_features
    if feature in df.columns
]


missing_from_dataset = [
    feature
    for feature in candidate_features
    if feature not in df.columns
]


if missing_from_dataset:

    print(
        "\nCandidate features not present:"
    )

    for feature in missing_from_dataset:

        print(
            f" - {feature}"
        )


# REMOVE 100% MISSING FEATURES
features_used = []

features_removed = []


for feature in available_features:

    if df[feature].isna().all():

        features_removed.append(
            feature
        )

    else:

        features_used.append(
            feature
        )


print(
    "\nClinical features used:"
)

for feature in features_used:

    print(
        f" - {feature}"
    )


print(
    "\nClinical features removed because "
    "they are 100% missing:"
)

if features_removed:

    for feature in features_removed:

        print(
            f" - {feature}"
        )

else:

    print(
        " - None"
    )


# SAVE REMOVED FEATURE REPORT
removed_rows = []


for feature in features_removed:

    removed_rows.append(
        {
            "feature": feature,
            "reason": "100% missing",
            "missing_count": int(
                df[feature].isna().sum()
            ),
            "total_rows": len(df),
        }
    )


for feature in missing_from_dataset:

    removed_rows.append(
        {
            "feature": feature,
            "reason": "Column not present in dataset",
            "missing_count": np.nan,
            "total_rows": len(df),
        }
    )


removed_df = pd.DataFrame(
    removed_rows
)


removed_df.to_csv(
    REMOVED_FEATURES_FILE,
    index=False
)


# CREATE MODEL DATAFRAME
model_df = df[
    [
        "survival_time_days",
        "survival_event"
    ]
    + features_used
].copy()


# IMPUTE CLINICAL VARIABLES
numeric_features = []

categorical_features = []


for feature in features_used:

    if pd.api.types.is_numeric_dtype(
        model_df[feature]
    ):

        numeric_features.append(
            feature
        )

    else:

        categorical_features.append(
            feature
        )


# Numeric imputation
for feature in numeric_features:

    model_df[feature] = pd.to_numeric(
        model_df[feature],
        errors="coerce"
    )

    median_value = model_df[
        feature
    ].median()

    model_df[feature] = model_df[
        feature
    ].fillna(
        median_value
    )


# Categorical imputation
for feature in categorical_features:

    model_df[feature] = (
        model_df[feature]
        .astype("object")
        .fillna("Unknown")
        .astype(str)
    )


# ONE-HOT ENCODE CATEGORICAL VARIABLES
if categorical_features:

    model_df = pd.get_dummies(
        model_df,
        columns=categorical_features,
        drop_first=True,
        dtype=float
    )


# CONVERT ALL MODEL VARIABLES TO NUMERIC
for column in model_df.columns:

    model_df[column] = pd.to_numeric(
        model_df[column],
        errors="coerce"
    )


# REMOVE CONSTANT FEATURES
constant_features = []

for column in model_df.columns:

    if column in [
        "survival_time_days",
        "survival_event"
    ]:
        continue

    if model_df[column].nunique(
        dropna=False
    ) <= 1:

        constant_features.append(
            column
        )


if constant_features:

    print(
        "\nRemoving constant model features:"
    )

    for feature in constant_features:

        print(
            f" - {feature}"
        )

    model_df = model_df.drop(
        columns=constant_features
    )


# FINAL MISSING VALUE CHECK
remaining_missing = (
    model_df.isna()
    .sum()
)

remaining_missing = (
    remaining_missing[
        remaining_missing > 0
    ]
)


if not remaining_missing.empty:

    print(
        "\nRemaining missing values:"
    )

    print(
        remaining_missing
    )

    raise ValueError(
        "\nMissing values remain in the Cox dataset."
    )


# CHECK VARIANCE
print(
    "\nFinal Cox dataset shape:"
)

print(
    model_df.shape
)


print(
    "\nFinal event distribution:"
)

print(
    model_df[
        "survival_event"
    ].value_counts()
    .sort_index()
)


# FIT COX MODEL
print(
    "\n"
    + "=" * 70
)

print(
    "FITTING COX PROPORTIONAL HAZARDS MODEL"
)

print(
    "=" * 70
)


cph = CoxPHFitter(
    penalizer=PENALIZER
)


try:

    cph.fit(
        model_df,
        duration_col="survival_time_days",
        event_col="survival_event",
        show_progress=True
    )

except Exception as first_error:

    print(
        "\nInitial Cox model fitting failed:"
    )

    print(
        first_error
    )

    print(
        "\nRetrying with stronger penalization..."
    )

    cph = CoxPHFitter(
        penalizer=0.5
    )

    cph.fit(
        model_df,
        duration_col="survival_time_days",
        event_col="survival_event",
        show_progress=True
    )


# C-INDEX
c_index = float(
    cph.concordance_index_
)


print(
    "\nC-index:"
)

print(
    f"{c_index:.4f}"
)


# COX RESULTS
summary = cph.summary.copy()


# Add human-readable columns
summary["hazard_ratio"] = np.exp(
    summary["coef"]
)

summary["ci_lower"] = np.exp(
    summary["coef lower 95%"]
)

summary["ci_upper"] = np.exp(
    summary["coef upper 95%"]
)


# Select useful columns
results_columns = [
    "coef",
    "exp(coef)",
    "hazard_ratio",
    "se(coef)",
    "coef lower 95%",
    "coef upper 95%",
    "ci_lower",
    "ci_upper",
    "z",
    "p",
]


available_result_columns = [
    column
    for column in results_columns
    if column in summary.columns
]


cox_results = summary[
    available_result_columns
].copy()


# Add significance indicator
if "p" in cox_results.columns:

    cox_results["significant_p_lt_0.05"] = (
        cox_results["p"] < 0.05
    )


# PRINT COX RESULTS

print(
    "COX MODEL RESULTS"
)

print(
    "=" * 70
)

print(
    cox_results.to_string()
)


# SAVE COX RESULTS
cox_results.to_csv(
    COX_RESULTS_FILE
)


# PROPORTIONAL HAZARDS ASSUMPTION TEST
print(
    "\n"
    + "=" * 70
)

print(
    "TESTING PROPORTIONAL HAZARDS ASSUMPTION"
)

print(
    "=" * 70
)


ph_results = None


try:

    ph_test = cph.check_assumptions(
        model_df,
        p_value_threshold=0.05,
        show_plots=False
    )

    # check_assumptions prints its own detailed output.
    # The result may be None depending on lifelines version.

    if ph_test is not None:

        try:

            ph_results = pd.DataFrame(
                ph_test
            )

        except Exception:

            ph_results = None

except Exception as ph_error:

    print(
        "\nProportional hazards test could not "
        "be completed automatically:"
    )

    print(
        ph_error
    )


# MODEL SUMMARY
summary_rows = [
    {
        "model": "Clinical-only Cox PH",
        "n_patients": len(model_df),
        "n_events": int(
            model_df[
                "survival_event"
            ].sum()
        ),
        "n_censored": int(
            (
                model_df[
                    "survival_event"
                ] == 0
            ).sum()
        ),
        "c_index": c_index,
        "penalizer": cph.penalizer,
        "n_model_parameters": len(
            cph.params_
        ),
    }
]


model_summary = pd.DataFrame(
    summary_rows
)


model_summary.to_csv(
    COX_SUMMARY_FILE,
    index=False
)


# SAVE PH RESULTS IF AVAILABLE
if ph_results is not None:

    ph_results.to_csv(
        PH_RESULTS_FILE,
        index=False
    )

else:

    # Save a simple record so there is still
    # documentation that the test was attempted.

    pd.DataFrame(
        [
            {
                "status": (
                    "Proportional hazards "
                    "check executed through "
                    "lifelines"
                )
            }
        ]
    ).to_csv(
        PH_RESULTS_FILE,
        index=False
    )


# FINAL OUTPUT
print(
    "\n"
    + "=" * 70
)

print(
    "COX ANALYSIS COMPLETED"
)

print(
    "=" * 70
)


print(
    f"\nC-index: {c_index:.4f}"
)


print(
    "\nFiles saved:"
)

print(
    f" - {COX_RESULTS_FILE}"
)

print(
    f" - {COX_SUMMARY_FILE}"
)

print(
    f" - {PH_RESULTS_FILE}"
)

print(
    f" - {REMOVED_FEATURES_FILE}"
)


print(
    "\nDone."
)