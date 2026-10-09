"""
Cox-Derived Risk Groups + Kaplan-Meier Analysis
Purpose:
    1. Fit the clinical-only Cox proportional hazards model.
    2. Calculate Cox partial hazard/risk scores.
    3. Divide patients into low- and high-risk groups
       using the median Cox risk score.
    4. Generate Kaplan-Meier survival curves.
    5. Perform a log-rank test between the two risk groups.

Important:
    - Uses the clinical dataset only.
    - Does NOT use SMOTE.
    - Does NOT use radiomic features.
    - Uses both event and censored patients.
    - The risk groups are descriptive/in-sample because
      the Cox model and risk groups are generated from
      the same cohort.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test


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

RISK_RESULTS_FILE = (
    RESULTS_DIR
    / "cox_risk_groups.csv"
)

LOGRANK_RESULTS_FILE = (
    RESULTS_DIR
    / "cox_risk_logrank_results.csv"
)

PLOT_FILE = (
    RESULTS_DIR
    / "kaplan_meier_cox_risk_groups.png"
)


# SETTINGS
PENALIZER = 0.1


print("COX-DERIVED RISK GROUPS + KAPLAN-MEIER ANALYSIS")
print("=" * 70)


# CHECK INPUT
if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nClinical survival file not found:\n"
        f"{INPUT_FILE}\n\n"
        "Run 02_clean_clinical_data.py first."
    )


# LOAD DATA
df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nOriginal dataset shape: {df.shape}"
)


# CHECK SURVIVAL VARIABLES
required_columns = [
    "survival_time_days",
    "survival_event",
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"\nRequired column missing: {column}"
        )


# PREPARE SURVIVAL DATA
df["survival_time_days"] = pd.to_numeric(
    df["survival_time_days"],
    errors="coerce"
)

df["survival_event"] = pd.to_numeric(
    df["survival_event"],
    errors="coerce"
)


# Remove missing survival information
df = df.dropna(
    subset=[
        "survival_time_days",
        "survival_event"
    ]
).copy()


# Remove invalid survival times
df = df[
    df["survival_time_days"] > 0
].copy()


print(
    f"Patients with usable survival information: "
    f"{len(df)}"
)

print(
    f"Events/deaths: "
    f"{int(df['survival_event'].sum())}"
)

print(
    f"Censored observations: "
    f"{int((df['survival_event'] == 0).sum())}"
)


# CLINICAL FEATURES
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


# FIND AVAILABLE FEATURES
available_features = [
    feature
    for feature in candidate_features
    if feature in df.columns
]


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


if not features_used:

    raise ValueError(
        "\nNo usable clinical features remain."
    )


# CREATE MODEL DATA
model_df = df[
    [
        "survival_time_days",
        "survival_event"
    ]
    + features_used
].copy()


# IDENTIFY FEATURE TYPES
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


# NUMERIC IMPUTATION
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


# CATEGORICAL IMPUTATION
for feature in categorical_features:

    model_df[feature] = (
        model_df[feature]
        .astype("object")
        .fillna("Unknown")
        .astype(str)
    )


# ONE-HOT ENCODING
if categorical_features:

    model_df = pd.get_dummies(
        model_df,
        columns=categorical_features,
        drop_first=True,
        dtype=float
    )


# CONVERT TO NUMERIC
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
        "\nRemoving constant features:"
    )

    for feature in constant_features:

        print(
            f" - {feature}"
        )

    model_df = model_df.drop(
        columns=constant_features
    )


# FINAL MISSING VALUE CHECK
missing_values = (
    model_df.isna().sum()
)

missing_values = (
    missing_values[
        missing_values > 0
    ]
)


if not missing_values.empty:

    print(
        "\nRemaining missing values:"
    )

    print(
        missing_values
    )

    raise ValueError(
        "\nMissing values remain. "
        "Cannot fit Cox model."
    )


# FIT COX MODEL
print(
    "\n"
    + "=" * 70
)

print(
    "FITTING CLINICAL-ONLY COX MODEL"
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
        "\nInitial Cox model failed:"
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


# CALCULATE COX PARTIAL HAZARD
print(
    "\nCalculating Cox risk scores..."
)


risk_scores = cph.predict_partial_hazard(
    model_df
)


# Convert Series/DataFrame to 1D Series
if isinstance(
    risk_scores,
    pd.DataFrame
):

    risk_scores = risk_scores.iloc[
        :, 0
    ]


risk_scores = pd.Series(
    np.asarray(
        risk_scores
    ).reshape(-1),
    index=model_df.index,
    name="cox_risk_score"
)


# MEDIAN RISK THRESHOLD
median_risk = float(
    risk_scores.median()
)


print(
    f"\nMedian Cox risk score: "
    f"{median_risk:.6f}"
)


# CREATE RISK GROUPS
"""
Low risk:
    Cox risk score <= median

High risk:
    Cox risk score > median
"""

risk_group = np.where(
    risk_scores <= median_risk,
    "Low risk",
    "High risk"
)


# CREATE ANALYSIS DATAFRAME
risk_df = pd.DataFrame(
    {
        "survival_time_days": model_df[
            "survival_time_days"
        ].values,

        "survival_event": model_df[
            "survival_event"
        ].values,

        "cox_risk_score": risk_scores.values,

        "risk_group": risk_group,
    }
)


# GROUP COUNTS
low_risk = risk_df[
    risk_df["risk_group"] == "Low risk"
].copy()

high_risk = risk_df[
    risk_df["risk_group"] == "High risk"
].copy()


print(
    "\n"
    + "=" * 70
)

print(
    "COX RISK GROUPS"
)

print(
    "=" * 70
)

print(
    f"Low-risk patients:  "
    f"{len(low_risk)}"
)

print(
    f"High-risk patients: "
    f"{len(high_risk)}"
)


print(
    "\nEvents by risk group:"
)

print(
    low_risk[
        "survival_event"
    ].value_counts()
    .sort_index()
)

print(
    high_risk[
        "survival_event"
    ].value_counts()
    .sort_index()
)


# LOG-RANK TEST
print(
    "\n"
    + "=" * 70
)

print(
    "LOG-RANK TEST"
)

print(
    "=" * 70
)


logrank_result = logrank_test(
    low_risk[
        "survival_time_days"
    ],

    high_risk[
        "survival_time_days"
    ],

    event_observed_A=low_risk[
        "survival_event"
    ],

    event_observed_B=high_risk[
        "survival_event"
    ]
)


logrank_statistic = float(
    logrank_result.test_statistic
)

logrank_p_value = float(
    logrank_result.p_value
)


print(
    f"\nLog-rank test statistic: "
    f"{logrank_statistic:.6f}"
)

print(
    f"Log-rank p-value: "
    f"{logrank_p_value:.6f}"
)


if logrank_p_value < 0.05:

    print(
        "\nThe two Cox-derived risk groups have "
        "statistically different survival distributions "
        "at the 0.05 significance level."
    )

else:

    print(
        "\nThe log-rank test does not show a "
        "statistically significant difference at "
        "the 0.05 significance level."
    )


# KAPLAN-MEIER FITTING
print(
    "\nGenerating Kaplan-Meier curves..."
)


km_low = KaplanMeierFitter()

km_high = KaplanMeierFitter()


km_low.fit(
    durations=low_risk[
        "survival_time_days"
    ],

    event_observed=low_risk[
        "survival_event"
    ],

    label="Low risk"
)


km_high.fit(
    durations=high_risk[
        "survival_time_days"
    ],

    event_observed=high_risk[
        "survival_event"
    ],

    label="High risk"
)


# KAPLAN-MEIER PLOT
plt.figure(
    figsize=(10, 6)
)


km_low.plot_survival_function()

km_high.plot_survival_function()


plt.title(
    "Kaplan–Meier Survival by Cox-Derived Risk Group"
)

plt.xlabel(
    "Time (days)"
)

plt.ylabel(
    "Survival Probability"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


plt.savefig(
    PLOT_FILE,
    dpi=300
)

plt.close()


# SAVE PATIENT RISK GROUPS
risk_df.to_csv(
    RISK_RESULTS_FILE,
    index=False
)


# SAVE LOG-RANK RESULTS
logrank_df = pd.DataFrame(
    [
        {
            "risk_threshold": median_risk,

            "low_risk_n": len(low_risk),

            "high_risk_n": len(high_risk),

            "low_risk_events": int(
                low_risk[
                    "survival_event"
                ].sum()
            ),

            "high_risk_events": int(
                high_risk[
                    "survival_event"
                ].sum()
            ),

            "logrank_test_statistic":
                logrank_statistic,

            "logrank_p_value":
                logrank_p_value,
        }
    ]
)


logrank_df.to_csv(
    LOGRANK_RESULTS_FILE,
    index=False
)


# FINAL OUTPUT
print(
    "\n"
    + "=" * 70
)

print(
    "COX RISK GROUP + KAPLAN-MEIER ANALYSIS COMPLETED"
)

print(
    "=" * 70
)


print(
    "\nFiles saved:"
)

print(
    f" - {RISK_RESULTS_FILE}"
)

print(
    f" - {LOGRANK_RESULTS_FILE}"
)

print(
    f" - {PLOT_FILE}"
)


print(
    "\nDone."
)