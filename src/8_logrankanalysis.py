"""
Log-Rank Survival Analysis

Person 1: Clinical + Survival Analysis

Purpose:
    Compare survival distributions between two clinical
    survival groups using the selected survival threshold.

Groups:
    Group 0 = survival <= selected threshold
    Group 1 = survival > selected threshold

The threshold is obtained from:
    results/naive_bayes_threshold_results.csv

Outputs:
    - Log-rank statistic
    - p-value
    - Group sizes
    - Kaplan-Meier survival plot
    - CSV results
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
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

THRESHOLD_FILE = (
    RESULTS_DIR
    / "naive_bayes_threshold_results.csv"
)

RESULT_FILE = (
    RESULTS_DIR
    / "logrank_threshold_results.csv"
)

PLOT_FILE = (
    RESULTS_DIR
    / "kaplan_meier_by_survival_threshold.png"
)


# HEADER
print("=" * 70)
print("LOG-RANK SURVIVAL ANALYSIS")
print("=" * 70)


# CHECK FILES
if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nClinical survival file not found:\n"
        f"{INPUT_FILE}\n\n"
        "Run 02_clean_clinical_data.py first."
    )


if not THRESHOLD_FILE.exists():

    raise FileNotFoundError(
        f"\nThreshold results file not found:\n"
        f"{THRESHOLD_FILE}\n\n"
        "Run 09_naive_bayes_threshold.py first."
    )


# LOAD DATA
df = pd.read_csv(
    INPUT_FILE
)

threshold_results = pd.read_csv(
    THRESHOLD_FILE
)


print(
    f"\nOriginal dataset shape: {df.shape}"
)


# GET SELECTED THRESHOLD
valid_thresholds = threshold_results[
    threshold_results["balanced_accuracy"].notna()
].copy()


if valid_thresholds.empty:

    raise ValueError(
        "\nNo valid threshold results were found."
    )


best_index = valid_thresholds[
    "balanced_accuracy"
].idxmax()


selected_threshold = int(
    valid_thresholds.loc[
        best_index,
        "threshold_days"
    ]
)


print(
    f"\nSelected survival threshold: "
    f"{selected_threshold} days"
)


# CHECK REQUIRED COLUMNS
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


# Remove records without survival information
df = df.dropna(
    subset=[
        "survival_time_days",
        "survival_event"
    ]
).copy()


# Remove impossible survival times
df = df[
    df["survival_time_days"] > 0
].copy()


print(
    f"Patients with usable survival information: "
    f"{len(df)}"
)


# CREATE SURVIVAL GROUPS
"""
Group 0:
    Survival time <= threshold

Group 1:
    Survival time > threshold

IMPORTANT:
    Unlike the Naive Bayes classification step, this
    log-rank analysis retains BOTH:
        - deaths/events
        - censored/alive observations

This is appropriate for survival analysis because
censored observations still contain information about
survival up to their censoring time.
"""

df["survival_group"] = (
    df["survival_time_days"]
    > selected_threshold
).astype(int)


# GROUP SIZES
group_0 = df[
    df["survival_group"] == 0
].copy()

group_1 = df[
    df["survival_group"] == 1
].copy()


print(
    "\nSurvival group distribution:"
)

print(
    f"Group 0 (<= {selected_threshold} days): "
    f"{len(group_0)} patients"
)

print(
    f"Group 1 (> {selected_threshold} days): "
    f"{len(group_1)} patients"
)


if len(group_0) == 0 or len(group_1) == 0:

    raise ValueError(
        "\nBoth survival groups must contain patients."
    )


# LOG-RANK TEST
print(
    "\n"
    + "=" * 70
)

print(
    "PERFORMING LOG-RANK TEST"
)

print(
    "=" * 70
)


logrank_result = logrank_test(
    group_0["survival_time_days"],
    group_1["survival_time_days"],
    event_observed_A=group_0["survival_event"],
    event_observed_B=group_1["survival_event"]
)


test_statistic = float(
    logrank_result.test_statistic
)

p_value = float(
    logrank_result.p_value
)


# PRINT LOG-RANK RESULTS
print(
    f"\nLog-rank test statistic: "
    f"{test_statistic:.4f}"
)

print(
    f"p-value: "
    f"{p_value:.6f}"
)


if p_value < 0.05:

    print(
        "\nThe survival distributions differ "
        "statistically at the 0.05 significance level."
    )

else:

    print(
        "\nThe test does not show a statistically "
        "significant difference at the 0.05 level."
    )


# KAPLAN-MEIER CURVES
print(
    "\nGenerating Kaplan-Meier plot..."
)


kmf_0 = KaplanMeierFitter()

kmf_1 = KaplanMeierFitter()


kmf_0.fit(
    durations=group_0["survival_time_days"],
    event_observed=group_0["survival_event"],
    label=f"≤ {selected_threshold} days"
)


kmf_1.fit(
    durations=group_1["survival_time_days"],
    event_observed=group_1["survival_event"],
    label=f"> {selected_threshold} days"
)


# PLOT
plt.figure(
    figsize=(10, 6)
)


kmf_0.plot_survival_function()

kmf_1.plot_survival_function()


plt.title(
    "Kaplan–Meier Survival by Survival-Time Group"
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


# SAVE RESULTS
result_df = pd.DataFrame(
    [
        {
            "threshold_days": selected_threshold,
            "group_0_label": (
                f"<= {selected_threshold} days"
            ),
            "group_1_label": (
                f"> {selected_threshold} days"
            ),
            "group_0_n": len(group_0),
            "group_1_n": len(group_1),
            "logrank_test_statistic": test_statistic,
            "p_value": p_value,
        }
    ]
)


result_df.to_csv(
    RESULT_FILE,
    index=False
)


# FINAL OUTPUT
print(
    "\n"
    + "=" * 70
)

print(
    "LOG-RANK ANALYSIS COMPLETED"
)

print(
    "=" * 70
)


print(
    "\nFiles saved:"
)

print(
    f" - {RESULT_FILE}"
)

print(
    f" - {PLOT_FILE}"
)


print("\nDone.")