"""
9_coxriskkm.py

Cox-derived risk groups + Kaplan-Meier analysis.

Steps:
    1. Prepare data and fit the stabilized clinical Cox model (cox_utils).
    2. Compute OUT-OF-FOLD risk scores: each patient is scored by a model
       that never saw that patient (5-fold cross-validation).
    3. Split patients at the median risk score into low / high risk.
    4. Kaplan-Meier curves and a log-rank test between the groups.

Notes:
    - Clinical data only. No SMOTE. No radiomics.
    - Censored patients are kept.
    - Out-of-fold scores come from different fold models, so the pooled
      median split is approximate, but it is not in-sample.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from cox_utils import (
    DURATION_COL,
    EVENT_COL,
    cross_val_risk_scores,
    fit_stable_cox,
    load_survival_data,
    prepare_cox_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "clinical_survival_clean.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RISK_RESULTS_FILE = RESULTS_DIR / "cox_risk_groups.csv"
LOGRANK_RESULTS_FILE = RESULTS_DIR / "cox_risk_logrank_results.csv"
PLOT_FILE = RESULTS_DIR / "kaplan_meier_cox_risk_groups.png"


print("COX-DERIVED RISK GROUPS + KAPLAN-MEIER ANALYSIS")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nClinical survival file not found:\n{INPUT_FILE}\n\n"
        "Run the data-cleaning script that creates "
        "clinical_survival_clean.csv first."
    )


# LOAD + PREPARE
df = load_survival_data(INPUT_FILE)

print(f"Patients with usable survival information: {len(df)}")
print(f"Events: {int(df[EVENT_COL].sum())}")
print(f"Censored: {int((df[EVENT_COL] == 0).sum())}")

model_df, info = prepare_cox_data(df)


# FIT (to choose the penalizer)
cph, penalizer, cv_c_index = fit_stable_cox(model_df)

print(f"\nCross-validated C-index: {cv_c_index:.4f}")


# OUT-OF-FOLD RISK SCORES
print("\nCalculating out-of-fold Cox risk scores...")

risk_scores = cross_val_risk_scores(model_df, penalizer)

median_risk = float(risk_scores.median())
print(f"Median risk score: {median_risk:.6f}")

risk_group = np.where(risk_scores <= median_risk, "Low risk", "High risk")


# ANALYSIS DATAFRAME
risk_df = pd.DataFrame({
    "survival_time_days": model_df[DURATION_COL].values,
    "survival_event": model_df[EVENT_COL].values,
    "cox_risk_score": risk_scores.values,
    "risk_group": risk_group,
})

low_risk = risk_df[risk_df["risk_group"] == "Low risk"]
high_risk = risk_df[risk_df["risk_group"] == "High risk"]

print("\n" + "=" * 70)
print("COX RISK GROUPS")
print("=" * 70)
print(f"Low-risk patients:  {len(low_risk)} "
      f"({int(low_risk['survival_event'].sum())} events)")
print(f"High-risk patients: {len(high_risk)} "
      f"({int(high_risk['survival_event'].sum())} events)")


# LOG-RANK TEST
result = logrank_test(
    low_risk["survival_time_days"],
    high_risk["survival_time_days"],
    event_observed_A=low_risk["survival_event"],
    event_observed_B=high_risk["survival_event"],
)

logrank_statistic = float(result.test_statistic)
logrank_p_value = float(result.p_value)

print("\n" + "=" * 70)
print("LOG-RANK TEST")
print("=" * 70)
print(f"Test statistic: {logrank_statistic:.4f}")
print(f"p-value: {logrank_p_value:.6f}")

if logrank_p_value < 0.05:
    print("\nOut-of-fold risk groups have statistically different survival "
          "at the 0.05 level.")
else:
    print("\nNo statistically significant difference between out-of-fold "
          "risk groups at the 0.05 level.")


# KAPLAN-MEIER PLOT
plt.figure(figsize=(10, 6))
ax = plt.gca()

for label, group in (("Low risk", low_risk), ("High risk", high_risk)):
    km = KaplanMeierFitter()
    km.fit(
        group["survival_time_days"],
        group["survival_event"],
        label=f"{label} (n={len(group)})",
    )
    km.plot_survival_function(ax=ax)

plt.title("Kaplan-Meier Survival by Cox-Derived Risk Group "
          "(out-of-fold scores)")
plt.xlabel("Time (days)")
plt.ylabel("Survival Probability")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=300)
plt.close()


# SAVE
risk_df.to_csv(RISK_RESULTS_FILE, index=False)

pd.DataFrame([{
    "risk_threshold": median_risk,
    "low_risk_n": len(low_risk),
    "high_risk_n": len(high_risk),
    "low_risk_events": int(low_risk["survival_event"].sum()),
    "high_risk_events": int(high_risk["survival_event"].sum()),
    "logrank_test_statistic": logrank_statistic,
    "logrank_p_value": logrank_p_value,
    "penalizer": penalizer,
    "risk_score_type": "out-of-fold (5-fold CV)",
}]).to_csv(LOGRANK_RESULTS_FILE, index=False)


print("\nFiles saved:")
print(f" - {RISK_RESULTS_FILE}")
print(f" - {LOGRANK_RESULTS_FILE}")
print(f" - {PLOT_FILE}")
print("\nDone.")