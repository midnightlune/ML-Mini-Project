"""
7_coxanalysis.py

Clinical-only Cox Proportional Hazards analysis (stabilized).

Preprocessing and fitting live in cox_utils.py (shared with script 9).
The penalizer is chosen by cross-validated C-index.

Outputs (same filenames as before, so script 10 keeps working):
    - cox_full_results.csv
    - clinical_model_summary.csv      (adds penalizer, cv_c_index)
    - cox_proportional_hazards_test.csv
    - cox_removed_features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lifelines.statistics import proportional_hazard_test

from cox_utils import (
    DURATION_COL,
    EVENT_COL,
    fit_stable_cox,
    load_survival_data,
    prepare_cox_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "clinical_survival_clean.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COX_RESULTS_FILE = RESULTS_DIR / "cox_full_results.csv"
COX_SUMMARY_FILE = RESULTS_DIR / "clinical_model_summary.csv"
PH_RESULTS_FILE = RESULTS_DIR / "cox_proportional_hazards_test.csv"
REMOVED_FEATURES_FILE = RESULTS_DIR / "cox_removed_features.csv"


print("CLINICAL-ONLY COX PROPORTIONAL HAZARDS ANALYSIS")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")


# LOAD + PREPARE
df = load_survival_data(INPUT_FILE)

print(f"Patients with usable survival information: {len(df)}")
print(f"Deaths/events: {int(df[EVENT_COL].sum())}")
print(f"Censored: {int((df[EVENT_COL] == 0).sum())}")

model_df, info = prepare_cox_data(df)


# SAVE REMOVAL / MERGE REPORT
rows = []
for f in info["missing_from_dataset"]:
    rows.append({"feature": f, "reason": "Column not present in dataset"})
for f in info["all_missing"]:
    rows.append({"feature": f, "reason": "100% missing"})
for f in info["redundant"]:
    rows.append({"feature": f, "reason": "Redundant with AJCC stage"})
for f in info["constant"]:
    rows.append({"feature": f, "reason": "Constant after encoding"})
for f in info["correlated"]:
    rows.append({"feature": f, "reason": "Correlation > 0.9 with another column"})
for f, levels in info["merged_levels"].items():
    rows.append({
        "feature": f,
        "reason": "Rare levels merged into 'Other': " + ", ".join(levels),
    })

pd.DataFrame(rows, columns=["feature", "reason"]).to_csv(
    REMOVED_FEATURES_FILE, index=False
)


# FIT
cph, penalizer, cv_c_index = fit_stable_cox(model_df)

c_index = float(cph.concordance_index_)
print(f"\nApparent (in-sample) C-index: {c_index:.4f}")
print(f"Cross-validated C-index:      {cv_c_index:.4f}")


# RESULTS TABLE
summary = cph.summary.copy()

summary["hazard_ratio"] = np.exp(summary["coef"])
summary["ci_lower"] = np.exp(summary["coef lower 95%"])
summary["ci_upper"] = np.exp(summary["coef upper 95%"])

cols = [
    "coef", "exp(coef)", "hazard_ratio", "se(coef)",
    "coef lower 95%", "coef upper 95%", "ci_lower", "ci_upper", "z", "p",
]
cox_results = summary[[c for c in cols if c in summary.columns]].copy()
cox_results["significant_p_lt_0.05"] = cox_results["p"] < 0.05

print("\nCOX MODEL RESULTS")
print("=" * 70)
print(cox_results.to_string())

cox_results.to_csv(COX_RESULTS_FILE)


# PROPORTIONAL HAZARDS TEST
print("\nTESTING PROPORTIONAL HAZARDS ASSUMPTION")
print("=" * 70)

try:
    ph = proportional_hazard_test(cph, model_df, time_transform="rank")
    ph_table = ph.summary.reset_index().rename(columns={"index": "feature"})
    print(ph_table.to_string(index=False))
    ph_table.to_csv(PH_RESULTS_FILE, index=False)
except Exception as error:
    print(f"PH test could not be completed: {error}")
    pd.DataFrame([{"status": f"PH test failed: {error}"}]).to_csv(
        PH_RESULTS_FILE, index=False
    )


# MODEL SUMMARY
n_events = int(model_df[EVENT_COL].sum())
n_params = len(cph.params_)

pd.DataFrame([{
    "model": "Clinical-only Cox PH",
    "n_patients": len(model_df),
    "n_events": n_events,
    "n_censored": int((model_df[EVENT_COL] == 0).sum()),
    "c_index": c_index,
    "cv_c_index": cv_c_index,
    "penalizer": penalizer,
    "n_model_parameters": n_params,
    "events_per_parameter": n_events / n_params,
}]).to_csv(COX_SUMMARY_FILE, index=False)


print("\nFiles saved:")
for p in (COX_RESULTS_FILE, COX_SUMMARY_FILE, PH_RESULTS_FILE, REMOVED_FEATURES_FILE):
    print(f" - {p}")
print("\nDone.")