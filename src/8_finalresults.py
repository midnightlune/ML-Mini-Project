"""
10_finalresults.py

Collects the final results from:
    5 - Naive Bayes survival threshold
    6 - SMOTENC + Naive Bayes
    7 - Clinical-only Cox PH model (stabilized)
    8 - Log-rank tests by baseline clinical groups
    9 - Cox-derived risk groups (out-of-fold) + Kaplan-Meier

Outputs:
    - final_clinical_results.csv          (hazard ratios per feature)
    - final_logrank_results.csv           (log-rank per clinical feature)
    - final_clinical_model_summary.csv    (one-row overall summary)
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

# INPUTS
THRESHOLD_FILE = RESULTS_DIR / "naive_bayes_threshold_results.csv"
CLASSIFICATION_FILE = RESULTS_DIR / "naive_bayes_classification_results.csv"
CLASS_DISTRIBUTION_FILE = RESULTS_DIR / "smote_class_distribution.csv"
COX_RESULTS_FILE = RESULTS_DIR / "cox_full_results.csv"
COX_SUMMARY_FILE = RESULTS_DIR / "clinical_model_summary.csv"
LOGRANK_CLINICAL_FILE = RESULTS_DIR / "logrank_clinical_results.csv"
COX_RISK_LOGRANK_FILE = RESULTS_DIR / "cox_risk_logrank_results.csv"

# OUTPUTS
FINAL_RESULTS_FILE = RESULTS_DIR / "final_clinical_results.csv"
FINAL_LOGRANK_FILE = RESULTS_DIR / "final_logrank_results.csv"
FINAL_MODEL_SUMMARY_FILE = RESULTS_DIR / "final_clinical_model_summary.csv"


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_required(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired result file not found:\n{path}\n\n"
            "Run the corresponding previous script first."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"\nResult file is empty:\n{path}")
    print(f"OK: {path.name}")
    return df


print("FINAL CLINICAL RESULTS")
print("\nChecking previous analysis results...")

threshold_results = load_required(THRESHOLD_FILE)
classification_results = load_required(CLASSIFICATION_FILE)
class_distribution = load_required(CLASS_DISTRIBUTION_FILE)
load_required(COX_RESULTS_FILE)  # existence check
cox_results = pd.read_csv(COX_RESULTS_FILE, index_col=0)
cox_summary = load_required(COX_SUMMARY_FILE)
clinical_logrank = load_required(LOGRANK_CLINICAL_FILE)
cox_risk_logrank = load_required(COX_RISK_LOGRANK_FILE)


# 1. NAIVE BAYES THRESHOLD
valid = threshold_results[threshold_results["balanced_accuracy"].notna()]

if valid.empty:
    raise ValueError("\nNo valid Naive Bayes threshold results found.")

best = valid.loc[valid["balanced_accuracy"].idxmax()]
selected_threshold = int(best["threshold_days"])
threshold_bal_acc = float(best["balanced_accuracy"])

banner("1. NAIVE BAYES THRESHOLD")
print(f"Selected threshold: {selected_threshold} days")
print(f"Cross-validated balanced accuracy: {threshold_bal_acc:.4f}")


# 2. SMOTENC + NAIVE BAYES
row = classification_results.iloc[0]

nb = {
    "accuracy": float(row["accuracy"]),
    "balanced_accuracy": float(row["balanced_accuracy"]),
    "recall": float(row["recall"]),
    "f1": float(row["f1"]),
    "roc_auc": float(row["roc_auc"]),
}

banner("2. SMOTENC + NAIVE BAYES (held-out test set)")
print(f"Threshold: {int(row['threshold_days'])} days")
for name, value in nb.items():
    print(f"{name}: {value:.4f}")

print("\nTraining class distribution:")
print(class_distribution.to_string(index=False))


# 3. COX MODEL SUMMARY
c = cox_summary.iloc[0]

cox_n_patients = int(c["n_patients"])
cox_n_events = int(c["n_events"])
cox_n_censored = int(c["n_censored"])
cox_c_index = float(c["c_index"])
cox_n_params = int(c["n_model_parameters"])
cox_cv_c_index = float(c["cv_c_index"]) if "cv_c_index" in c else np.nan
cox_penalizer = float(c["penalizer"]) if "penalizer" in c else np.nan
events_per_param = cox_n_events / cox_n_params

banner("3. CLINICAL COX MODEL")
print(f"Patients: {cox_n_patients}")
print(f"Events: {cox_n_events}")
print(f"Censored: {cox_n_censored}")
print(f"Model parameters: {cox_n_params}")
print(f"Events per parameter: {events_per_param:.1f} (aim for >= 10)")
print(f"Penalizer: {cox_penalizer}")
print(f"Apparent C-index: {cox_c_index:.4f}")
print(f"Cross-validated C-index: {cox_cv_c_index:.4f}")


# 4. COX HAZARD RATIOS
banner("4. COX HAZARD RATIOS")

features = cox_results.copy()
features.index.name = "feature"
features = features.reset_index()

features = features.rename(columns={"p": "p_value"})

if "hazard_ratio" not in features.columns and "exp(coef)" in features.columns:
    features = features.rename(columns={"exp(coef)": "hazard_ratio"})

features = features.loc[:, ~features.columns.duplicated()]

if {"ci_lower", "ci_upper"}.issubset(features.columns):
    features["95% CI"] = (
        features["ci_lower"].round(4).astype(str)
        + " - "
        + features["ci_upper"].round(4).astype(str)
    )

if "p_value" in features.columns:
    features["significant_p_lt_0.05"] = features["p_value"] < 0.05

preferred = [
    "feature", "hazard_ratio", "95% CI", "p_value", "significant_p_lt_0.05",
]
final_feature_table = features[[c for c in preferred if c in features.columns]]

print(final_feature_table.to_string(index=False))
final_feature_table.to_csv(FINAL_RESULTS_FILE, index=False)


# 5. LOG-RANK BY BASELINE CLINICAL GROUPS
banner("5. LOG-RANK BY BASELINE CLINICAL GROUPS")

logrank_cols = [
    "feature", "n_patients", "n_events", "n_groups",
    "test_statistic", "p_value", "p_value_fdr",
]
logrank_table = clinical_logrank[
    [c for c in logrank_cols if c in clinical_logrank.columns]
]
print(logrank_table.to_string(index=False))

n_features_tested = len(clinical_logrank)
n_sig_raw = int((clinical_logrank["p_value"] < 0.05).sum())
n_sig_fdr = int((clinical_logrank["p_value_fdr"] < 0.05).sum())

print(f"\nFeatures tested: {n_features_tested}")
print(f"Significant (raw p < 0.05): {n_sig_raw}")
print(f"Significant (FDR < 0.05):   {n_sig_fdr}")

clinical_logrank.to_csv(FINAL_LOGRANK_FILE, index=False)


# 6. COX RISK GROUPS
r = cox_risk_logrank.iloc[0]

risk = {
    "low_n": int(r["low_risk_n"]),
    "high_n": int(r["high_risk_n"]),
    "low_events": int(r["low_risk_events"]),
    "high_events": int(r["high_risk_events"]),
    "statistic": float(r["logrank_test_statistic"]),
    "p_value": float(r["logrank_p_value"]),
}

banner("6. COX-DERIVED RISK GROUPS (out-of-fold scores)")
print(f"Median risk threshold: {float(r['risk_threshold']):.6f}")
print(f"Low-risk:  {risk['low_n']} patients, {risk['low_events']} events")
print(f"High-risk: {risk['high_n']} patients, {risk['high_events']} events")
print(f"Log-rank statistic: {risk['statistic']:.4f}")
print(f"Log-rank p-value: {risk['p_value']:.6f}")


# 7. OVERALL SUMMARY
summary = pd.DataFrame([{
    "analysis": "Clinical-only Cox Proportional Hazards",
    "n_patients": cox_n_patients,
    "n_events": cox_n_events,
    "n_censored": cox_n_censored,
    "n_model_parameters": cox_n_params,
    "events_per_parameter": events_per_param,
    "penalizer": cox_penalizer,
    "c_index_apparent": cox_c_index,
    "c_index_cross_validated": cox_cv_c_index,

    "naive_bayes_threshold_days": selected_threshold,
    "naive_bayes_threshold_balanced_accuracy": threshold_bal_acc,
    "smotenc_accuracy": nb["accuracy"],
    "smotenc_balanced_accuracy": nb["balanced_accuracy"],
    "smotenc_recall": nb["recall"],
    "smotenc_f1": nb["f1"],
    "smotenc_roc_auc": nb["roc_auc"],

    "logrank_features_tested": n_features_tested,
    "logrank_significant_raw": n_sig_raw,
    "logrank_significant_fdr": n_sig_fdr,

    "cox_risk_low_n": risk["low_n"],
    "cox_risk_high_n": risk["high_n"],
    "cox_risk_logrank_statistic": risk["statistic"],
    "cox_risk_logrank_p_value": risk["p_value"],
}])

summary.to_csv(FINAL_MODEL_SUMMARY_FILE, index=False)

banner("OVERALL MODEL SUMMARY")
print(summary.T.to_string(header=False))

print("\nFiles saved:")
print(f" - {FINAL_RESULTS_FILE}")
print(f" - {FINAL_LOGRANK_FILE}")
print(f" - {FINAL_MODEL_SUMMARY_FILE}")
print("\nDone.")