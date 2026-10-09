"""
Collects the final results from:

    5 - Naive Bayes survival threshold
    6 - SMOTENC + Naive Bayes
    7 - Clinical-only Cox PH model
    8 - Log-rank survival analysis
    9 - Cox-derived risk groups + Kaplan-Meier
"""


from pathlib import Path

import numpy as np
import pandas as pd


# PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)


# INPUT FILES
THRESHOLD_FILE = (
    RESULTS_DIR
    / "naive_bayes_threshold_results.csv"
)

CLASSIFICATION_FILE = (
    RESULTS_DIR
    / "naive_bayes_classification_results.csv"
)

CLASS_DISTRIBUTION_FILE = (
    RESULTS_DIR
    / "smote_class_distribution.csv"
)

COX_RESULTS_FILE = (
    RESULTS_DIR
    / "cox_full_results.csv"
)

COX_SUMMARY_FILE = (
    RESULTS_DIR
    / "clinical_model_summary.csv"
)

LOGRANK_THRESHOLD_FILE = (
    RESULTS_DIR
    / "logrank_threshold_results.csv"
)

COX_RISK_LOGRANK_FILE = (
    RESULTS_DIR
    / "cox_risk_logrank_results.csv"
)

RISK_GROUP_FILE = (
    RESULTS_DIR
    / "cox_risk_groups.csv"
)


# OUTPUT FILES
FINAL_RESULTS_FILE = (
    RESULTS_DIR
    / "final_clinical_results.csv"
)

FINAL_MODEL_SUMMARY_FILE = (
    RESULTS_DIR
    / "final_clinical_model_summary.csv"
)


print("FINAL CLINICAL RESULTS")

# HELPER FUNCTION
def check_file(path):
    """
    Check whether a required results file exists.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired result file not found:\n"
            f"{path}\n\n"
            "Run the corresponding previous script first."
        )


# CHECK REQUIRED FILES
required_files = [
    THRESHOLD_FILE,
    CLASSIFICATION_FILE,
    CLASS_DISTRIBUTION_FILE,
    COX_RESULTS_FILE,
    COX_SUMMARY_FILE,
    LOGRANK_THRESHOLD_FILE,
    COX_RISK_LOGRANK_FILE,
    RISK_GROUP_FILE,
]


print(
    "\nChecking previous analysis results..."
)


for file in required_files:

    check_file(file)

    print(
        f"OK: {file.name}"
    )


# LOAD RESULTS
threshold_results = pd.read_csv(
    THRESHOLD_FILE
)

classification_results = pd.read_csv(
    CLASSIFICATION_FILE
)

class_distribution = pd.read_csv(
    CLASS_DISTRIBUTION_FILE
)

cox_results = pd.read_csv(
    COX_RESULTS_FILE,
    index_col=0
)

cox_summary = pd.read_csv(
    COX_SUMMARY_FILE
)

threshold_logrank = pd.read_csv(
    LOGRANK_THRESHOLD_FILE
)

cox_risk_logrank = pd.read_csv(
    COX_RISK_LOGRANK_FILE
)

risk_groups = pd.read_csv(
    RISK_GROUP_FILE
)


# 1. NAIVE BAYES THRESHOLD
valid_thresholds = threshold_results[
    threshold_results["balanced_accuracy"].notna()
].copy()


if valid_thresholds.empty:

    raise ValueError(
        "\nNo valid Naive Bayes threshold results found."
    )


best_threshold_index = valid_thresholds[
    "balanced_accuracy"
].idxmax()


selected_threshold = int(
    valid_thresholds.loc[
        best_threshold_index,
        "threshold_days"
    ]
)


best_threshold_balanced_accuracy = float(
    valid_thresholds.loc[
        best_threshold_index,
        "balanced_accuracy"
    ]
)


print("\n"+ "=" * 70)

print(
    "1. NAIVE BAYES THRESHOLD"
)

print(
    "=" * 70
)

print(
    f"Selected threshold: "
    f"{selected_threshold} days"
)

print(
    f"Balanced accuracy: "
    f"{best_threshold_balanced_accuracy:.4f}"
)


# 2. SMOTENC + NAIVE BAYES
if classification_results.empty:

    raise ValueError(
        "\nNaive Bayes classification results are empty."
    )


classification_row = (
    classification_results.iloc[0]
)


classification_threshold = int(
    classification_row[
        "threshold_days"
    ]
)


classification_accuracy = float(
    classification_row[
        "accuracy"
    ]
)


classification_balanced_accuracy = float(
    classification_row[
        "balanced_accuracy"
    ]
)


classification_recall = float(
    classification_row[
        "recall"
    ]
)


classification_f1 = float(
    classification_row[
        "f1"
    ]
)


classification_roc_auc = float(
    classification_row[
        "roc_auc"
    ]
)


print(
    "\n"
    + "=" * 70
)

print(
    "2. SMOTENC + NAIVE BAYES"
)

print(
    "=" * 70
)

print(
    f"Threshold: "
    f"{classification_threshold} days"
)

print(
    f"Accuracy: "
    f"{classification_accuracy:.4f}"
)

print(
    f"Balanced accuracy: "
    f"{classification_balanced_accuracy:.4f}"
)

print(
    f"Recall: "
    f"{classification_recall:.4f}"
)

print(
    f"F1: "
    f"{classification_f1:.4f}"
)

print(
    f"ROC-AUC: "
    f"{classification_roc_auc:.4f}"
)


# 3. SMOTENC CLASS DISTRIBUTION
before_smote = class_distribution[
    class_distribution["stage"]
    == "Before SMOTENC"
].copy()


after_smote = class_distribution[
    class_distribution["stage"]
    == "After SMOTENC"
].copy()


print(
    "\nTraining class distribution:"
)

print(
    class_distribution.to_string(
        index=False
    )
)


# 4. COX MODEL SUMMARY
if cox_summary.empty:

    raise ValueError(
        "\nClinical Cox summary is empty."
    )


cox_summary_row = (
    cox_summary.iloc[0]
)


cox_n_patients = int(
    cox_summary_row[
        "n_patients"
    ]
)


cox_n_events = int(
    cox_summary_row[
        "n_events"
    ]
)


cox_n_censored = int(
    cox_summary_row[
        "n_censored"
    ]
)


cox_c_index = float(
    cox_summary_row[
        "c_index"
    ]
)


cox_n_parameters = int(
    cox_summary_row[
        "n_model_parameters"
    ]
)


print(
    "\n"
    + "=" * 70
)

print(
    "3. CLINICAL COX MODEL"
)

print(
    "=" * 70
)

print(
    f"Patients: "
    f"{cox_n_patients}"
)

print(
    f"Events: "
    f"{cox_n_events}"
)

print(
    f"Censored: "
    f"{cox_n_censored}"
)

print(
    f"C-index: "
    f"{cox_c_index:.4f}"
)

print(
    f"Model parameters: "
    f"{cox_n_parameters}"
)


# 5. COX FEATURE RESULTS
print(
    "\n"
    + "=" * 70
)

print(
    "4. COX HAZARD RATIOS"
)

print(
    "=" * 70
)


# Add feature name as an explicit column
cox_feature_results = cox_results.copy()


cox_feature_results.index.name = (
    "feature"
)


cox_feature_results = (
    cox_feature_results
    .reset_index()
)


# Rename columns if they exist
rename_map = {}

if "exp(coef)" in cox_feature_results.columns:

    rename_map[
        "exp(coef)"
    ] = "hazard_ratio"


if "p" in cox_feature_results.columns:

    rename_map[
        "p"
    ] = "p_value"


cox_feature_results = (
    cox_feature_results
    .rename(
        columns=rename_map
    )
)

#cox analysis already has this column so after the rename it would be duplicated. 
cox_feature_results = cox_feature_results.loc[
    :, ~cox_feature_results.columns.duplicated()
]


print(
    cox_feature_results.to_string(
        index=False
    )
)


# 6. THRESHOLD LOG-RANK
if threshold_logrank.empty:

    raise ValueError(
        "\nThreshold log-rank results are empty."
    )


threshold_logrank_row = (
    threshold_logrank.iloc[0]
)


threshold_logrank_statistic = float(
    threshold_logrank_row[
        "logrank_test_statistic"
    ]
)


threshold_logrank_p = float(
    threshold_logrank_row[
        "p_value"
    ]
)


threshold_group_0_n = int(
    threshold_logrank_row[
        "group_0_n"
    ]
)


threshold_group_1_n = int(
    threshold_logrank_row[
        "group_1_n"
    ]
)


print(
    "\n"
    + "=" * 70
)

print(
    "5. THRESHOLD-BASED LOG-RANK"
)

print(
    "=" * 70
)

print(
    f"Group <= {selected_threshold} days: "
    f"{threshold_group_0_n}"
)

print(
    f"Group > {selected_threshold} days: "
    f"{threshold_group_1_n}"
)

print(
    f"Log-rank statistic: "
    f"{threshold_logrank_statistic:.4f}"
)

print(
    f"p-value: "
    f"{threshold_logrank_p:.6f}"
)


# 7. COX RISK GROUPS
if cox_risk_logrank.empty:

    raise ValueError(
        "\nCox risk-group log-rank results are empty."
    )


cox_risk_row = (
    cox_risk_logrank.iloc[0]
)


risk_threshold = float(
    cox_risk_row[
        "risk_threshold"
    ]
)


low_risk_n = int(
    cox_risk_row[
        "low_risk_n"
    ]
)


high_risk_n = int(
    cox_risk_row[
        "high_risk_n"
    ]
)


low_risk_events = int(
    cox_risk_row[
        "low_risk_events"
    ]
)


high_risk_events = int(
    cox_risk_row[
        "high_risk_events"
    ]
)


cox_risk_logrank_statistic = float(
    cox_risk_row[
        "logrank_test_statistic"
    ]
)


cox_risk_logrank_p = float(
    cox_risk_row[
        "logrank_p_value"
    ]
)


print(
    "\n"
    + "=" * 70
)

print(
    "6. COX-DERIVED RISK GROUPS"
)

print(
    "=" * 70
)

print(
    f"Median risk threshold: "
    f"{risk_threshold:.6f}"
)

print(
    f"Low-risk patients: "
    f"{low_risk_n}"
)

print(
    f"High-risk patients: "
    f"{high_risk_n}"
)

print(
    f"Low-risk events: "
    f"{low_risk_events}"
)

print(
    f"High-risk events: "
    f"{high_risk_events}"
)

print(
    f"Log-rank statistic: "
    f"{cox_risk_logrank_statistic:.4f}"
)

print(
    f"Log-rank p-value: "
    f"{cox_risk_logrank_p:.6f}"
)


# 8. FINAL FEATURE RESULTS TABLE
"""
Create a clean feature-level table containing:

    Feature
    Hazard Ratio
    95% CI
    p-value
    significance indicator
"""


final_feature_table = cox_feature_results.copy()


# Find CI columns generated by script 7
if (
    "ci_lower" in final_feature_table.columns
    and
    "ci_upper" in final_feature_table.columns
):

    final_feature_table[
        "95% CI"
    ] = (
        final_feature_table[
            "ci_lower"
        ].round(4).astype(str)
        + " - "
        + final_feature_table[
            "ci_upper"
        ].round(4).astype(str)
    )


# Significance
if "p_value" in final_feature_table.columns:

    final_feature_table[
        "significant_p_lt_0.05"
    ] = (
        final_feature_table[
            "p_value"
        ] < 0.05
    )


# Keep only useful columns
preferred_columns = [
    "feature",
    "hazard_ratio",
    "95% CI",
    "p_value",
    "significant_p_lt_0.05",
]


available_final_columns = [
    column
    for column in preferred_columns
    if column in final_feature_table.columns
]


final_feature_table = final_feature_table[
    available_final_columns
].copy()


# SAVE FINAL FEATURE TABLE
final_feature_table.to_csv(
    FINAL_RESULTS_FILE,
    index=False
)


# 9. CREATE OVERALL MODEL SUMMARY
final_model_summary = pd.DataFrame(
    [
        {
            "analysis": (
                "Clinical-only Cox "
                "Proportional Hazards"
            ),
            "n_patients": cox_n_patients,
            "n_events": cox_n_events,
            "n_censored": cox_n_censored,
            "c_index": cox_c_index,

            "naive_bayes_threshold_days":
                selected_threshold,

            "naive_bayes_threshold_balanced_accuracy":
                best_threshold_balanced_accuracy,

            "smotenc_accuracy":
                classification_accuracy,

            "smotenc_balanced_accuracy":
                classification_balanced_accuracy,

            "smotenc_recall":
                classification_recall,

            "smotenc_f1":
                classification_f1,

            "smotenc_roc_auc":
                classification_roc_auc,

            "threshold_logrank_statistic":
                threshold_logrank_statistic,

            "threshold_logrank_p_value":
                threshold_logrank_p,

            "cox_risk_low_n":
                low_risk_n,

            "cox_risk_high_n":
                high_risk_n,

            "cox_risk_logrank_statistic":
                cox_risk_logrank_statistic,

            "cox_risk_logrank_p_value":
                cox_risk_logrank_p,
        }
    ]
)


final_model_summary.to_csv(
    FINAL_MODEL_SUMMARY_FILE,
    index=False
)


# FINAL OUTPUT
print(
    "\n"
    + "=" * 70
)

print(
    "RESULTS"
)

print(
    "=" * 70
)


print(
    "\nFinal clinical feature table:"
)

print(
    final_feature_table.to_string(
        index=False
    )
)


print(
    "\n"
    + "-" * 70
)

print(
    "Overall model summary:"
)

print(
    final_model_summary.to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 70
)

print(
    "RESULTS"
)

print(
    "=" * 70
)


print(
    "\nFiles saved:"
)

print(
    f" - {FINAL_RESULTS_FILE}"
)

print(
    f" - {FINAL_MODEL_SUMMARY_FILE}"
)


print(
    "\nDone."
)
