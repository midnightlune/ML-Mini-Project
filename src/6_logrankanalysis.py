"""
8_logrankanalysis.py

Log-rank survival analysis by baseline clinical characteristics.

    Here, groups are defined only by information known at diagnosis
    (grade, stage, sex, age split at the median, ...), so the log-rank
    test answers a real question: does survival differ across these groups?

    Censored patients are retained.

Outputs:
    - results/logrank_clinical_results.csv
    - results/kaplan_meier_by_<feature>.png  (for raw p < 0.05)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "clinical_survival_clean.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = RESULTS_DIR / "logrank_clinical_results.csv"

MIN_GROUP_SIZE = 5

CATEGORICAL_FEATURES = [
    "Sex",
    "Race",
    "Ethnicity",
    "Tumor Grade",
    "AJCC Pathologic Stage",
    "AJCC Pathologic T",
    "AJCC Pathologic N",
    "AJCC Pathologic M",
    "Metastasis At Diagnosis",
    "Lymphatic Invasion Present",
    "Perineural Invasion Present",
    "Vascular Invasion Present",
    "Margins Involved Site",
]

# Numeric variables split at the median
MEDIAN_SPLIT_FEATURES = [
    "Age_years",
    "Lymph Nodes Positive",
    "Lymph Nodes Tested",
]


print("=" * 70)
print("LOG-RANK ANALYSIS BY BASELINE CLINICAL GROUPS")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Clinical survival file not found:\n{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

for column in ("survival_time_days", "survival_event"):
    if column not in df.columns:
        raise ValueError(f"Required column missing: {column}")
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=["survival_time_days", "survival_event"])
df = df[df["survival_time_days"] > 0].copy()

print(f"Patients with usable survival information: {len(df)}")
print(f"Events: {int(df['survival_event'].sum())}")


def make_groups(data, feature):
    """Return a Series of group labels (NaN where unknown)."""
    values = data[feature]
    groups = pd.Series(pd.NA, index=data.index, dtype="object")

    if feature in MEDIAN_SPLIT_FEATURES:
        values = pd.to_numeric(values, errors="coerce")
        median = values.median()
        groups[values <= median] = f"<= {median:g}"
        groups[values > median] = f"> {median:g}"
    else:
        known = values.notna()
        groups[known] = values[known].astype(str)

    return groups


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


rows = []
plot_data = {}

for feature in CATEGORICAL_FEATURES + MEDIAN_SPLIT_FEATURES:

    if feature not in df.columns:
        continue

    groups = make_groups(df, feature)

    counts = groups.value_counts()
    keep = counts[counts >= MIN_GROUP_SIZE].index
    mask = groups.isin(keep)

    if len(keep) < 2:
        print(f"Skipping {feature}: fewer than 2 groups with >= "
              f"{MIN_GROUP_SIZE} patients")
        continue

    subset = df[mask]
    subset_groups = groups[mask]

    try:
        result = multivariate_logrank_test(
            subset["survival_time_days"],
            subset_groups,
            subset["survival_event"],
        )
    except Exception as error:
        print(f"Skipping {feature}: {error}")
        continue

    rows.append({
        "feature": feature,
        "n_patients": len(subset),
        "n_events": int(subset["survival_event"].sum()),
        "n_groups": len(keep),
        "group_sizes": "; ".join(f"{g} (n={counts[g]})" for g in keep),
        "test_statistic": float(result.test_statistic),
        "degrees_of_freedom": int(result.degrees_of_freedom),
        "p_value": float(result.p_value),
    })

    plot_data[feature] = (subset, subset_groups)


if not rows:
    raise ValueError("No feature produced a valid log-rank test.")

results = pd.DataFrame(rows)
results["p_value_fdr"] = benjamini_hochberg(results["p_value"])
results["significant_raw_p_lt_0.05"] = results["p_value"] < 0.05
results["significant_fdr_lt_0.05"] = results["p_value_fdr"] < 0.05
results = results.sort_values("p_value").reset_index(drop=True)

print("\n" + "=" * 70)
print("LOG-RANK RESULTS (sorted by p-value)")
print("=" * 70)
print(
    results[
        ["feature", "n_patients", "n_groups",
         "test_statistic", "p_value", "p_value_fdr"]
    ].to_string(index=False)
)

print(
    "\nNote: many features are tested, so use the FDR-adjusted p-value "
    "(p_value_fdr) for conclusions. Raw p < 0.05 alone is expected to "
    "occur by chance for roughly 1 in 20 features."
)


# KAPLAN-MEIER PLOTS FOR NOMINALLY SIGNIFICANT FEATURES
saved_plots = []

for feature in results.loc[results["p_value"] < 0.05, "feature"]:

    subset, subset_groups = plot_data[feature]
    p_value = float(results.loc[results["feature"] == feature, "p_value"].iloc[0])

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for label in sorted(subset_groups.unique()):
        member = subset_groups == label
        kmf = KaplanMeierFitter()
        kmf.fit(
            subset.loc[member, "survival_time_days"],
            subset.loc[member, "survival_event"],
            label=f"{label} (n={int(member.sum())})",
        )
        kmf.plot_survival_function(ax=ax)

    plt.title(f"Kaplan-Meier by {feature} (log-rank p = {p_value:.4f})")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", feature).strip("_").lower()
    plot_path = RESULTS_DIR / f"kaplan_meier_by_{safe_name}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    saved_plots.append(plot_path)


results.to_csv(RESULT_FILE, index=False)

print("\nFiles saved:")
print(f" - {RESULT_FILE}")
for path in saved_plots:
    print(f" - {path}")
print("\nDone.")