"""
2_kaplanmeier.py

Kaplan-Meier survival curves: overall and by tumor grade.

Outputs (in results/):
    - kaplan_meier_overall_survival.png
    - kaplan_meier_by_tumor_grade.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "clinical_survival_clean.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nClinical survival file not found:\n{INPUT_FILE}\n\n"
        "Run the data-cleaning script that creates "
        "clinical_survival_clean.csv first."
    )

df = pd.read_csv(INPUT_FILE)

for column in ("survival_time_days", "survival_event"):
    if column not in df.columns:
        raise ValueError(f"Required column missing: {column}")
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=["survival_time_days", "survival_event"])
df = df[df["survival_time_days"] > 0].copy()


# OVERALL SURVIVAL
kmf = KaplanMeierFitter()
kmf.fit(df["survival_time_days"], df["survival_event"], label="All patients")

plt.figure(figsize=(9, 6))
kmf.plot_survival_function()
plt.title("Kaplan-Meier Overall Survival Curve")
plt.xlabel("Time (days)")
plt.ylabel("Survival Probability")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "kaplan_meier_overall_survival.png", dpi=300)
plt.close()

median = kmf.median_survival_time_

if np.isinf(median):
    print("Median survival not reached (curve never drops below 0.5).")
else:
    print(f"Estimated median survival: {median:.0f} days "
          f"({median / 30.44:.1f} months)")


# BY TUMOR GRADE
if "Tumor Grade" in df.columns:

    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    for grade in sorted(df["Tumor Grade"].dropna().unique()):

        group = df[df["Tumor Grade"] == grade]

        if len(group) < 2:
            continue

        grade_kmf = KaplanMeierFitter()
        grade_kmf.fit(
            group["survival_time_days"],
            group["survival_event"],
            label=f"Tumor Grade {grade} (n={len(group)})",
        )
        grade_kmf.plot_survival_function(ax=ax)

    plt.title("Kaplan-Meier Survival by Tumor Grade")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "kaplan_meier_by_tumor_grade.png", dpi=300)
    plt.close()

print("Done.")