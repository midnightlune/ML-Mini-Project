"""
1_datamain.py

Missing-data report and exploratory analysis of the clinical dataset.

Outputs (in results/):
    - missing_data_report.csv
    - age_distribution.png
    - tumor_grade_distribution.png
    - survival_time_distribution.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "clinical_survival_clean.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MISSING_REPORT_FILE = RESULTS_DIR / "missing_data_report.csv"


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nClinical survival file not found:\n{INPUT_FILE}\n\n"
        "Run the data-cleaning script that creates "
        "clinical_survival_clean.csv first."
    )

df = pd.read_csv(INPUT_FILE)


# MISSING DATA REPORT
report = pd.DataFrame({
    "variable": df.columns,
    "missing_count": df.isna().sum().values,
    "total": len(df),
})
report["missing_percentage"] = report["missing_count"] / report["total"] * 100
report = report.sort_values("missing_percentage", ascending=False)

print(report.to_string(index=False))
report.to_csv(MISSING_REPORT_FILE, index=False)
print(f"\nSaved: {MISSING_REPORT_FILE}")


# BASIC SUMMARY
print("\nCLINICAL DATA SUMMARY")
print("=" * 60)
print(f"\nNumber of patients: {len(df)}")

if "Age_years" in df.columns:
    print("\nAge statistics:")
    print(df["Age_years"].describe())

for column in ("Sex", "Tumor Grade", "AJCC Pathologic Stage", "survival_event"):
    if column in df.columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))


def save_current_figure(name):
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / name, dpi=300)
    plt.close()


# AGE DISTRIBUTION
if "Age_years" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df["Age_years"].dropna(), bins=15)
    plt.xlabel("Age at Diagnosis (years)")
    plt.ylabel("Number of Patients")
    plt.title("Age Distribution")
    save_current_figure("age_distribution.png")


# TUMOR GRADE
if "Tumor Grade" in df.columns:
    plt.figure(figsize=(8, 5))
    df["Tumor Grade"].value_counts().plot(kind="bar")
    plt.xlabel("Tumor Grade")
    plt.ylabel("Number of Patients")
    plt.title("Tumor Grade Distribution")
    save_current_figure("tumor_grade_distribution.png")


# SURVIVAL TIME
if "survival_time_months" in df.columns:
    months = df["survival_time_months"]
elif "survival_time_days" in df.columns:
    months = df["survival_time_days"] / 30.44
else:
    months = None

if months is not None:
    plt.figure(figsize=(8, 5))
    plt.hist(months.dropna(), bins=20)
    plt.xlabel("Survival / Follow-up Time (months)")
    plt.ylabel("Number of Patients")
    plt.title("Survival Time Distribution")
    save_current_figure("survival_time_distribution.png")


print("\nEDA complete.")