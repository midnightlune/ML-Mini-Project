# MISSING DATA
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_PATH = Path(
    "data/processed/clinical_survival_clean.csv"
)

OUTPUT_PATH = Path(
    "results/missing_data_report.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(INPUT_PATH)

report = pd.DataFrame({
    "variable": df.columns,
    "missing_count": df.isna().sum(),
    "total": len(df),
})

report["missing_percentage"] = (
    report["missing_count"] /
    report["total"] * 100
)

report = report.sort_values(
    "missing_percentage",
    ascending=False
)


print(report.to_string(index=False))

report.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)

# EXPLORATORY ANALYSIS

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# BASIC SUMMARY
print("CLINICAL DATA SUMMARY")
print("=" * 60)

print("\nNumber of patients:")
print(len(df))

print("\nAge statistics:")
print(df["Age_years"].describe())

print("\nSex:")
print(df["Sex"].value_counts())

print("\nTumor Grade:")
print(df["Tumor Grade"].value_counts())

print("\nAJCC Pathologic Stage:")
print(df["AJCC Pathologic Stage"].value_counts())

print("\nSurvival event:")
print(df["survival_event"].value_counts())

# AGE DISTRIBUTION
plt.figure(figsize=(8, 5))

plt.hist(
    df["Age_years"].dropna(),
    bins=15
)

plt.xlabel("Age at Diagnosis (years)")
plt.ylabel("Number of Patients")
plt.title("Age Distribution")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "age_distribution.png",
    dpi=300
)

#plt.show()

# AGE DISTRIBUTION
plt.figure(figsize=(8, 5))

plt.hist(
    df["Age_years"].dropna(),
    bins=15
)

plt.xlabel("Age at Diagnosis (years)")
plt.ylabel("Number of Patients")
plt.title("Age Distribution")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "age_distribution.png",
    dpi=300
)

#plt.show()

# TUMOR GRADE

grade_counts = df["Tumor Grade"].value_counts()

plt.figure(figsize=(8, 5))

grade_counts.plot(
    kind="bar"
)

plt.xlabel("Tumor Grade")
plt.ylabel("Number of Patients")
plt.title("Tumor Grade Distribution")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "tumor_grade_distribution.png",
    dpi=300
)

#plt.show()

# SURVIVAL TIME

plt.figure(figsize=(8, 5))

plt.hist(
    df["survival_time_months"],
    bins=20
)

plt.xlabel("Survival / Follow-up Time (months)")
plt.ylabel("Number of Patients")
plt.title("Survival Time Distribution")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "survival_time_distribution.png",
    dpi=300
)

#plt.show()

print("\nEDA complete.")