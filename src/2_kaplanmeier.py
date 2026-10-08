import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from lifelines import KaplanMeierFitter

INPUT_PATH = Path(
    "data/processed/clinical_survival_clean.csv"
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(INPUT_PATH)

# KAPLAN-MEIER MODEL

kmf = KaplanMeierFitter()

kmf.fit(
    durations=df["survival_time_days"],
    event_observed=df["survival_event"]
)

# PLOT
plt.figure(figsize=(9, 6))

kmf.plot_survival_function()

plt.title(
    "Kaplan-Meier Overall Survival Curve"
)

plt.xlabel(
    "Time (days)"
)

plt.ylabel(
    "Survival Probability"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "kaplan_meier_overall_survival.png",
    dpi=300
)

#plt.show()

# MEDIAN SURVIVAL
print(
    "Estimated median survival time:",
    kmf.median_survival_time_,
    "days"
)

print(
    "Estimated median survival time:",
    kmf.median_survival_time_ / 30.44,
    "months"
)

#KM BY GRADE
plt.figure(figsize=(9, 6))

for grade in ["G1", "G2", "G3", "G4"]:

    group = df[
        df["Tumor Grade"] == grade
    ]

    if len(group) < 2:
        continue

    kmf.fit(
        durations=group["survival_time_days"],
        event_observed=group["survival_event"],
        label=f"Tumor Grade {grade}"
    )

    kmf.plot_survival_function()

plt.title(
    "Kaplan-Meier Survival by Tumor Grade"
)

plt.xlabel("Time (days)")
plt.ylabel("Survival Probability")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "kaplan_meier_by_tumor_grade.png",
    dpi=300
)

#plt.show()