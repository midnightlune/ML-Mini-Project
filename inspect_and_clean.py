import pandas as pd
from pathlib import Path
import numpy as np

INPUT_PATH = Path("data/raw/PDC_clinical_manifest_10032026_200348.csv")

df = pd.read_csv(INPUT_PATH)

print("CPTAC-PDA CLINICAL DATASET")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nNumber of patients/records:")
print(len(df))

#Column names
print("\nColumns:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i:3}. {column}")

#Show first 5 rows
print("\nFirst 5 rows:")
print(df.head())

#Show data types
print("\nData types:")
print(df.dtypes)

# Missing values
missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percentage": df.isna().mean() * 100
})

missing = missing.sort_values(
    "missing_percentage",
    ascending=False
)

print("\nMissing values:")
print(missing.head(30))


#Important clinical variables
important_columns = [
    "Case Submitter ID",
    "Sex",
    "Race",
    "Ethnicity",
    "Age at Diagnosis",
    "Tumor Grade",
    "AJCC Pathologic Stage",
    "AJCC Pathologic T",
    "AJCC Pathologic N",
    "AJCC Pathologic M",
    "Lymph Nodes Positive",
    "Lymph Nodes Tested",
    "Metastasis At Diagnosis",
    "Lymphatic Invasion Present",
    "Perineural Invasion Present",
    "Vascular Invasion Present",
    "Margins Involved Site",
    "Overall Survival",
    "Days to Death",
    "Days to Last Follow Up",
    "Vital Status",
]

print("\nImportant columns:")
print(df[important_columns].head(10))

#Survival Status
print("\nVital Status values:")
print(df["Vital Status"].value_counts(dropna=False))


OUTPUT_PATH = Path(
    "data/processed/clinical_survival_clean.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("Original dataset:")
print(df.shape)

#STANDARISING EMPTY VALUES

# Convert blank strings and common "missing" labels to NaN
missing_values = [
    "",
    " ",
    "Not Reported",
    "Unknown",
    "not reported",
    "unknown"
]

df = df.replace(missing_values, np.nan)


#SELECT CLINICAL VARIABLES
columns_to_keep = [
    "Case Submitter ID",
    "Sex",
    "Race",
    "Ethnicity",
    "Age at Diagnosis",
    "Tumor Grade",
    "AJCC Pathologic Stage",
    "AJCC Pathologic T",
    "AJCC Pathologic N",
    "AJCC Pathologic M",
    "Lymph Nodes Positive",
    "Lymph Nodes Tested",
    "Metastasis At Diagnosis",
    "Metastasis At Diagnosis Site",
    "Lymphatic Invasion Present",
    "Perineural Invasion Present",
    "Vascular Invasion Present",
    "Vascular Invasion Type",
    "Margins Involved Site",
    "Tumor Regression Grade",
    "Days to Death",
    "Days to Last Follow Up",
    "Vital Status",
]

df = df[columns_to_keep].copy()

#CLEAN PATIENT ID
df["Case Submitter ID"] = (
    df["Case Submitter ID"]
    .astype(str)
    .str.strip()
)

#AGE FROM DAYS TO YEARS
df["Age at Diagnosis"] = pd.to_numeric(
    df["Age at Diagnosis"],
    errors="coerce"
)

df["Age_years"] = (
    df["Age at Diagnosis"] / 365.25
)


# CONVERT SURVIVAL VARIABLES TO NUMERIC
df["Days to Death"] = pd.to_numeric(
    df["Days to Death"],
    errors="coerce"
)

df["Days to Last Follow Up"] = pd.to_numeric(
    df["Days to Last Follow Up"],
    errors="coerce"
)


#CREATING SURVIVAL EVENT
df["survival_event"] = np.nan

df.loc[
    df["Vital Status"].str.lower() == "dead",
    "survival_event"
] = 1

df.loc[
    df["Vital Status"].str.lower() == "alive",
    "survival_event"
] = 0

# CREATE SURVIVAL TIME

# If dead:
# use Days to Death
#
# If alive:
# use Days to Last Follow Up

df["survival_time_days"] = np.where(
    df["survival_event"] == 1,
    df["Days to Death"],
    df["Days to Last Follow Up"]
)

# REMOVE PATIENTS WITHOUT USABLE SURVIVAL INFORMATION
before = len(df)

df = df.dropna(
    subset=[
        "survival_event",
        "survival_time_days"
    ]
)

after = len(df)

print("\nPatients before survival filtering:", before)
print("Patients after survival filtering:", after)
print("Patients removed:", before - after)

# REMOVE IMPOSSIBLE SURVIVAL TIMES
df = df[
    df["survival_time_days"] >= 0
].copy()

# CONVERT SURVIVAL TIME TO MONTHS
df["survival_time_months"] = (
    df["survival_time_days"] / 30.44
)


#CHECKING RESULT
print("\nFinal dataset shape:")
print(df.shape)

print("\nSurvival event:")
print(df["survival_event"].value_counts())

print("\nClinical dataset preview:")
print(
    df[
        [
            "Case Submitter ID",
            "Age_years",
            "Sex",
            "Tumor Grade",
            "AJCC Pathologic Stage",
            "survival_time_days",
            "survival_event"
        ]
    ].head(10)
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved cleaned dataset to:")
print(OUTPUT_PATH)