import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler


# PATHS

INPUT_PATH = Path(
    "data/processed/clinical_survival_clean.csv"
)

OUTPUT_PATH = Path(
    "data/processed/clinical_preprocessed.csv"
)

REMOVED_PATH = Path(
    "results/removed_clinical_features.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

REMOVED_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# LOAD DATA
df = pd.read_csv(INPUT_PATH)

print("FINAL CLINICAL PREPROCESSING")
print("=" * 70)

print("\nOriginal dataset shape:")
print(df.shape)


# CLINICAL FEATURES
clinical_features = [
    "Age_years",
    "Sex",
    "Race",
    "Ethnicity",
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
]


# Only keep columns that actually exist
available_features = [
    column
    for column in clinical_features
    if column in df.columns
]


print("\nClinical variables found in dataset:")

for column in available_features:
    print(" -", column)


# CREATE WORKING DATASET
required_columns = [
    "Case Submitter ID",
    "survival_time_days",
    "survival_event"
]

columns = (
    required_columns +
    available_features
)

work = df[columns].copy()


# REMOVE COMPLETELY EMPTY FEATURES
feature_columns = [
    column
    for column in work.columns
    if column not in required_columns
]

all_missing_features = [
    column
    for column in feature_columns
    if work[column].isna().all()
]


print("\nCompletely missing clinical variables:")

if all_missing_features:

    for column in all_missing_features:
        print(
            f" - {column} "
            "(100% missing)"
        )

else:

    print("None")


# Remove them from the modeling dataset
work = work.drop(
    columns=all_missing_features
)


# Save information about removed variables
removed_records = []

for column in all_missing_features:

    removed_records.append({

        "feature": column,

        "reason":
            "100% missing in CPTAC-PDA clinical cohort",

        "missing_count":
            int(len(df)),

        "missing_percentage":
            100.0
    })


removed_df = pd.DataFrame(
    removed_records
)

removed_df.to_csv(
    REMOVED_PATH,
    index=False
)


# NUMERICAL VARIABLES
numeric_candidates = [
    "Age_years",
    "Lymph Nodes Positive",
    "Lymph Nodes Tested"
]


numeric_features = [
    column
    for column in numeric_candidates
    if column in work.columns
]


for column in numeric_features:

    work[column] = pd.to_numeric(
        work[column],
        errors="coerce"
    )


# CATEGORICAL VARIABLES
categorical_candidates = [
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
    "Margins Involved Site"
]


categorical_features = [
    column
    for column in categorical_candidates
    if column in work.columns
]


# REMOVE CATEGORICAL FEATURES THAT BECAME EMPTY
second_missing_check = [
    column
    for column in (
        numeric_features +
        categorical_features
    )
    if work[column].isna().all()
]


if second_missing_check:

    print(
        "\nAdditional completely empty "
        "variables detected:"
    )

    for column in second_missing_check:

        print(
            f" - {column}"
        )

    work = work.drop(
        columns=second_missing_check
    )

    numeric_features = [
        column
        for column in numeric_features
        if column not in second_missing_check
    ]

    categorical_features = [
        column
        for column in categorical_features
        if column not in second_missing_check
    ]


# NUMERICAL IMPUTATION
for column in numeric_features:

    median_value = work[column].median()

    if pd.isna(median_value):

        print(
            f"WARNING: {column} "
            "has no usable median."
        )

    else:

        work[column] = work[column].fillna(
            median_value
        )


# CATEGORICAL IMPUTATION
for column in categorical_features:

    work[column] = work[column].fillna(
        "Unknown"
    )


# ENCODE CATEGORICAL VARIABLES
if categorical_features:

    work = pd.get_dummies(
        work,
        columns=categorical_features,
        drop_first=True,
        dtype=int
    )


# STANDARDIZE NUMERICAL FEATURES

numeric_features_after_encoding = [
    column
    for column in numeric_features
    if column in work.columns
]


if numeric_features_after_encoding:

    scaler = StandardScaler()

    work[
        numeric_features_after_encoding
    ] = scaler.fit_transform(
        work[numeric_features_after_encoding]
    )


# FINAL SAFETY CHECK

print(
    "\nRemaining missing values:"
)

missing = work.isna().sum()

missing = missing[
    missing > 0
]

if len(missing) == 0:

    print(
        "None"
    )

else:

    print(missing)


#SAVE
work.to_csv(
    OUTPUT_PATH,
    index=False
)


# SUMMARY
print()
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(
    "\nFinal dataset shape:"
)

print(
    work.shape
)

print(
    "\nFinal columns:"
)

for column in work.columns:

    print(
        " -",
        column
    )

print(
    "\nRemoved completely missing variables:"
)

if all_missing_features:

    for column in all_missing_features:

        print(
            " -",
            column
        )

else:

    print(
        "None"
    )

print(
    "\nSaved cleaned dataset:"
)

print(
    OUTPUT_PATH
)

print(
    "\nSaved removed-feature report:"
)

print(
    REMOVED_PATH
)