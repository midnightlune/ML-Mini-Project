"""
Tests different survival-time thresholds and selects the threshold
that gives the best balanced accuracy using clinical features.

Important:
- Only patients with confirmed death are used for threshold classification.
- Features that are 100% missing are removed.
- Preprocessing is performed inside the ML pipeline to avoid leakage.
- OneHotEncoder outputs dense data because GaussianNB requires dense input.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clinical_survival_clean.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "naive_bayes_threshold_results.csv"
)

REMOVED_FEATURES_FILE = (
    OUTPUT_DIR
    / "removed_clinical_features.csv"
)


# Create results directory if it does not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# SETTINGS

# Survival thresholds to test
THRESHOLDS = [240, 270, 300, 330, 360]

# Random seed for reproducibility
RANDOM_STATE = 42

# Number of cross-validation folds
N_SPLITS = 5


# LOAD DATA

print("NAIVE BAYES SURVIVAL THRESHOLD SELECTION")
print("=" * 70)

print("\nLoading clinical survival data...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}\n\n"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Original dataset shape: {df.shape}")


# CLINICAL FEATURES

# These are the candidate clinical variables.
# Some may be completely missing in the CPTAC-PDA dataset.
candidate_features = [
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


# Keep only columns that actually exist
available_features = [
    feature
    for feature in candidate_features
    if feature in df.columns
]

missing_columns_from_dataset = [
    feature
    for feature in candidate_features
    if feature not in df.columns
]

if missing_columns_from_dataset:
    print("\nCandidate features not found in dataset:")
    for feature in missing_columns_from_dataset:
        print(f" - {feature}")


# REMOVE 100% MISSING FEATURES

work = df.copy()

features_used = []
features_removed = []

for feature in available_features:

    # Check whether the entire column is missing
    if work[feature].isna().all():

        features_removed.append(feature)

    else:

        features_used.append(feature)


print("\nFeatures used by Naive Bayes:")

for feature in features_used:
    print(f" - {feature}")


print("\nFeatures removed because they are 100% missing:")

if features_removed:

    for feature in features_removed:
        print(f" - {feature}")

else:

    print(" - None")


# SAVE REMOVED FEATURES REPORT

removed_rows = []

for feature in features_removed:

    removed_rows.append(
        {
            "feature": feature,
            "reason": "100% missing",
            "missing_count": int(work[feature].isna().sum()),
            "total_rows": int(len(work)),
        }
    )


# Also record candidate columns that were not present
for feature in missing_columns_from_dataset:

    removed_rows.append(
        {
            "feature": feature,
            "reason": "Column not present in dataset",
            "missing_count": np.nan,
            "total_rows": int(len(work)),
        }
    )


removed_df = pd.DataFrame(removed_rows)

removed_df.to_csv(
    REMOVED_FEATURES_FILE,
    index=False
)

print(
    f"\nRemoved-feature report saved to:\n"
    f"{REMOVED_FEATURES_FILE}"
)


# CHECK FEATURES

if not features_used:

    raise ValueError(
        "\nNo usable clinical features remain."
    )


# CHECK SURVIVAL VARIABLES

required_columns = [
    "survival_time_days",
    "survival_event",
]

for column in required_columns:

    if column not in work.columns:

        raise ValueError(
            f"\nRequired column '{column}' was not found."
        )


# Make sure survival variables are numeric
work["survival_time_days"] = pd.to_numeric(
    work["survival_time_days"],
    errors="coerce"
)

work["survival_event"] = pd.to_numeric(
    work["survival_event"],
    errors="coerce"
)


# Remove records with missing survival information
work = work.dropna(
    subset=[
        "survival_time_days",
        "survival_event",
    ]
).copy()


# USE ONLY PATIENTS WITH CONFIRMED DEATH

"""
For this classification task, we need a known survival time
relative to death.

Therefore:
    survival_event = 1
is used.

Patients who are alive/censored do not have a confirmed
death time and are therefore not used for this threshold
classification.
"""

classification_df = work[
    work["survival_event"] == 1
].copy()


print(
    "\nPatients available for threshold classification: "
    f"{len(classification_df)}"
)


if len(classification_df) == 0:

    raise ValueError(
        "\nNo patients with confirmed death were found."
    )


# PREPARE FEATURES
X = classification_df[features_used].copy()

survival_time = classification_df[
    "survival_time_days"
].copy()


# IDENTIFY NUMERIC AND CATEGORICAL FEATURES
numeric_features = []
categorical_features = []

for feature in features_used:

    if pd.api.types.is_numeric_dtype(X[feature]):

        numeric_features.append(feature)

    else:

        categorical_features.append(feature)


print("\nNumeric features:")
for feature in numeric_features:
    print(f" - {feature}")


print("\nCategorical features:")
for feature in categorical_features:
    print(f" - {feature}")


# PREPROCESSING

# Numeric:
# 1. Replace missing values with median
# 2. Standardize values

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ]
)


# Categorical:
# 1. Replace missing values with most frequent value
# 2. Convert categories into numerical dummy variables
#
# IMPORTANT:
# sparse_output=False is required because GaussianNB
# requires a dense matrix.

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ),
    ]
)


# COMBINE PREPROCESSING

transformers = []

if numeric_features:

    transformers.append(
        (
            "numeric",
            numeric_transformer,
            numeric_features
        )
    )


if categorical_features:

    transformers.append(
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    )


preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# NAIVE BAYES MODEL

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            GaussianNB()
        ),
    ]
)


# CROSS-VALIDATION SETUP

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# TEST SURVIVAL THRESHOLDS

results = []


for threshold in THRESHOLDS:

    print("\n" + "-" * 70)

    print(
        f"Testing threshold: {threshold} days"
    )

    print("-" * 70)


    #CREATE BINARY TARGET

    # Class 0:
    # Survival <= threshold

    # Class 1:
    # Survival > threshold

    # This is only for patients with confirmed death.

    y = (
        survival_time > threshold
    ).astype(int)


    # CHECK CLASS DISTRIBUTION
    class_counts = y.value_counts()

    print(
        f"Class 0 (<= {threshold} days): "
        f"{class_counts.get(0, 0)}"
    )

    print(
        f"Class 1 (> {threshold} days): "
        f"{class_counts.get(1, 0)}"
    )


    # Need both classes
    if len(class_counts) < 2:

        print(
            "Skipping threshold because only one class exists."
        )

        results.append(
            {
                "threshold_days": threshold,
                "balanced_accuracy": np.nan,
                "status": "Skipped - one class only",
            }
        )

        continue


    # CHECK MINIMUM CLASS SIZE
    # --------------------------------------------------------

    minimum_class_size = class_counts.min()

    if minimum_class_size < N_SPLITS:

        print(
            f"Skipping threshold because the smallest class "
            f"has only {minimum_class_size} samples."
        )

        results.append(
            {
                "threshold_days": threshold,
                "balanced_accuracy": np.nan,
                "status": "Skipped - insufficient samples",
            }
        )

        continue

    # CROSS-VALIDATION
    # --------------------------------------------------------

    fold_scores = []


    for fold, (train_index, test_index) in enumerate(
        cv.split(X, y),
        start=1
    ):

        X_train = X.iloc[train_index]

        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]

        y_test = y.iloc[test_index]


        
        # FIT MODEL
        model.fit(
            X_train,
            y_train
        )

        # PREDICT
        y_pred = model.predict(
            X_test
        )


        # BALANCED ACCURACY
        score = balanced_accuracy_score(
            y_test,
            y_pred
        )

        fold_scores.append(score)


        print(
            f"Fold {fold}: "
            f"balanced accuracy = {score:.4f}"
        )


    # MEAN CV SCORE
    mean_score = np.mean(
        fold_scores
    )

    std_score = np.std(
        fold_scores
    )


    print(
        f"\nMean balanced accuracy: "
        f"{mean_score:.4f}"
    )

    print(
        f"Standard deviation: "
        f"{std_score:.4f}"
    )


    results.append(
        {
            "threshold_days": threshold,
            "balanced_accuracy": mean_score,
            "std_balanced_accuracy": std_score,
            "status": "Completed",
        }
    )


# RESULTS DATAFRAME

results_df = pd.DataFrame(
    results
)


# SELECT BEST THRESHOLD
valid_results = results_df[
    results_df["balanced_accuracy"].notna()
].copy()


if valid_results.empty:

    raise ValueError(
        "\nNo valid threshold results were produced."
    )


best_index = valid_results[
    "balanced_accuracy"
].idxmax()


best_threshold = int(
    valid_results.loc[
        best_index,
        "threshold_days"
    ]
)


best_score = float(
    valid_results.loc[
        best_index,
        "balanced_accuracy"
    ]
)


# Add selected flag
results_df["selected"] = (
    results_df["threshold_days"]
    == best_threshold
)


# SAVE RESULTS

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# PRINT FINAL RESULTS

print("\n")
print("=" * 70)
print("NAIVE BAYES THRESHOLD RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


print(
    f"SELECTED SURVIVAL THRESHOLD: "
    f"{best_threshold} days"
)

print(
    f"BEST BALANCED ACCURACY: "
    f"{best_score:.4f}"
)

print("=" * 70)


print(
    f"\nResults saved to:\n"
    f"{OUTPUT_FILE}"
)

print("\nDone.")