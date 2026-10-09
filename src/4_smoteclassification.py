"""
SMOTE + Naive Bayes Classification
Purpose:
    Handle class imbalance in the clinical survival
    classification problem using SMOTENC and train
    a Gaussian Naive Bayes classifier.

Important:
    - 100% missing features are removed.
    - Only patients with confirmed death are used.
    - The survival threshold comes from script 5.
    - Train/test split happens BEFORE SMOTENC.
    - SMOTENC is applied ONLY to the training data.
    - The test set is never oversampled.
    - Preprocessing is fitted only on training data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTENC

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clinical_survival_clean.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CLASSIFICATION_RESULTS_FILE = (
    RESULTS_DIR
    / "naive_bayes_classification_results.csv"
)

CLASS_DISTRIBUTION_FILE = (
    RESULTS_DIR
    / "smote_class_distribution.csv"
)

REMOVED_FEATURES_FILE = (
    RESULTS_DIR
    / "removed_clinical_features_smote.csv"
)


# SETTINGS
RANDOM_STATE = 42
TEST_SIZE = 0.20


# HEADER
print("SMOTENC + NAIVE BAYES")
print("=" * 70)


# LOAD DATA

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nClinical dataset not found:\n{INPUT_FILE}\n\n"
    )


df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nOriginal dataset shape: {df.shape}"
)


# GET SELECTED THRESHOLD FROM SCRIPT 5
threshold_file = (
    RESULTS_DIR
    / "naive_bayes_threshold_results.csv"
)


if not threshold_file.exists():

    raise FileNotFoundError(
        "\nThreshold results were not found:\n"
        f"{threshold_file}\n\n"
        "Run 5_naivebayesthreshold.py first."
    )


threshold_results = pd.read_csv(
    threshold_file
)


# Keep only valid threshold results
valid_thresholds = threshold_results[
    threshold_results["balanced_accuracy"].notna()
].copy()


if valid_thresholds.empty:

    raise ValueError(
        "\nNo valid survival threshold was found."
    )


# Select threshold with highest balanced accuracy
best_index = valid_thresholds[
    "balanced_accuracy"
].idxmax()


selected_threshold = int(
    valid_thresholds.loc[
        best_index,
        "threshold_days"
    ]
)


print(
    f"\nSelected threshold: "
    f"{selected_threshold} days"
)


# CHECK REQUIRED SURVIVAL COLUMNS
required_columns = [
    "survival_time_days",
    "survival_event",
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"\nRequired column is missing: {column}"
        )


# CONVERT SURVIVAL VARIABLES
df["survival_time_days"] = pd.to_numeric(
    df["survival_time_days"],
    errors="coerce"
)

df["survival_event"] = pd.to_numeric(
    df["survival_event"],
    errors="coerce"
)


# USE ONLY PATIENTS WITH CONFIRMED DEATH
df = df[
    df["survival_event"] == 1
].copy()


df = df.dropna(
    subset=[
        "survival_time_days"
    ]
).copy()


print(
    f"Patients with confirmed death: "
    f"{len(df)}"
)


# CREATE BINARY SURVIVAL CLASS
"""
Class 0:
    Survival <= selected threshold

Class 1:
    Survival > selected threshold
"""

df["risk_class"] = (
    df["survival_time_days"]
    > selected_threshold
).astype(int)


# CANDIDATE CLINICAL FEATURES
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


# KEEP FEATURES THAT EXIST
available_features = [
    feature
    for feature in candidate_features
    if feature in df.columns
]


missing_from_dataset = [
    feature
    for feature in candidate_features
    if feature not in df.columns
]


if missing_from_dataset:

    print(
        "\nCandidate features not present in dataset:"
    )

    for feature in missing_from_dataset:

        print(
            f" - {feature}"
        )


# REMOVE 100% MISSING FEATURES
features_used = []

features_removed = []


for feature in available_features:

    if df[feature].isna().all():

        features_removed.append(
            feature
        )

    else:

        features_used.append(
            feature
        )


print(
    "\nFeatures used:"
)

for feature in features_used:

    print(
        f" - {feature}"
    )


print(
    "\nFeatures removed because they are 100% missing:"
)

if features_removed:
    for feature in features_removed:
        print(
            f" - {feature}"
        )

else:

    print(
        " - None"
    )


# SAVE REMOVED FEATURE INFORMATION

removed_rows = []


for feature in features_removed:

    removed_rows.append(
        {
            "feature": feature,
            "reason": "100% missing",
            "missing_count": int(
                df[feature].isna().sum()
            ),
            "total_rows": len(df),
        }
    )


for feature in missing_from_dataset:

    removed_rows.append(
        {
            "feature": feature,
            "reason": "Column not present in dataset",
            "missing_count": np.nan,
            "total_rows": len(df),
        }
    )


removed_df = pd.DataFrame(
    removed_rows
)


removed_df.to_csv(
    REMOVED_FEATURES_FILE,
    index=False
)


# CHECK FEATURES

if not features_used:

    raise ValueError(
        "\nNo usable clinical features remain."
    )


# CREATE X AND y

X = df[
    features_used
].copy()

y = df[
    "risk_class"
].copy()


# SHOW ORIGINAL CLASS DISTRIBUTION

print(
    "\nOriginal class distribution:"
)

print(
    y.value_counts()
    .sort_index()
)


# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


print(
    "\nTraining distribution BEFORE SMOTENC:"
)

print(
    y_train.value_counts()
    .sort_index()
)


print(
    "\nTesting distribution:"
)

print(
    y_test.value_counts()
    .sort_index()
)


# IDENTIFY NUMERIC AND CATEGORICAL FEATURES

numeric_features = []

categorical_features = []


for feature in features_used:

    if pd.api.types.is_numeric_dtype(
        X_train[feature]
    ):

        numeric_features.append(
            feature
        )

    else:

        categorical_features.append(
            feature
        )


print(
    "\nNumeric features:"
)

for feature in numeric_features:

    print(
        f" - {feature}"
    )


print(
    "\nCategorical features:"
)

for feature in categorical_features:

    print(
        f" - {feature}"
    )


# COPY TRAIN / TEST DATA

X_train_processed = X_train.copy()

X_test_processed = X_test.copy()


# NUMERIC IMPUTATION

if numeric_features:

    numeric_imputer = SimpleImputer(
        strategy="median"
    )

    X_train_processed[
        numeric_features
    ] = numeric_imputer.fit_transform(
        X_train[
            numeric_features
        ]
    )

    X_test_processed[
        numeric_features
    ] = numeric_imputer.transform(
        X_test[
            numeric_features
        ]
    )


# CATEGORICAL IMPUTATION

if categorical_features:

    categorical_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    X_train_processed[
        categorical_features
    ] = categorical_imputer.fit_transform(
        X_train[
            categorical_features
        ]
    )

    X_test_processed[
        categorical_features
    ] = categorical_imputer.transform(
        X_test[
            categorical_features
        ]
    )


# ENCODE CATEGORICAL VARIABLES

if categorical_features:

    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    X_train_processed[
        categorical_features
    ] = ordinal_encoder.fit_transform(
        X_train_processed[
            categorical_features
        ]
    )

    X_test_processed[
        categorical_features
    ] = ordinal_encoder.transform(
        X_test_processed[
            categorical_features
        ]
    )


# SCALE NUMERIC FEATURES

if numeric_features:

    scaler = StandardScaler()

    X_train_processed[
        numeric_features
    ] = scaler.fit_transform(
        X_train_processed[
            numeric_features
        ]
    )

    X_test_processed[
        numeric_features
    ] = scaler.transform(
        X_test_processed[
            numeric_features
        ]
    )


# DETERMINE CATEGORICAL COLUMN INDICES

categorical_indices = [
    features_used.index(feature)
    for feature in categorical_features
]


# APPLY SMOTENC

print(
    "\nApplying SMOTENC to training data..."
)


if categorical_indices:

    # Determine minority class size
    minority_count = (
        y_train.value_counts()
        .min()
    )

    # SMOTENC requires at least k_neighbors + 1
    # minority samples.
    #
    # With 18 minority samples, the default k=5
    # is safe. We calculate it dynamically anyway.

    k_neighbors = min(
        5,
        minority_count - 1
    )

    if k_neighbors < 1:

        raise ValueError(
            "\nNot enough minority samples "
            "to apply SMOTENC."
        )


    smote = SMOTENC(
        categorical_features=categorical_indices,
        k_neighbors=k_neighbors,
        random_state=RANDOM_STATE
    )


    X_train_resampled, y_train_resampled = (
        smote.fit_resample(
            X_train_processed,
            y_train
        )
    )

else:

    # If there are no categorical variables,
    # ordinary SMOTE could be used.
    #
    # However, this branch should not occur
    # with the current CPTAC-PDA clinical data.

    from imblearn.over_sampling import SMOTE

    minority_count = (
        y_train.value_counts()
        .min()
    )

    k_neighbors = min(
        5,
        minority_count - 1
    )

    smote = SMOTE(
        k_neighbors=k_neighbors,
        random_state=RANDOM_STATE
    )

    X_train_resampled, y_train_resampled = (
        smote.fit_resample(
            X_train_processed,
            y_train
        )
    )


# SHOW SMOTE RESULTS

print(
    "\nTraining distribution AFTER SMOTENC:"
)

resampled_distribution = (
    pd.Series(
        y_train_resampled
    )
    .value_counts()
    .sort_index()
)


print(
    resampled_distribution
)


# SAVE CLASS DISTRIBUTION

class_distribution_rows = []


for class_value in sorted(
    y_train.value_counts().index
):

    class_distribution_rows.append(
        {
            "stage": "Before SMOTENC",
            "class": int(class_value),
            "count": int(
                y_train.value_counts()
                .loc[class_value]
            ),
        }
    )


for class_value in sorted(
    pd.Series(
        y_train_resampled
    ).value_counts().index
):

    class_distribution_rows.append(
        {
            "stage": "After SMOTENC",
            "class": int(class_value),
            "count": int(
                pd.Series(
                    y_train_resampled
                )
                .value_counts()
                .loc[class_value]
            ),
        }
    )


class_distribution_df = pd.DataFrame(
    class_distribution_rows
)


class_distribution_df.to_csv(
    CLASS_DISTRIBUTION_FILE,
    index=False
)


# TRAIN NAIVE BAYES

print(
    "\nTraining Gaussian Naive Bayes..."
)


nb_model = GaussianNB()


nb_model.fit(
    X_train_resampled,
    y_train_resampled
)


# PREDICTIONS

y_pred = nb_model.predict(
    X_test_processed
)


y_probability = nb_model.predict_proba(
    X_test_processed
)[:, 1]


# EVALUATION

accuracy = accuracy_score(
    y_test,
    y_pred
)


balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)


recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ROC-AUC requires both classes in test data
if len(
    np.unique(y_test)
) == 2:

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

else:

    roc_auc = np.nan


conf_matrix = confusion_matrix(
    y_test,
    y_pred
)


# PRINT RESULTS

print(
    "\n"
    + "=" * 70
)

print(
    "TEST RESULTS"
)

print(
    "=" * 70
)


print(
    f"Threshold:            "
    f"{selected_threshold} days"
)

print(
    f"Accuracy:             "
    f"{accuracy:.4f}"
)

print(
    f"Balanced accuracy:    "
    f"{balanced_accuracy:.4f}"
)

print(
    f"Recall:               "
    f"{recall:.4f}"
)

print(
    f"F1 score:             "
    f"{f1:.4f}"
)

print(
    f"ROC-AUC:              "
    f"{roc_auc:.4f}"
)


print(
    "\nConfusion matrix:"
)

print(
    conf_matrix
)


# SAVE RESULTS

results_df = pd.DataFrame(
    [
        {
            "threshold_days": selected_threshold,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "train_samples_before_smote": len(y_train),
            "train_samples_after_smote": len(
                y_train_resampled
            ),
            "test_samples": len(y_test),
        }
    ]
)


results_df.to_csv(
    CLASSIFICATION_RESULTS_FILE,
    index=False
)


print(
    f"\nSaved:\n"
    f"{CLASSIFICATION_RESULTS_FILE}"
)

print(
    f"Saved:\n"
    f"{CLASS_DISTRIBUTION_FILE}"
)

print(
    f"Saved:\n"
    f"{REMOVED_FEATURES_FILE}"
)


print(
    "\n"
    + "=" * 70
)

print(
    "SMOTENC + NAIVE BAYES COMPLETED"
)

print(
    "=" * 70
)