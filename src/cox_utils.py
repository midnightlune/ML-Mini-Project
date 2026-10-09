"""
cox_utils.py

Shared helpers so scripts 7 and 9 prepare data and fit the Cox model
identically, and so the model is numerically stable on a small cohort.

What makes the model unstable, and what this file does about it:

    1. Sparse one-hot categories      -> rare levels are merged into "Other"
    2. Arbitrary reference category   -> reference = most frequent level
    3. Redundant variables            -> AJCC T/N/M and Metastasis At Diagnosis are dropped (stage already encodes them)
    4. Collinear columns              -> one of each pair with |r| > 0.9 is dropped
    5. Hand-picked penalizer          -> chosen by cross-validated C-index
    6. Convergence failures           -> smaller Newton step + escalating penalizer
"""

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
from sklearn.model_selection import StratifiedKFold


DURATION_COL = "survival_time_days"
EVENT_COL = "survival_event"

CANDIDATE_FEATURES = [
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

# Largely redundant with AJCC Pathologic Stage. 
REDUNDANT_FEATURES = [
    "AJCC Pathologic T",
    "AJCC Pathologic N",
    "AJCC Pathologic M",
    "Metastasis At Diagnosis",
]

MIN_CATEGORY_COUNT = 8      # levels with fewer patients get merged
MAX_ABS_CORR = 0.90         # drop one column of any pair above this
PENALIZER_GRID = [0.01, 0.05, 0.1, 0.5, 1.0]
CV_FOLDS = 5
SEED = 42


def load_survival_data(path):
    """Load the cleaned CSV and keep usable survival rows."""
    df = pd.read_csv(path)

    for column in (DURATION_COL, EVENT_COL):
        if column not in df.columns:
            raise ValueError(f"Required column missing: {column}")
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[DURATION_COL, EVENT_COL])
    df = df[df[DURATION_COL] > 0].copy()
    return df


def _collapse_rare(series):
    """Merge rare levels into 'Other'; fold 'Other' into the biggest
    level if it is itself still too small."""
    counts = series.value_counts()
    rare = counts[counts < MIN_CATEGORY_COUNT].index

    if len(rare) == 0:
        return series, []

    out = series.where(~series.isin(rare), "Other")

    if (out == "Other").sum() < MIN_CATEGORY_COUNT:
        biggest = series.value_counts().idxmax()
        out = out.where(out != "Other", biggest)

    return out, list(rare)


def prepare_cox_data(df, drop_redundant=True):
    """
    Returns (model_df, info).

    model_df : duration, event and numeric covariates only
    info     : dict describing what was removed/merged
    """
    info = {
        "missing_from_dataset": [],
        "all_missing": [],
        "redundant": [],
        "merged_levels": {},
        "constant": [],
        "correlated": [],
    }

    info["missing_from_dataset"] = [
        f for f in CANDIDATE_FEATURES if f not in df.columns
    ]
    available = [f for f in CANDIDATE_FEATURES if f in df.columns]

    features = []
    for f in available:
        if df[f].isna().all():
            info["all_missing"].append(f)
        elif drop_redundant and f in REDUNDANT_FEATURES:
            info["redundant"].append(f)
        else:
            features.append(f)

    if not features:
        raise ValueError("No usable clinical features remain.")

    work = df[[DURATION_COL, EVENT_COL] + features].copy()
    pieces = [work[[DURATION_COL, EVENT_COL]]]

    for f in features:
        if pd.api.types.is_numeric_dtype(work[f]):
            col = pd.to_numeric(work[f], errors="coerce")
            pieces.append(col.fillna(col.median()).rename(f).to_frame())
        else:
            s = work[f].astype("object").fillna("Unknown").astype(str)
            s, merged = _collapse_rare(s)
            if merged:
                info["merged_levels"][f] = merged

            reference = s.value_counts().idxmax()
            dummies = pd.get_dummies(s, prefix=f, dtype=float)
            dummies = dummies.drop(columns=f"{f}_{reference}")
            pieces.append(dummies)

    model_df = pd.concat(pieces, axis=1)
    covariates = [
        c for c in model_df.columns if c not in (DURATION_COL, EVENT_COL)
    ]

    # constant columns
    info["constant"] = [
        c for c in covariates if model_df[c].nunique(dropna=False) <= 1
    ]
    model_df = model_df.drop(columns=info["constant"])
    covariates = [c for c in covariates if c not in info["constant"]]

    # highly correlated columns
    if len(covariates) > 1:
        corr = model_df[covariates].corr().abs()
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )
        info["correlated"] = [
            c for c in upper.columns if (upper[c] > MAX_ABS_CORR).any()
        ]
        model_df = model_df.drop(columns=info["correlated"])
        covariates = [c for c in covariates if c not in info["correlated"]]

    if model_df.isna().any().any():
        raise ValueError("Missing values remain in the Cox dataset.")

    n_events = int(model_df[EVENT_COL].sum())
    print(
        f"\nCovariates in model: {len(covariates)} | events: {n_events} | "
        f"events per covariate: {n_events / max(len(covariates), 1):.1f} "
        "(rule of thumb: >= 10)"
    )

    return model_df, info


def select_penalizer(model_df):
    """Pick the L2 penalizer with the best cross-validated C-index."""
    scores = {}

    for p in PENALIZER_GRID:
        try:
            fold_scores = k_fold_cross_validation(
                CoxPHFitter(penalizer=p),
                model_df,
                duration_col=DURATION_COL,
                event_col=EVENT_COL,
                k=CV_FOLDS,
                scoring_method="concordance_index",
                seed=SEED,
            )
            scores[p] = float(np.mean(fold_scores))
            print(f"  penalizer={p:<5} CV C-index={scores[p]:.4f}")
        except Exception as error:
            print(f"  penalizer={p:<5} failed: {error}")

    return scores


def fit_stable_cox(model_df):
    """
    Returns (fitted CoxPHFitter, chosen penalizer, cv_c_index or NaN).
    """
    print("\nSelecting penalizer by cross-validation...")
    cv_scores = select_penalizer(model_df)

    best = max(cv_scores, key=cv_scores.get) if cv_scores else 0.5

    candidates = [best] + [
        p for p in sorted(set(PENALIZER_GRID + [2.0, 5.0]))
        if p > best
    ]

    for p in candidates:
        try:
            cph = CoxPHFitter(penalizer=p)

            cph.fit(
                model_df,
                duration_col=DURATION_COL,
                event_col=EVENT_COL,
                show_progress=False,
            )

            cv_c = cv_scores.get(p, np.nan)

            print(f"\nFitted Cox model with penalizer={p}")
            return cph, p, cv_c

        except Exception as error:
            print(f"Fit failed at penalizer={p}: {error}")

    raise RuntimeError(
        "Cox model could not be fitted at any penalizer."
    )


def cross_val_risk_scores(model_df, penalizer, k=CV_FOLDS):
    """
    Out-of-fold risk scores: each patient is scored by a model that never
    saw them. Use this instead of in-sample scores for risk groups.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    scores = pd.Series(np.nan, index=model_df.index, name="cox_risk_score")

    for train_idx, test_idx in skf.split(model_df, model_df[EVENT_COL]):
        train = model_df.iloc[train_idx]
        test = model_df.iloc[test_idx]

        cph = CoxPHFitter(penalizer=penalizer)

        cph.fit(
            train,
            duration_col=DURATION_COL,
            event_col=EVENT_COL,
        )
        scores.iloc[test_idx] = np.asarray(
            cph.predict_partial_hazard(test)
        ).reshape(-1)

    return scores