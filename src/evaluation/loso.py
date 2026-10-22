from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mne  # imported lazily so the unit tests do not need MNE
from src.features.feature_extraction import feature_extraction
from sklearn.model_selection import LeaveOneGroupOut
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results" / "loso"

# Label convention used by the whole project: 1 = NonTarget, 2 = Target.
NONTARGET_LABEL = 1
TARGET_LABEL = 2

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
]



class LeakageError(RuntimeError):
    """Raised when a LOSO fold would let the held-out subject leak."""
# ----------------------------------------------------------------------
# Model (must stay identical to the configuration used in Task 6)
# ----------------------------------------------------------------------
def build_model() -> Pipeline:
    """StandardScaler + LDA(priors=[0.5, 0.5]) - same as Task 6."""
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LinearDiscriminantAnalysis(priors=[0.5, 0.5]),
        ),
    ])


# ----------------------------------------------------------------------
# Data loading (subject IDs are kept explicitly for every feature row)
# ----------------------------------------------------------------------
def discover_subjects(preprocessed_dir: Path = PREPROCESSED_DIR) -> list[int]:
    subjects = []
    for path in Path(preprocessed_dir).glob("subject_*_clean-epo.fif"):
        number = path.name.split("_")[1]   # "subject_5_clean-epo.fif" -> "5"
        subjects.append(int(number))
    return sorted(subjects) ## Return a list with all subject IDs inside  preprocessed folder.

def load_all_subjects(
    subjects: list[int],
    preprocessed_dir: Path = PREPROCESSED_DIR,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load every subject's clean epochs and extract features with the existing
    project feature extraction (stateless window means, no learned step).

    Returns
    -------
    X : DataFrame (n_rows, 128)
    y : ndarray, 1 = NonTarget, 2 = Target
    subject_ids : ndarray, subject ID of every row of X
    """


    X_parts, y_parts, id_parts = [], [], []

    for subject in subjects:
        epochs = mne.read_epochs(
            Path(preprocessed_dir) / f"subject_{subject}_clean-epo.fif",
            preload=True,
            verbose=False,
        )
        y_subject = epochs.events[:, 2]

        # feature_extraction prints a lot; silence it for 8 subjects.
        with contextlib.redirect_stdout(io.StringIO()):
            X_subject = feature_extraction(epochs)

        if X_subject.empty or len(X_subject) != len(y_subject):
            raise RuntimeError(
                f"Feature extraction failed for subject {subject}: "
                f"{X_subject.shape} features vs {len(y_subject)} labels."
            )
        if not set(np.unique(y_subject)) <= {NONTARGET_LABEL, TARGET_LABEL}:
            raise ValueError(
                f"Subject {subject} has unexpected labels: "
                f"{np.unique(y_subject)}"
            )

        X_parts.append(X_subject)
        y_parts.append(y_subject)
        id_parts.append(np.full(len(y_subject), subject, dtype=int))
        print(f"Loaded subject {subject}: {X_subject.shape[0]} epochs, "
              f"{X_subject.shape[1]} features")

    X = pd.concat(X_parts, ignore_index=True)
    y = np.concatenate(y_parts)
    subject_ids = np.concatenate(id_parts)
    return X, y, subject_ids


# ----------------------------------------------------------------------
# LOSO folds
# ----------------------------------------------------------------------

def make_loso_folds(subject_ids):
    loso = LeaveOneGroupOut()
    folds = []
    for i, (train_idx, test_idx) in enumerate(loso.split(subject_ids, groups=subject_ids), start=1):
        held_out = int(subject_ids[test_idx[0]])
        folds.append({
            "fold": i,
            "held_out_subject": held_out,
            "train_subjects": sorted(set(subject_ids[train_idx])),
            "train_idx": train_idx,
            "test_idx": test_idx,
        })
    return folds


# def assert_fold_integrity(fold: dict, subject_ids: np.ndarray) -> None:
#     """Fail loudly if the held-out subject could reach training."""
#     subject_ids = np.asarray(subject_ids)
#     held_out = fold["held_out_subject"]
#     train_idx = np.asarray(fold["train_idx"])
#     test_idx = np.asarray(fold["test_idx"])
#
#     train_subjects_actual = {int(s) for s in np.unique(subject_ids[train_idx])}
#     test_subjects_actual = {int(s) for s in np.unique(subject_ids[test_idx])}
#
#     if held_out in train_subjects_actual:
#         raise LeakageError(
#             f"Held-out subject {held_out} appears in the training rows."
#         )
#     if held_out in fold["train_subjects"]:
#         raise LeakageError(
#             f"Held-out subject {held_out} is in the recorded training list."
#         )
#     if test_subjects_actual != {held_out}:
#         raise LeakageError(
#             f"Test rows must contain only subject {held_out}; "
#             f"found {sorted(test_subjects_actual)}."
#         )
#     if train_subjects_actual != set(fold["train_subjects"]):
#         raise LeakageError(
#             "Recorded training subjects do not match the training rows."
#         )
#     if np.intersect1d(train_idx, test_idx).size != 0:
#         raise LeakageError("Train and test rows overlap.")


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob_target) -> dict:
    """Same definitions as Task 6 (Target = positive class)."""
    y_true = np.asarray(y_true)
    cm = confusion_matrix(
        y_true, y_pred, labels=[NONTARGET_LABEL, TARGET_LABEL]
    )
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, pos_label=TARGET_LABEL, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, pos_label=TARGET_LABEL, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, pos_label=TARGET_LABEL, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true == TARGET_LABEL, y_prob_target),
        "pr_auc": average_precision_score(
            y_true == TARGET_LABEL, y_prob_target
        ),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_fold(fold: dict, X, y, model_factory=build_model):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # 1. Split the rows
    X_train, y_train = X[fold["train_idx"]], y[fold["train_idx"]]
    X_test,  y_test  = X[fold["test_idx"]],  y[fold["test_idx"]]

    # 2. Train a new model
    model = model_factory()
    model.fit(X_train, y_train)

    # 3. Predict on the held-out person
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]   # chance of Target

    # 4. Score it
    metrics = compute_metrics(y_test, y_pred, y_prob)
    return {"held_out_subject": fold["held_out_subject"], **metrics}


def run_loso(X, y, subject_ids, model_factory=build_model,
             verbose: bool = False) -> pd.DataFrame:
    """Run all LOSO folds and return one row per held-out subject."""
    folds = make_loso_folds(subject_ids)
    rows = []
    for fold in folds:
        row = evaluate_fold(fold, X, y, model_factory=model_factory)
        rows.append(row)
        if verbose:
            print(
                f"Held-out subject {row['held_out_subject']} | "
                f"acc={row['accuracy']:.3f} prec={row['precision']:.3f} "
                f"rec={row['recall']:.3f} f1={row['f1']:.3f} "
                f"roc_auc={row['roc_auc']:.3f} "
                f"pr_auc={row['pr_auc']:.3f} "
                f"balanced_acc={row['balanced_accuracy']:.3f} "
                f"CM[[tn fp][fn tp]]=[[{row['tn']} {row['fp']}]"
                f"[{row['fn']} {row['tp']}]]"
            )
    return pd.DataFrame(rows)


def summarize(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample std (ddof=1, as in Task 6) across held-out subjects."""
    return pd.DataFrame({
        "metric": METRICS,
        "mean": [fold_df[m].mean() for m in METRICS],
        "std": [fold_df[m].std(ddof=1) for m in METRICS],
    })


def save_results(fold_df: pd.DataFrame, summary_df: pd.DataFrame,
                 results_dir: Path = RESULTS_DIR) -> tuple[Path, Path]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    fold_path = results_dir / "loso_fold_metrics.csv"
    summary_path = results_dir / "loso_summary.csv"
    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return fold_path, summary_path


# def print_comparison(summary_df: pd.DataFrame) -> None:
#     print("\nTask 6 (subject-dependent 5-fold) vs Task 7 (LOSO)")
#     print(f"{'Metric':<10} | {'Subject-Dependent 5-Fold':<26} | "
#           f"{'Subject-Independent LOSO':<26} | Delta (LOSO - SD)")
#     for _, row in summary_df.iterrows():
#         m = row["metric"]
#         t6_mean, t6_std = TASK6_RESULTS[m]
#         print(f"{m:<10} | {t6_mean:.3f} ± {t6_std:.3f}".ljust(40)
#               + f"| {row['mean']:.3f} ± {row['std']:.3f}".ljust(29)
#               + f"| {row['mean'] - t6_mean:+.3f}")
#

def main() -> None:
    subjects = discover_subjects()
    if not subjects:
        raise FileNotFoundError(
            f"No subject_<id>_clean-epo.fif files in {PREPROCESSED_DIR}"
        )
    print(f"Subjects found: {subjects} (N = {len(subjects)})")

    X, y, subject_ids = load_all_subjects(subjects)
    print(f"Feature matrix: {X.shape}, labels: {y.shape}, "
          f"subject_ids: {subject_ids.shape}\n")

    fold_df = run_loso(X, y, subject_ids, verbose=True)
    summary_df = summarize(fold_df)
    fold_path, summary_path = save_results(fold_df, summary_df)

    print(f"\nLOSO mean ± std across {len(fold_df)} held-out subjects")
    for _, row in summary_df.iterrows():
        print(f"{row['metric']:<10}: {row['mean']:.3f} ± {row['std']:.3f}")

    # print_comparison(summary_df)
    print(f"\nSaved: {fold_path}\nSaved: {summary_path}")

SUBJECTS = [1, 2, 3, 4, 5, 6]
N_PER_SUBJECT = {1: 120, 2: 110, 3: 130, 4: 100, 5: 125, 6: 115}
N_FEATURES = 6
SENTINEL_COL = N_FEATURES - 1


def sentinel_value(subject: int) -> float:
    return 1000.0 + subject


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    X_parts, y_parts, id_parts = [], [], []
    for subject in SUBJECTS:
        n = N_PER_SUBJECT[subject]
        y_subject = np.where(
            rng.random(n) < 0.2, TARGET_LABEL, NONTARGET_LABEL
        )
        X_subject = rng.normal(size=(n, N_FEATURES))
        X_subject[y_subject == TARGET_LABEL, 0] += 2.0
        X_subject[:, SENTINEL_COL] = sentinel_value(subject)
        X_parts.append(X_subject)
        y_parts.append(y_subject)
        id_parts.append(np.full(n, subject))
    return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(id_parts)


class SpyModel:
    fit_sentinels = []
    predict_sentinels = []
    fit_sizes = []

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LinearDiscriminantAnalysis(priors=[0.5, 0.5])),
        ])

    @property
    def classes_(self):
        return self.model.classes_

    def fit(self, X, y):
        type(self).fit_sentinels.append(set(np.unique(X[:, SENTINEL_COL])))
        type(self).fit_sizes.append(len(X))
        self.model.fit(X, y)
        return self

    def predict(self, X):
        type(self).predict_sentinels.append(set(np.unique(X[:, SENTINEL_COL])))
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def reset_spy():
    SpyModel.fit_sentinels = []
    SpyModel.predict_sentinels = []
    SpyModel.fit_sizes = []


def test_held_out_subject_never_seen_in_training(synthetic, monkeypatch):
    X, y, ids = synthetic
    reset_spy()
    monkeypatch.setattr(sys.modules[__name__], "build_model", SpyModel)
    df = run_loso(X, y, ids, model_factory=SpyModel)

    assert len(SpyModel.fit_sentinels) == len(SUBJECTS)
    assert df["held_out_subject"].tolist() == SUBJECTS
    for fold_no, held in enumerate(SUBJECTS):
        fit_seen = SpyModel.fit_sentinels[fold_no]
        pred_seen = SpyModel.predict_sentinels[fold_no]
        assert sentinel_value(held) not in fit_seen
        assert fit_seen == {sentinel_value(s) for s in SUBJECTS if s != held}
        assert pred_seen == {sentinel_value(held)}
        assert SpyModel.fit_sizes[fold_no] == (
            sum(N_PER_SUBJECT.values()) - N_PER_SUBJECT[held]
        )

    # run_loso's output omits train_subjects; verify its existing fold records.
    for fold in make_loso_folds(ids):
        held = fold["held_out_subject"]
        train = fold["train_subjects"]
        assert held not in train
        assert len(train) == len(SUBJECTS) - 1



if __name__ == "__main__":
    main()
