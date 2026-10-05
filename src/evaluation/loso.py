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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

# The existing modules use both "from features..." (src on the path, as in
# Task 6) and "from src.loaders..." (project root on the path).
for _p in (str(PROJECT_ROOT), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results" / "loso"

# Label convention used by the whole project: 1 = NonTarget, 2 = Target.
NONTARGET_LABEL = 1
TARGET_LABEL = 2

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Fixed, already-completed Task 6 results (mean, std) - reference only.
TASK6_RESULTS = {
    "accuracy": (0.808, 0.040),
    "precision": (0.458, 0.075),
    "recall": (0.759, 0.053),
    "f1": (0.570, 0.071),
    "roc_auc": (0.865, 0.045),
}


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
    """Subject IDs derived from data/preprocessed/subject_<id>_clean-epo.fif."""
    pattern = re.compile(r"^subject_(\d+)_clean-epo\.fif$")
    subjects = []
    for path in Path(preprocessed_dir).glob("subject_*_clean-epo.fif"):
        match = pattern.match(path.name)
        if match:
            subjects.append(int(match.group(1)))
    return sorted(subjects)


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
    import mne  # imported lazily so the unit tests do not need MNE
    from features.feature_extraction import feature_extraction

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
def make_loso_folds(subject_ids: np.ndarray) -> list[dict]:
    """One fold per subject; the held-out subject is wholly excluded."""
    subject_ids = np.asarray(subject_ids)
    subjects = [int(s) for s in np.unique(subject_ids)]

    folds = []
    for fold_number, held_out in enumerate(subjects, start=1):
        train_mask = subject_ids != held_out
        test_mask = subject_ids == held_out
        folds.append({
            "fold": fold_number,
            "held_out_subject": held_out,
            "train_subjects": [s for s in subjects if s != held_out],
            "train_idx": np.flatnonzero(train_mask),
            "test_idx": np.flatnonzero(test_mask),
        })
    return folds


def assert_fold_integrity(fold: dict, subject_ids: np.ndarray) -> None:
    """Fail loudly if the held-out subject could reach training."""
    subject_ids = np.asarray(subject_ids)
    held_out = fold["held_out_subject"]
    train_idx = np.asarray(fold["train_idx"])
    test_idx = np.asarray(fold["test_idx"])

    train_subjects_actual = {int(s) for s in np.unique(subject_ids[train_idx])}
    test_subjects_actual = {int(s) for s in np.unique(subject_ids[test_idx])}

    if held_out in train_subjects_actual:
        raise LeakageError(
            f"Held-out subject {held_out} appears in the training rows."
        )
    if held_out in fold["train_subjects"]:
        raise LeakageError(
            f"Held-out subject {held_out} is in the recorded training list."
        )
    if test_subjects_actual != {held_out}:
        raise LeakageError(
            f"Test rows must contain only subject {held_out}; "
            f"found {sorted(test_subjects_actual)}."
        )
    if train_subjects_actual != set(fold["train_subjects"]):
        raise LeakageError(
            "Recorded training subjects do not match the training rows."
        )
    if np.intersect1d(train_idx, test_idx).size != 0:
        raise LeakageError("Train and test rows overlap.")


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
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_fold(fold: dict, X, y, subject_ids, model_factory=build_model):
    """Fit on the training subjects only; evaluate on the held-out subject."""
    assert_fold_integrity(fold, subject_ids)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    train_idx, test_idx = fold["train_idx"], fold["test_idx"]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = model_factory()          # fresh scaler + LDA for every fold
    model.fit(X_train, y_train)      # training subjects only

    y_pred = model.predict(X_test)

    # Use the Target probability column, not an assumed column 1.
    target_column = int(np.flatnonzero(model.classes_ == TARGET_LABEL)[0])
    y_prob = model.predict_proba(X_test)[:, target_column]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    return {
        "fold": fold["fold"],
        "held_out_subject": fold["held_out_subject"],
        "train_subjects": ",".join(str(s) for s in fold["train_subjects"]),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        **metrics,
    }


def run_loso(X, y, subject_ids, model_factory=build_model,
             verbose: bool = False) -> pd.DataFrame:
    """Run all LOSO folds and return one row per held-out subject."""
    folds = make_loso_folds(subject_ids)
    rows = []
    for fold in folds:
        row = evaluate_fold(fold, X, y, subject_ids, model_factory)
        rows.append(row)
        if verbose:
            print(
                f"Fold {row['fold']} | held-out subject "
                f"{row['held_out_subject']} | train subjects "
                f"[{row['train_subjects']}] | n_train={row['n_train']} "
                f"n_test={row['n_test']}\n"
                f"   acc={row['accuracy']:.3f} prec={row['precision']:.3f} "
                f"rec={row['recall']:.3f} f1={row['f1']:.3f} "
                f"auc={row['roc_auc']:.3f}  "
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


def print_comparison(summary_df: pd.DataFrame) -> None:
    print("\nTask 6 (subject-dependent 5-fold) vs Task 7 (LOSO)")
    print(f"{'Metric':<10} | {'Subject-Dependent 5-Fold':<26} | "
          f"{'Subject-Independent LOSO':<26} | Delta (LOSO - SD)")
    for _, row in summary_df.iterrows():
        m = row["metric"]
        t6_mean, t6_std = TASK6_RESULTS[m]
        print(f"{m:<10} | {t6_mean:.3f} ± {t6_std:.3f}".ljust(40)
              + f"| {row['mean']:.3f} ± {row['std']:.3f}".ljust(29)
              + f"| {row['mean'] - t6_mean:+.3f}")


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

    print_comparison(summary_df)
    print(f"\nSaved: {fold_path}\nSaved: {summary_path}")


if __name__ == "__main__":
    main()