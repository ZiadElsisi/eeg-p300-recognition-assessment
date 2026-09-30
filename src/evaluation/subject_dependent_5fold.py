from pathlib import Path
from scipy.io import loadmat
import numpy as np
import pandas as pd
import mne
import json

from sklearn.model_selection import StratifiedGroupKFold
from features.feature_extraction import feature_extraction
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Evaluate each subject separately.
for SUBJECT_ID in range(1, 9):

    epochs_path = (
        PROJECT_ROOT
        / "data"
        / "preprocessed"
        / f"subject_{SUBJECT_ID}_clean-epo.fif"
    )

    epochs = mne.read_epochs(
        epochs_path,
        preload=True,
        verbose=False
    )

    print("Number of epochs:", len(epochs))
    print("EEG data shape:", epochs.get_data().shape)

    # True labels: 1 = NonTarget, 2 = Target.
    y = epochs.events[:, 2]

    print("Number of labels:", len(y))
    print("First 10 labels:", y[:10])

    # Read trial boundaries from the original dataset.
    raw_path = (
        PROJECT_ROOT
        / "data"
        / "moabb"
        / "MNE-bnci-data"
        / "~bci"
        / "database"
        / "008-2014"
        / f"A{SUBJECT_ID:02d}.mat"
    )

    mat = loadmat(raw_path, simplify_cells=True)

    print(
        "Start samples of the first 5 trials:",
        mat["data"]["trial"][:5]
    )

    # MATLAB indexing starts at 1; Python indexing starts at 0.
    trial_starts = (
        np.asarray(mat["data"]["trial"]).ravel().astype(int) - 1
    )

    print(
        "Trial start samples (Python indexing):",
        trial_starts[:5]
    )

    # Assign each epoch to its original trial.
    groups = np.searchsorted(
        trial_starts,
        epochs.events[:, 0],
        side="right"
    ) - 1

    print("Number of groups:", len(np.unique(groups)))
    print("Groups of the first 10 epochs:", groups[:10])

    assert len(groups) == len(epochs), (
        "Group count does not match epoch count."
    )
    assert np.all(groups >= 0), (
        "An epoch occurs before the first trial."
    )

    print("Grouping checks passed.")

    # Keep each trial together and approximately preserve class ratios.
    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Only sample counts, labels and groups are needed for splitting.
    splits = list(
        cv.split(
            X=np.zeros((len(y), 1)),
            y=y,
            groups=groups
        )
    )

    # Check that no trial appears in both train and test.
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        shared_groups = np.intersect1d(
            groups[train_idx],
            groups[test_idx]
        )

        assert len(shared_groups) == 0, (
            "A trial appears in both train and test."
        )

        print(
            f"Fold {fold}: "
            f"train={len(train_idx)}, "
            f"test={len(test_idx)}, "
            f"shared_trials={len(shared_groups)}"
        )

    # Reuse the team's feature extraction: 128 features per epoch.
    X = feature_extraction(epochs)

    print("Feature matrix shape:", X.shape)

    # Reset results for the current subject.
    fold_results = []
    total_cm = np.zeros((2, 2), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Create pipeline
        model = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LinearDiscriminantAnalysis(priors=[0.5, 0.5])
            )
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Use Target probabilities to calculate ROC-AUC.
        target_column = np.flatnonzero(model.classes_ == 2)[0]
        y_prob = model.predict_proba(X_test)[:, target_column]
        roc_auc = roc_auc_score(y_test == 2, y_prob)

        # Calculate precision, recall and F1 for the Target class.
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, pos_label=2, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, pos_label=2, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, pos_label=2, zero_division=0
        )

        fold_results.append({
            "fold": fold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        })

        print(f"\nFold {fold}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Target precision: {precision:.3f}")
        print(f"Target recall: {recall:.3f}")
        print(f"Target F1: {f1:.3f}")
        print(f"ROC-AUC: {roc_auc:.3f}")

        # Rows are actual classes; columns are predicted classes.
        cm = confusion_matrix(y_test, y_pred, labels=[1, 2])
        total_cm += cm

        print("Confusion matrix:")
        print(cm)

    # Combine test counts from all five folds.
    print("================================")
    print(f"\nSubject {SUBJECT_ID} - Combined confusion matrix:")
    print(total_cm)
    print("================================")

    print(f"\nSubject {SUBJECT_ID} - Mean scores across 5 folds")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        values = [result[metric] for result in fold_results]
        print(f"Mean {metric}: {np.mean(values):.3f}")

    results_dir = PROJECT_ROOT / "results" / "subject_dependent"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save epoch row indices used in each fold.
    split_records = [
        {
            "fold": fold,
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist()
        }
        for fold, (train_idx, test_idx) in enumerate(splits, start=1)
    ]

    splits_path = results_dir / f"subject_{SUBJECT_ID}_splits.json"

    with splits_path.open("w", encoding="utf-8") as file:
        json.dump(split_records, file, indent=2)

    print("Split indices saved to:", splits_path)

    # Save individual fold metrics.
    results_df = pd.DataFrame(fold_results)

    results_path = results_dir / f"subject_{SUBJECT_ID}_fold_metrics.csv"
    results_df.to_csv(results_path, index=False)
    print("Fold metrics saved to:", results_path)

    # Save the subject's combined confusion matrix.
    cm_df = pd.DataFrame(
        total_cm,
        index=["Actual NonTarget", "Actual Target"],
        columns=["Predicted NonTarget", "Predicted Target"]
    )

    cm_path = results_dir / f"subject_{SUBJECT_ID}_confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print("Confusion matrix saved to:", cm_path)

    # Save mean fold scores for this subject.
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    summary = results_df[metrics].mean().to_frame().T
    summary.insert(0, "subject", SUBJECT_ID)

    summary_path = results_dir / f"subject_{SUBJECT_ID}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Subject summary saved to:", summary_path)

# Combine summaries from all eight subjects.
summary_files = [
    results_dir / f"subject_{subject}_summary.csv"
    for subject in range(1, 9)
]

all_subjects = pd.concat(
    [pd.read_csv(path) for path in summary_files],
    ignore_index=True
)

all_subjects.to_csv(
    results_dir / "all_subjects_summary.csv",
    index=False
)

# Calculate mean and sample standard deviation across subjects.
overall = all_subjects[metrics].agg(["mean", "std"])

overall.to_csv(
    results_dir / "overall_summary.csv"
)

print("\nAll subjects:")
print(all_subjects.round(3))
print("\nMean and standard deviation across subjects:")
print(overall.round(3))