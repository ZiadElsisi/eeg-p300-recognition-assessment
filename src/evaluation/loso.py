from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import mne  # imported lazily so the unit tests do not need MNE

from Utility.config import LABELS
from src.models.LDA_pipeline import EEGClassifier, EvaluationReporter
from sklearn.model_selection import LeaveOneGroupOut
from src.Utility.config import (
    NUM_SUBJECTS,
    LOSO_SUMMARY_PATH,
    LOSO_FOLDS_PATH,
    PREPROCESSED_DATA_DIR,
    METRICS,
    RESULT_DIR,
    CLASSIFICATION_THRESHOLD,
)




class LeaveOneOut:
    def __init__(self):
        self.loso = LeaveOneGroupOut()
        self.evaluator=EvaluationReporter()
        self.model=EEGClassifier()
        self.folds_results = []
        self.is_active=False

    def run(self,verbose: bool = False):
        if self.is_active:return
        self.folds_results = []
        epochs, subject_ids=self.load_all_subjects()
        """Run all LOSO folds and return one row per held-out subject."""
        folds = self.make_loso_folds(subject_ids)
        for fold in folds:
            self.assert_fold_integrity(fold, subject_ids)
            row = self.evaluate_fold(fold, epochs)
            self.folds_results.append(row)
            if verbose:
                print(
                    f"Held-out subject {row['held_out_subject']} | "
                    f"acc={row['accuracy']:.3f} prec={row['precision']:.3f} "
                    f"rec={row['recall']:.3f} f1={row['f1_score']:.3f} "
                    f"roc_auc={row['roc_auc']:.3f} "
                    f"pr_auc={row['pr_auc']:.3f} "
                    f"balanced_acc={row['balanced_accuracy']:.3f} "
                    f"CM[[tn fp][fn tp]]=[[{row['tn']} {row['fp']}]"
                    f"[{row['fn']} {row['tp']}]]"
                )
        self.save_loso_results()
        self.is_active=True

    def evaluate_fold(self,fold: dict, epochs):

        # 1. get fold's train and test epochs
        train_epochs = epochs[fold["train_idx"]]
        test_epoch = epochs[fold["test_idx"]]

        # 2. Train a new model
        self.model.fit(train_epochs)

        # 3. Predict on the held-out person
        probabilities = self.model.predict_proba(test_epoch)
        predictions = np.where(
            probabilities >= CLASSIFICATION_THRESHOLD,
            LABELS["target"],
            LABELS["nontarget"]
        )
        y_true = self.model.get_last_transformed_epochs().events[:, 2]

        # 4. Score it
        metrics,_,cm,_ = self.evaluator.evaluate(y_true=y_true, y_pred=predictions,y_prob=probabilities)
        tn, fp, fn, tp = (int(v) for v in cm.ravel())
        metrics=metrics|{
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,}
        return {"held_out_subject": fold["held_out_subject"], **metrics}

    def make_loso_folds(self,subject_ids):
        folds = []
        for i, (train_idx, test_idx) in enumerate(self.loso.split(subject_ids, groups=subject_ids), start=1):
            held_out = int(subject_ids[test_idx[0]])
            folds.append({
                "fold": i,
                "held_out_subject": held_out,
                "train_subjects": sorted(set(subject_ids[train_idx])),
                "train_idx": train_idx,
                "test_idx": test_idx,
            })
        return folds

    def load_all_subjects(self):
        id_parts = []
        all_subjects_epochs = []
        for subject in range(1, NUM_SUBJECTS + 1):
            epochs = mne.read_epochs(
                Path(PREPROCESSED_DATA_DIR) / f"subject_{subject}_clean-epo.fif",
                preload=True,
                verbose=False,
            )
            all_subjects_epochs.append(epochs)
            id_parts.append(np.full(len(epochs), subject, dtype=int))
        combined_epochs = mne.concatenate_epochs(all_subjects_epochs)
        subject_ids = np.concatenate(id_parts)

        assert len(combined_epochs) == len(subject_ids)

        return combined_epochs, subject_ids
    def save_loso_results(self):
        fold_df=pd.DataFrame(self.folds_results)
        summary_df = self.summarize(fold_df=fold_df)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(LOSO_FOLDS_PATH, index=False)
        summary_df.to_csv(LOSO_SUMMARY_PATH, index=False)

        print(f"\nLOSO mean ± std across {len(fold_df)} held-out subjects")
        for _, row in summary_df.iterrows():
            print(f"{row['metric']:<10}: {row['mean']:.3f} ± {row['std']:.3f}")

        # print_comparison(summary_df)
        print(f"\nSaved: {LOSO_FOLDS_PATH}\nSaved: {LOSO_SUMMARY_PATH}")

    def summarize(self,fold_df:pd.DataFrame) -> pd.DataFrame:
        """Mean and sample std (ddof=1, as in Task 6) across held-out subjects."""
        return pd.DataFrame({
            "metric": METRICS,
            "mean": [fold_df[m].mean() for m in METRICS],
            "std": [fold_df[m].std(ddof=1) for m in METRICS],
        })

    def assert_fold_integrity(self,fold: dict, subject_ids: np.ndarray) -> None:
        """Fail if a LOSO fold mixes its held-out subject into training."""
        subject_ids = np.asarray(subject_ids)
        held_out = int(fold["held_out_subject"])
        train_idx = np.asarray(fold["train_idx"], dtype=int)
        test_idx = np.asarray(fold["test_idx"], dtype=int)

        if np.intersect1d(train_idx, test_idx).size:
            raise LeakageError("Train and test row indices overlap.")

        train_subjects_actual = {int(s) for s in np.unique(subject_ids[train_idx])}
        test_subjects_actual = {int(s) for s in np.unique(subject_ids[test_idx])}
        recorded_train_subjects = {
            int(s) for s in fold["train_subjects"]
        }

        if held_out in train_subjects_actual or held_out in recorded_train_subjects:
            raise LeakageError(
                f"Held-out subject {held_out} appears in training."
            )
        if test_subjects_actual != {held_out}:
            raise LeakageError(
                f"Test rows must contain only subject {held_out}; "
                f"found {sorted(test_subjects_actual)}."
            )
        if train_subjects_actual != recorded_train_subjects:
            raise LeakageError(
                "Recorded training subjects do not match the training rows."
            )

class LeakageError(RuntimeError):
    """Raised when a LOSO fold would let the held-out subject leak."""

if __name__ == "__main__":
    LeaveOneOut().run()

