from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import mne
import json
import platform
import scipy
import sklearn
from src.Utility.config import (BLOCKS_PER_SESSION, TRIALS_PER_BLOCK, SEED, PROJECT_ROOT, RESULT_DIR, METRICS,
                                NUM_SUBJECTS, CLASSIFICATION_THRESHOLD,LABELS)
import src.features.feature_extraction as ext
from src.validation.epoch_splitting import get_block_ids,EpochManager
from src.models.LDA_pipeline import EEGClassifier, EvaluationReporter

class SubjectDependent5FoldCV:
    def __init__(self):
        self.all_subjects_summaries=[]
        self.fold_results = []
        self.split_records = []
        self.total_cm = np.zeros((2, 2), dtype=int)
        self.evaluator=EvaluationReporter()
        self.epoch_manager = EpochManager()
        self.model = EEGClassifier()
        self.is_active=False
    def run(self):
        if self.is_active:return
        for subject_id in range(1,NUM_SUBJECTS+1):
            epochs_path = (
                    PROJECT_ROOT
                    / "data"
                    / "preprocessed"
                    / f"subject_{subject_id}_clean-epo.fif"
            )
            epochs = mne.read_epochs(
                epochs_path,
                preload=True,
                verbose=False
            )
            # ------------------------------------------------------------
            # get epochs' LABELS DATAFRAME for each subject.
            # ------------------------------------------------------------

            # The validation module reads trial boundaries and assigns epochs to blocks.
            raw_path = (
                    PROJECT_ROOT
                    / "data"
                    / "moabb"
                    / "MNE-bnci-data"
                    / "~bci"
                    / "database"
                    / "008-2014"
                    / f"A{subject_id:02d}.mat"
            )
            groups = get_block_ids(mat_path=raw_path, epochs=epochs)
            splits = self.epoch_manager.make_cv_splits(
                epochs,
                epochs.events[:, 2],
                raw_path,
                n_splits=5,
                seed=SEED,
            )

            print("Grouping level: block (five consecutive character trials per block)")
            print("Number of blocks:", len(np.unique(groups)))
            print("Block IDs of the first 10 epochs:", groups[:10])
            # perform group checks
            assert len(groups) == len(epochs), (
                "Group count does not match epoch count."
            )
            assert np.all(groups >= 0), (
                "An epoch occurs before the first trial."
            )

            print("Grouping checks passed.")

            # ------------------------------------------------------------
            # Start the 5 Fold Cross Validation.
            # ------------------------------------------------------------

            # Reset results for the current subject.
            for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                # No block may appear in both train and test.
                assert_no_group_overlap(groups, train_idx, test_idx)
                shared_groups = np.intersect1d(groups[train_idx], groups[test_idx])
                # GET TRAIN AND TEST EPOCHS
                train_epochs = epochs[train_idx]
                test_epochs = epochs[test_idx]
                # TRAIN
                self.model.fit(train_epochs)
                #PREDICT
                fold_probabilities = self.model.predict_proba(test_epochs)
                fold_predictions = np.where(
                fold_probabilities >= CLASSIFICATION_THRESHOLD,
                LABELS["target"],
                LABELS["nontarget"]
                    )

                fold_y_true = self.model.get_last_transformed_epochs().events[:, 2]
                # START EVALUATING THIS FOLD.
                metrics,_,cm,_=self.evaluator.evaluate(y_true=fold_y_true,y_pred=fold_predictions,y_prob=fold_probabilities)
                self.fold_results.append({"fold": fold, **metrics})
                # APPEND train-test-arrangment of indices used in this fold.
                self.split_records.append({
                    "fold": fold,
                    "train_idx": train_idx.tolist(),
                    "test_idx": test_idx.tolist(),
                    "train_block_ids": sorted(np.unique(groups[train_idx]).astype(int).tolist()),
                    "test_block_ids": sorted(np.unique(groups[test_idx]).astype(int).tolist()),
                })

                # Rows are actual classes; columns are predicted classes.
                self.total_cm += cm

                print(
                    f"Fold {fold}: "
                    f"train={len(train_idx)}, "
                    f"test={len(test_idx)}, "
                    f"shared_blocks={len(shared_groups)}"
                )
                print(f"\nFold {fold}")
                print(f"Accuracy: {metrics['accuracy']:.3f}")
                print(f"Target precision: {metrics['precision']:.3f}")
                print(f"Target recall: {metrics['recall']:.3f}")
                print(f"Target F1: {metrics['f1_score']:.3f}")
                print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
                print(f"PR-AUC (average precision): {metrics['pr_auc']:.3f}")
                print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
                print("Confusion matrix:")
                print(cm)

            self.save_subject_folds_results(subject_id=subject_id,epochs_path=epochs_path,raw_path=raw_path)
            self.all_subjects_summaries.append(RESULT_DIR / f"subject_{subject_id}_summary.csv")

        # Combine summaries from all eight subjects.
        all_subjects = pd.concat([pd.read_csv(path) for path in self.all_subjects_summaries],ignore_index=True)
        all_subjects.to_csv(
            RESULT_DIR / "all_subjects_summary.csv",
            index=False
        )

        # Calculate mean and sample standard deviation across subjects.
        overall = all_subjects[METRICS].agg(["mean", "std"])

        overall.to_csv(
            RESULT_DIR / "overall_summary.csv"
        )
        self.is_active=True
    def save_subject_folds_results(self,subject_id,epochs_path,raw_path):
        # ------------------------------------------------------------
        # Save results of each subject.
        # ------------------------------------------------------------

        # create folder
        RESULT_DIR.mkdir(parents=True, exist_ok=True)

        # Save each fold's Metrics.
        results_df = pd.DataFrame(self.fold_results)

        results_path = RESULT_DIR / f"subject_{subject_id}_fold_metrics.csv"
        results_df.to_csv(results_path, index=False)

        print("Fold metrics saved to:", results_path)

        # Save mean fold scores for this subject.
        summary = results_df[METRICS].mean().to_frame().T
        summary.insert(0, "subject", subject_id)

        summary_path = RESULT_DIR / f"subject_{subject_id}_summary.csv"
        summary.to_csv(summary_path, index=False)
        print("Subject summary saved to:", summary_path)

        # SAVE train-test-arrangment of indices used in each fold.
        splits_path = RESULT_DIR / f"subject_{subject_id}_splits.json"

        with splits_path.open("w", encoding="utf-8") as file:
            json.dump(self.split_records, file, indent=2)
        print("Split indices saved to:", splits_path)

        # Save the subject's combined confusion matrix.
        cm_df = pd.DataFrame(
            self.total_cm,
            index=["Actual NonTarget", "Actual Target"],
            columns=["Predicted NonTarget", "Predicted Target"]
        )

        cm_path = RESULT_DIR / f"subject_{subject_id}_confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        print("Confusion matrix saved to:", cm_path)

        # Save the run settings and input fingerprints next to the saved folds.
        # This records provenance without changing the existing evaluation logic.
        manifest = {
            "subject_id": subject_id,
            "seed": SEED,
            "splitter": "src.validation.block_split.make_block_splits (StratifiedGroupKFold)",
            "n_splits": 5,
            "shuffle": True,
            "grouping": {
                "level": "block",
                "group_id": "zero-based block index within subject session",
                "source": "derived from original trial starts in the subject MAT file",
                "mapping": "block_id = trial_id // 5; five consecutive character trials per block",
                "blocks_per_session": BLOCKS_PER_SESSION,
                "trials_per_block": TRIALS_PER_BLOCK,
            },
            "labels": {"NonTarget": 1, "Target": 2},
            "feature_configuration": {
                "extractor": "FeatureExtraction",
                "cropping_window_ms": list(ext.FeatureExtraction().cropping_window),
                "n_components": ext.FeatureExtraction().n_components,
                "window_threshold": ext.FeatureExtraction().threshold,
                "window_offset_samples": ext.FeatureExtraction().offset,
            },
            "artifact_rejection": {
                "method": "autoreject",
                "fit_scope": "training epochs within each fold",
            },
            "preprocessed_file": str(epochs_path.relative_to(PROJECT_ROOT)),
            "preprocessed_sha256": sha256_file(epochs_path),
            "raw_trial_file": str(raw_path.relative_to(PROJECT_ROOT)),
            "raw_trial_sha256": sha256_file(raw_path),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "pandas": pd.__version__,
                "mne": mne.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "folds": self.split_records,
        }

        manifest_path = RESULT_DIR / f"subject_{subject_id}_run_manifest.json"

        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)

        print("Run manifest saved to:", manifest_path)

        # Combine test counts from all five folds.
        print("================================")
        print(f"\nSubject {subject_id} - Combined confusion matrix:")
        print(self.total_cm)
        print("================================")

        print(f"\nSubject {subject_id} - Mean scores across 5 folds")
        for metric in METRICS:
            values = [result[metric] for result in self.fold_results]
            print(f"Mean {metric}: {np.mean(values):.3f}")


def assert_no_group_overlap(groups, train_idx, test_idx):
    """Raise if any group appears in both sides of a fold."""
    shared_groups = np.intersect1d(groups[train_idx], groups[test_idx])
    if shared_groups.size:
        raise AssertionError(
            f"Groups appear in both train and test: {shared_groups.tolist()}"
        )

def sha256_file(path):
    """Return a stable fingerprint for an input file used by this run."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    SubjectDependent5FoldCV().run()
