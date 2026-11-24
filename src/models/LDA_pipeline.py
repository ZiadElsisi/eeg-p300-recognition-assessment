import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import src.features.feature_extraction as ext
from src.preprocessing.threshold_rejection import ArtifactRejection
from src.Utility.config import PROJECT_ROOT
from src.validation.block_split import make_block_splits


SEED = 42
DATASET_NAME = "BNCI2014-008"
EVALUATION_NAME = "LDA evaluation"
EVALUATION_RESULTS_DIR = PROJECT_ROOT / "results" / EVALUATION_NAME / DATASET_NAME
EVALUATION_FIGURES_DIR = PROJECT_ROOT / "figures" / EVALUATION_NAME / DATASET_NAME

labels = {
    "target": 2,
    "nontarget": 1,
}


class EEGClassifier:
    """Default xDAWN feature extraction, scaling, and LDA classifier."""

    def __init__(
        self,
        feature_extractor=None,
        artifact_rejection=None,
        artifact_method="autoreject",
    ):
        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else ext.FeatureExtraction()
        )
        self.artifact_rejection = (
            artifact_rejection
            if artifact_rejection is not None
            else ArtifactRejection(method=artifact_method)
        )
        self.artifact_method = artifact_method
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LDA(priors=[0.5, 0.5])),
        ])
        self.is_fitted = False
        self.last_transformed_epochs_ = None

    def fit(self, epochs):
        self.artifact_rejection.fit(epochs)
        clean_epochs, _ = self.artifact_rejection.transform(epochs)
        if len(clean_epochs) == 0:
            raise ValueError("Artifact rejection removed every training epoch.")

        features = self.feature_extractor.fit_transform(clean_epochs)
        targets = clean_epochs.events[:, 2]
        self.model.fit(features, targets)
        self.is_fitted = True
        return self

    def predict_proba(self, epochs):
        if not self.is_fitted:
            raise RuntimeError("The classifier must be fitted first.")

        clean_epochs, _ = self.artifact_rejection.transform(epochs)
        if len(clean_epochs) == 0:
            raise ValueError("Artifact rejection removed every prediction epoch.")
        self.last_transformed_epochs_ = clean_epochs
        features = self.feature_extractor.transform(clean_epochs)
        target_column = np.flatnonzero(
            self.model.classes_ == labels["target"]
        )[0]
        return self.model.predict_proba(features)[:, target_column]

    def predict(self, epochs, threshold=0.5):
        target_probabilities = self.predict_proba(epochs)
        return np.where(
            target_probabilities >= threshold,
            labels["target"],
            labels["nontarget"],
        )


class EvaluationReporter:
    def __init__(self, results_dir, figures_dir):
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, y_true, y_pred, y_prob, selected_subjects):
        pr_precision, pr_recall, _ = precision_recall_curve(
            y_true, y_prob, pos_label=labels["target"]
        )
        pr_auc = round(auc(pr_recall, pr_precision), 2)
        plt.plot(pr_recall, pr_precision, label=f"AUC = {pr_auc}")
        baseline = np.mean(np.asarray(y_true) == labels["target"])
        plt.axhline(y=baseline, linestyle="--", color="gray", label="Baseline")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{DATASET_NAME} — LDA Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.figures_dir / f"{DATASET_NAME}_LDA_PR_AUC_curve.png")
        plt.close()

        metrics = {
            "accuracy": round(accuracy_score(y_true, y_pred), 2),
            "precision": round(precision_score(
                y_true, y_pred, pos_label=labels["target"], zero_division=0
            ), 2),
            "recall": round(recall_score(
                y_true, y_pred, pos_label=labels["target"], zero_division=0
            ), 2),
            "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 2),
            "f1_score": round(f1_score(
                y_true, y_pred, pos_label=labels["target"], zero_division=0
            ), 2),
            "roc_auc": round(roc_auc_score(y_true == labels["target"], y_prob), 2),
            "pr_auc": pr_auc,
        }
        pd.DataFrame.from_dict([selected_subjects | metrics]).to_csv(
            self.results_dir / f"{DATASET_NAME}_LDA_holdout_metrics.csv",
            index=False,
        )

        matrix = confusion_matrix(y_true, y_pred, labels=[1, 2])
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["not target", "target"],
        )
        display.plot()
        plt.title(f"{DATASET_NAME} — LDA Confusion Matrix")
        plt.savefig(self.figures_dir / f"{DATASET_NAME}_LDA_confusion_matrix.png")
        plt.close()

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true, y_prob, pos_label=labels["target"]
        )
        plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {metrics['roc_auc']}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title(f"{DATASET_NAME} — LDA ROC Curve — Target vs NonTarget")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.figures_dir / f"{DATASET_NAME}_LDA_ROC_curve.png")
        plt.close()

        print(classification_report(
            y_true, y_pred, target_names=["Non Target", "Target"], zero_division=0
        ))


def run_holdout_evaluation(subjects: list[int]):
    reporter = EvaluationReporter(EVALUATION_RESULTS_DIR, EVALUATION_FIGURES_DIR)
    subject_epochs = {}
    subject_splits = {}

    for subject_id in subjects:
        epochs_path = (
            PROJECT_ROOT / "data" / "preprocessed"
            / f"subject_{subject_id}_clean-epo.fif"
        )
        raw_path = (
            PROJECT_ROOT / "data" / "moabb" / "MNE-bnci-data" / "~bci"
            / "database" / "008-2014" / f"A{subject_id:02d}.mat"
        )
        epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
        _, splits = make_block_splits(
            epochs, epochs.events[:, 2], raw_path, n_splits=5, seed=SEED
        )
        subject_epochs[subject_id] = epochs
        subject_splits[subject_id] = splits

    y_true_parts = []
    prediction_parts = []
    probability_parts = []
    fold_metric_rows = []

    for fold in range(5):
        training_epochs = []
        testing_epochs = []
        for subject_id in subjects:
            epochs = subject_epochs[subject_id]
            train_idx, test_idx = subject_splits[subject_id][fold]
            training_epochs.append(epochs[train_idx])
            testing_epochs.append(epochs[test_idx])

        combined_train = mne.concatenate_epochs(training_epochs)
        combined_test = mne.concatenate_epochs(testing_epochs)
        classifier = EEGClassifier().fit(combined_train)
        fold_probabilities = classifier.predict_proba(combined_test)
        fold_predictions = np.where(
            fold_probabilities >= 0.5,
            labels["target"],
            labels["nontarget"],
        )
        fold_y_true = classifier.last_transformed_epochs_.events[:, 2]

        y_true_parts.append(fold_y_true)
        prediction_parts.append(fold_predictions)
        probability_parts.append(fold_probabilities)
        fold_metric_rows.append({
            "fold": fold + 1,
            "artifact_method": "autoreject",
            "support": len(fold_y_true),
            "accuracy": accuracy_score(fold_y_true, fold_predictions),
            "target_precision": precision_score(
                fold_y_true, fold_predictions, pos_label=labels["target"],
                zero_division=0,
            ),
            "target_recall": recall_score(
                fold_y_true, fold_predictions, pos_label=labels["target"],
                zero_division=0,
            ),
            "target_f1": f1_score(
                fold_y_true, fold_predictions, pos_label=labels["target"],
                zero_division=0,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                fold_y_true, fold_predictions
            ),
        })

    y_true = np.concatenate(y_true_parts)
    predictions = np.concatenate(prediction_parts)
    probabilities = np.concatenate(probability_parts)
    pd.DataFrame(fold_metric_rows).to_csv(
        EVALUATION_RESULTS_DIR / f"{DATASET_NAME}_LDA_block_cv_fold_metrics.csv",
        index=False,
    )

    selected_subjects = {
        f"subject_{subject_id}": int(subject_id in subjects)
        for subject_id in range(1, 9)
    }
    selected_subjects["artifact_rejection_method"] = "autoreject"
    reporter.evaluate(y_true, predictions, probabilities, selected_subjects)


if __name__ == "__main__":
    run_holdout_evaluation(subjects=[8])
