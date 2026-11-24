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
from src.validation.epoch_splitting import EpochManager
# importing constants
from src.Utility.config import (SEED,DATASET_NAME,LABELS,FOLDS,NUM_SUBJECTS,ARTIFACT_REJECTION,EVALUATION_RESULTS_DIR,EVALUATION_FIGURES_DIR,CLASSIFICATION_THRESHOLD)




class EEGClassifier:
    """Default xDAWN feature extraction, scaling, and LDA classifier."""

    def __init__(
        self,
        estimator=None,
        feature_extractor=None,
        artifact_rejection=None,
        artifact_method=ARTIFACT_REJECTION,
    ):
        self.model =(
            estimator
            if estimator is not None
            else Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LDA(priors=[0.5, 0.5]))
            ])
        )
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
        self.is_fitted = False
        self.last_transformed_epochs_ = None

    def fit(self, epochs):
        # Artifact rejection
        self.artifact_rejection.fit(epochs)
        clean_epochs, _ = self.artifact_rejection.transform(epochs)
        if len(clean_epochs) == 0:
            raise ValueError("Artifact rejection removed every training epoch.")
        # Feature extraction
        features = self.feature_extractor.fit_transform(clean_epochs)
        # Target extraction
        targets = clean_epochs.events[:, 2]
        # Train
        self.model.fit(features, targets)

        self.is_fitted = True
        return self

    def predict_proba(self, epochs):
        if not self.is_fitted:
            raise RuntimeError("The classifier must be fitted first.")
        # Artifact rejection
        clean_epochs, _ = self.artifact_rejection.transform(epochs)
        if len(clean_epochs) == 0:
            raise ValueError("Artifact rejection removed every prediction epoch.")
        self.last_transformed_epochs_ = clean_epochs
        # Feature extraction
        features = self.feature_extractor.transform(clean_epochs)
        # get Target column
        target_column = np.flatnonzero(
            self.model.classes_ == LABELS["target"]
        )[0]
        return self.model.predict_proba(features)[:, target_column]

    def predict(self, epochs, threshold=CLASSIFICATION_THRESHOLD):
        target_probabilities = self.predict_proba(epochs)
        return np.where(
            target_probabilities >= threshold,
            LABELS["target"],
            LABELS["nontarget"],
        )
    def get_last_transformed_epochs(self):
        if self.last_transformed_epochs_ is None:
            raise RuntimeError("last transformed epochs not available")
        return self.last_transformed_epochs_


class EvaluationReporter:
    def __init__(self, results_dir=EVALUATION_RESULTS_DIR, figures_dir=EVALUATION_FIGURES_DIR):
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, y_true, y_pred, y_prob):
        """
           Evaluate the classification model using several performance metrics
           and generate the data required for evaluation plots.

           Parameters
           ----------
           y_true : array-like
               The actual class labels (Non-Target and Target).

           y_pred : array-like
               The class labels predicted by the model.

           y_prob : array-like
               The predicted probabilities for the Target class.

           Returns
           -------
           metrics : dict
               Classification performance metrics:
                {"accuracy":,"precision":,"recall":,"balanced_accuracy":,"f1_score":,"roc_auc":,"pr_auc":,}

           pr_data : dict
               Precision-recall curve data containing
               {"pr_precision":,"pr_recall":,}

           matrix : ndarray
               Confusion matrix with rows and columns ordered as
               Non-Target and Target.

           roc_data : dict
               ROC curve data containing
               {"false_positive_rate":,"true_positive_rate":}
           """
        pr_precision, pr_recall, _ = precision_recall_curve(
            y_true, y_prob, pos_label=LABELS["target"]
        )
        pr_auc = round(auc(pr_recall, pr_precision), 2)
        metrics = {
            "accuracy": round(accuracy_score(y_true, y_pred), 2),
            "precision": round(precision_score(
                y_true, y_pred, pos_label=LABELS["target"], zero_division=0
            ), 2),
            "recall": round(recall_score(
                y_true, y_pred, pos_label=LABELS["target"], zero_division=0
            ), 2),
            "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 2),
            "f1_score": round(f1_score(
                y_true, y_pred, pos_label=LABELS["target"], zero_division=0
            ), 2),
            "roc_auc": round(roc_auc_score(y_true == LABELS["target"], y_prob), 2),
            "pr_auc": pr_auc,
        }
        matrix = confusion_matrix(y_true, y_pred, labels=[LABELS["nontarget"], LABELS["target"]])
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true, y_prob, pos_label=LABELS["target"]
        )
        print(classification_report(
            y_true, y_pred, target_names=["Non Target", "Target"], zero_division=0
        ))
        return metrics,{"pr_precision":pr_precision,"pr_recall":pr_recall},matrix,{"false_positive_rate":false_positive_rate,"true_positive_rate":true_positive_rate}
    def hold_out_evaluation(self,subjects,epoch_manager: EpochManager):
        # create the fixed splitting if not already split
        if not epoch_manager.is_split:
            epoch_manager.make_holdout_evaluation_splits()
        # load the training and test epochs
        training_epochs = []
        testing_epochs = []

        for subject_id in subjects:
            training_epochs.append(
                epoch_manager.load_epochs(subject_id, "train")
            )

            testing_epochs.append(
                epoch_manager.load_epochs(subject_id, "test")
            )

        # Combine training subjects into 1 epoch collection
        combined_train = mne.concatenate_epochs(training_epochs)

        # Combine testing subjects into 1 epoch collection
        combined_test = mne.concatenate_epochs(testing_epochs)
        # Train one common pipeline
        model = EEGClassifier()
        model.fit(combined_train)

        # Predict
        predictions = model.predict(combined_test)
        probabilities = model.predict_proba(combined_test)

        y_true = model.get_last_transformed_epochs().events[:, 2]
        # prepare for the subject column in evaluation result table
        selected_subjects = {}
        for idx in range(1, NUM_SUBJECTS+1):
            if idx in subjects:
                selected_subjects[f"subject_{idx}"] = 1
            else:
                selected_subjects[f"subject_{idx}"] = 0
        # Evaluate
        metrices,pr_auc_metrices,cm,roc_auc_metrices=self.evaluate(y_true, predictions, probabilities)
        # percision/Recall Auc curve
        plt.plot(pr_auc_metrices["pr_recall"], pr_auc_metrices["pr_precision"], label=f"AUC = {metrices['pr_auc']}")
        baseline = np.mean(np.asarray(y_true) == LABELS["target"])
        plt.axhline(y=baseline, linestyle="--", color="gray", label="Baseline")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{DATASET_NAME} — LDA Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.figures_dir / f"{DATASET_NAME}_LDA_PR_AUC_curve.png")
        plt.close()
        # other metrices
        pd.DataFrame.from_dict([selected_subjects | metrices]).to_csv(
            self.results_dir / f"{DATASET_NAME}_LDA_holdout_metrics.csv",
            index=False,
        )
        # confusion matrix
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["not target", "target"],
        )
        display.plot()
        plt.title(f"{DATASET_NAME} — LDA Confusion Matrix")
        plt.savefig(self.figures_dir / f"{DATASET_NAME}_LDA_confusion_matrix.png")
        plt.close()
        #LDA ROC Curve
        plt.plot(roc_auc_metrices["false_positive_rate"], roc_auc_metrices["true_positive_rate"], label=f"AUC = {metrices['roc_auc']}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title(f"{DATASET_NAME} — LDA ROC Curve — Target vs NonTarget")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.figures_dir / f"{DATASET_NAME}_LDA_ROC_curve.png")
        plt.close()

    def cross_validation(self,subjects,epoch_manager: EpochManager):

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
            splits = epoch_manager.make_cv_splits(
                epochs, epochs.events[:, 2], raw_path, n_splits=5, seed=SEED
            )
            subject_epochs[subject_id] = epochs
            subject_splits[subject_id] = splits

        y_true_parts = []
        prediction_parts = []
        probability_parts = []
        fold_metric_rows = []

        for fold in range(FOLDS):
            training_epochs = []
            testing_epochs = []
            for subject_id in subjects:
                # get the subject all epochs
                epochs = subject_epochs[subject_id]
                # get the subject fold train and test ids "No leakage accross blocks"
                train_idx, test_idx = subject_splits[subject_id][fold]
                # use the indices obtained and the epochs to get the train and test epochs
                training_epochs.append(epochs[train_idx])
                testing_epochs.append(epochs[test_idx])
            # all training epochs (no leakage)
            combined_train = mne.concatenate_epochs(training_epochs)
            # all testing epochs (no leakage)
            combined_test = mne.concatenate_epochs(testing_epochs)
            # train using the train epochs
            model = EEGClassifier()
            model.fit(combined_train)
            fold_probabilities = model.predict_proba(combined_test)
            fold_predictions = model.predict(combined_test)
            # get the true values of this fold test epochs
            fold_y_true = model.get_last_transformed_epochs().events[:, 2]
            y_true_parts.append(fold_y_true)
            prediction_parts.append(fold_predictions)
            probability_parts.append(fold_probabilities)
            fold_metric_rows.append({
                "subjects": ", ".join(map(str, subjects)),
                "fold": fold + 1,
                "artifact_method": ARTIFACT_REJECTION,
                "support": len(fold_y_true),
                "accuracy": round(accuracy_score(fold_y_true, fold_predictions), 2),
                "target_precision": round(precision_score(
                    fold_y_true, fold_predictions, pos_label=LABELS["target"],
                    zero_division=0,
                ), 2),
                "target_recall": round(recall_score(
                    fold_y_true, fold_predictions, pos_label=LABELS["target"],
                    zero_division=0,
                ), 2),
                "target_f1": round(f1_score(
                    fold_y_true, fold_predictions, pos_label=LABELS["target"],
                    zero_division=0,
                ), 2),
                "balanced_accuracy": round(balanced_accuracy_score(
                    fold_y_true, fold_predictions
                ), 2),
            })
            pd.DataFrame(fold_metric_rows).to_csv(
                EVALUATION_RESULTS_DIR / f"{DATASET_NAME}_LDA_block_cv_fold_metrics.csv",
                index=False,
            )


def run_holdout_evaluation(subjects: list[int]):
    reporter = EvaluationReporter(EVALUATION_RESULTS_DIR, EVALUATION_FIGURES_DIR)
    epoch_manager = EpochManager()
    reporter.cross_validation(subjects=subjects,epoch_manager=epoch_manager)
    reporter.hold_out_evaluation(subjects=subjects,epoch_manager=epoch_manager)



if __name__ == "__main__":
    run_holdout_evaluation(subjects=[8])
