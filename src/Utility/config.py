import hashlib
from pathlib import Path
### METHODS
def find_project_root(marker="Pipfile.lock"): ## The _name is a Python convention meaning private helper, only for use inside this file.
    for parent in Path(__file__).resolve().parents: ## parents are returned as addresses
        if (parent / marker).exists():
            print(parent)
            return parent
    raise RuntimeError(f"Could not find {marker}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
### CONSTANTS
# GENERAL
SEED = 42
DATASET_NAME = "BNCI2014-008"
LABELS = {
    "target": 2,
    "nontarget": 1,
}
NUM_SUBJECTS=8
BLOCKS_PER_SESSION=7
TRIALS_PER_BLOCK=5
TRIALS_PER_SESSION = BLOCKS_PER_SESSION * TRIALS_PER_BLOCK
N_TRIALS=35 # 7 blocks x 5 trials
PROJECT_ROOT = find_project_root()
# Evaluation
TEST_SIZE = 0.2
FOLDS=5
RESULT_DIR = PROJECT_ROOT / "results" / "subject_dependent"
EVALUATION_NAME = "LDA evaluation"
EVALUATION_RESULTS_DIR = PROJECT_ROOT / "results" / EVALUATION_NAME / DATASET_NAME
EVALUATION_FIGURES_DIR = PROJECT_ROOT / "figures" / EVALUATION_NAME / DATASET_NAME
LOSO_FOLDS_PATH = RESULT_DIR / "loso_fold_metrics.csv"
LOSO_SUMMARY_PATH = RESULT_DIR / "loso_summary.csv"
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
]
# CLASSIFICATION
CLASSIFICATION_THRESHOLD=0.5
# PREPROCESSING
ARTIFACT_REJECTION = "autoreject"
PREPROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"

