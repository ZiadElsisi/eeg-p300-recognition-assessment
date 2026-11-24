# EEG P300 Recognition Assessment

Python research code for preprocessing and analyzing P300 EEG data from the BNCI2014-008 dataset. The repository includes EEG preprocessing, ERP analysis, feature extraction, LDA classification, and subject-dependent and leave-one-subject-out evaluation.

This is an analysis project. It does not currently contain a website, frontend, API backend, application database, or deep-learning model such as EEGNet.

## Implemented workflow

1. Load BNCI2014-008 data through MOABB.
2. Preprocess EEG with MNE: select channels, extract events, apply a 0.1–30 Hz band-pass filter, and epoch from −0.2 to 0.8 seconds. The current `Epochs` call does not apply baseline correction. The optional absolute-amplitude rejection threshold is configurable and is not enabled by default in `preprocess_eeg`.
3. Extract features using the project’s xDAWN spatial filter and adaptive time windows.
4. Train and evaluate a scikit-learn Linear Discriminant Analysis (LDA) classifier. The block-based five-fold evaluation uses the shared splitter in `src/validation/block_split.py` so epochs from a block stay on one side of each fold. Its artifact-rejection parameters are fitted on training epochs and applied to the test epochs.
5. Save evaluation metrics, split details, preprocessing reports, and plots as local files.

The main event labels are `1 = NonTarget` and `2 = Target`.

## Technology

| Purpose | Technology used in this repository |
|---|---|
| Language and environment | Python 3.11, Pipenv |
| EEG loading and processing | MOABB, MNE-Python |
| Signal and data handling | NumPy, SciPy, pandas |
| Feature extraction | MNE xDAWN and project-defined adaptive feature windows |
| Classification and evaluation | scikit-learn, LDA, AutoReject |
| Plots | Matplotlib |
| Version control | Git |

## Repository layout

```text
src/
├── analysis/       ERP analysis
├── evaluation/     Block-based five-fold and LOSO evaluation
├── features/       xDAWN spatial filtering and feature extraction
├── loaders/        BNCI2014-008 loading through MOABB
├── models/         LDA pipeline and evaluation plots
├── preprocessing/  EEG preprocessing and artifact rejection
├── Reporting/      Preprocessing reports
├── simulation/     Synthetic P300 and timing checks
├── Utility/        Project configuration
└── validation/     Shared block-based train/test splitting

data/               Downloaded and preprocessed data (local)
figures/             Generated plots
notebooks/           Dataset inspection and ERP analysis notebooks
reports/             Test and traceability reports
results/             Evaluation metrics, manifests, and analysis outputs
tests/               Automated checks
```

## Setup

Requirements: Python 3.11 and Pipenv.

```bash
pip install pipenv
pipenv sync --dev
```

MOABB may download the dataset the first time it is requested. Keep downloaded data in the repository's `data/` directory; dataset files are not part of the source code.

## Run the workflows

Preprocess BNCI2014-008 subjects and save epochs and preprocessing reports:

```bash
pipenv run python -m src.preprocessing.pipeline
```

Run subject-dependent five-fold evaluation for subjects 1–8. Folds are created through the shared block splitter, and results are written under `results/subject_dependent/`:

```bash
pipenv run python -m src.evaluation.subject_dependent_5fold
```

Run leave-one-subject-out evaluation and save its metrics under `results/loso/`:

```bash
pipenv run python -m src.evaluation.loso
```

The evaluation workflows expect preprocessed epoch files under `data/preprocessed/`. Block-based splitting also needs each subject's source BNCI2014-008 MAT file so epochs can be assigned to their original blocks.

## Outputs

Generated outputs are stored locally in `results/` and `figures/`. Depending on the workflow, they include fold metrics, summary CSVs, confusion matrices, ROC and precision-recall plots, fold membership, preprocessing reports, and run manifests. Existing results are examples from previous runs; they are not guaranteed to match a new run's results or environment.

## Tests

```bash
pipenv run pytest
```
