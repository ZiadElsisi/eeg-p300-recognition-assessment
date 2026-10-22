# EEG P300 Recognition Assessment Platform

## Overview

The EEG P300 Recognition Assessment Platform is a graduation project focused on developing an AI-based platform for EEG signal acquisition, P300 Event-Related Potential (ERP) analysis, and P300 recognition.

The platform is planned to support:

- EEG signal acquisition
- Visual stimulus presentation
- EEG preprocessing
- P300/ERP extraction
- P300 feature extraction
- AI/ML-based P300 recognition
- Experiment management
- Visualization
- Technical reporting

The project is being developed incrementally, starting with a reproducible Python development environment and project structure.

## Project Structure

```text
eeg-p300-recognition-assessment/
│
├── data/
│   └── .gitkeep
│
├── notebooks/
│   └── .gitkeep
│
├── src/
│   ├── loaders/
│   │   └── __init__.py
│   ├── preprocessing/
│   │   └── __init__.py
│   ├── features/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   └── simulation/
│       └── __init__.py
│
├── tests/
│   └── .gitkeep
│
├── results/
│   └── .gitkeep
│
├── figures/
│   └── .gitkeep
│
├── .gitignore
├── Pipfile
├── Pipfile.lock
└── README.md
```

### Directory Description

| Directory | Purpose |
|---|---|
| `data/` | EEG datasets and data files |
| `notebooks/` | Exploratory analysis and experiments |
| `src/loaders/` | EEG and data loading modules |
| `src/preprocessing/` | EEG preprocessing modules |
| `src/features/` | P300 feature extraction |
| `src/models/` | Machine-learning models |
| `src/evaluation/` | Model and pipeline evaluation |
| `src/simulation/` | Simulation-related modules |
| `tests/` | Automated tests |
| `results/` | Experiment and evaluation results |
| `figures/` | Generated figures and plots |

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.11 |
| EEG Acquisition | BrainFlow |
| Signal Processing | MNE-Python, NumPy, SciPy |
| Machine Learning | scikit-learn |
| Baseline Classifier | LDA |
| Deep Learning | PyTorch / EEGNet |
| Backend | FastAPI |
| Frontend | React + TypeScript |
| Database | PostgreSQL |
| Version Control | Git + GitHub |
| Environment Management | Pipenv |
| Containerization | Docker |

## EEG Processing Pipeline

The planned EEG/P300 processing workflow includes:

1. EEG acquisition
2. Signal preprocessing
3. Epoch extraction using event markers
4. Baseline correction
5. Artifact rejection
6. ERP/P300 generation
7. P300 feature extraction
8. Machine-learning classification
9. Evaluation and reporting

The baseline classification approach uses Linear Discriminant Analysis (LDA). EEGNet with PyTorch may also be explored as an additional model.

### ERP channel policy and interpretation

Pz is the **pre-specified primary channel** for the BNCI2014-008 ERP analysis. Fz is reported only as an exploratory secondary channel; it was not promoted because it produced a larger response. The current per-subject analysis finds a positive Pz Target-minus-NonTarget mean in 4 of 8 subjects, compared with 7 of 8 at exploratory Fz. The equal-subject-weighted grand average and per-subject values are saved under `results/erp/BNCI2014-008/channel_comparison/`.

The stored epoch metadata reports no custom reference applied and a zero-sample offset of 0 ms for every subject. This confirms alignment to the event marker saved in the epochs, but cannot establish the original acquisition-time trigger accuracy or recover undocumented reference changes made before saving.

### Artifact threshold policy

The current public-dataset preprocessing code rejects an epoch when any selected EEG sample exceeds **±70 µV absolute amplitude** after baseline correction. This is an absolute-amplitude rule, not a 150 µV peak-to-peak rule. A report that describes the current implementation as 150 µV peak-to-peak is inaccurate and should be corrected unless the code and results are deliberately regenerated under that rule. The hardware's proposed ±100 µV threshold is a separate provisional value; finalize it only after a Cyton pilot and do not treat it as equivalent to the public-data rule.

### Subject-dependent cross-validation and leakage control

The subject-dependent evaluation uses stratified five-fold group cross-validation with **block ID** as the group. The source BNCI 2014-008 MAT files contain trial start samples but no explicit block field. Dataset metadata specifies 35 character trials (seven five-character words) and seven blocks per session; the implementation maps each five consecutive trial IDs to one block (`block_id = trial_id // 5`) and stops if the expected 35 trial starts are absent. Each epoch is mapped to its trial from its event sample and the original trial starts, then to the corresponding block. All epochs from a block stay together in one fold. Runtime checks and automated tests assert that train and test block IDs do not overlap.

### Imbalanced-class evaluation policy

The BNCI2014-008 evaluation has an approximately 1:5 Target-to-NonTarget ratio. Accuracy is reported for context and is never a success criterion by itself. Target (label `2`) is the positive class. Subject-dependent five-fold and LOSO results report Target precision, recall, and F1, ROC-AUC, PR-AUC (average precision), and balanced accuracy for each fold or held-out subject; summaries report the mean and standard deviation across folds/subjects.

For this imbalanced dataset, the evaluation's **above-chance target-detection criterion** is that the grouped-CV macro means exceed their no-skill baselines: ROC-AUC > 0.5, balanced accuracy > 0.5, and PR-AUC > the observed Target prevalence (about 0.167). Target F1, recall, and precision at the model's default decision threshold must also be reported to show the precision/recall tradeoff. These criteria support an above-chance finding; they do not establish application readiness. No minimum application-specific Target precision has been agreed, so no stronger usefulness/pass threshold is claimed. Any future operating-threshold selection must use training data only, not the held-out fold.

On the current subject-dependent grouped five-fold run, the across-subject means are Target F1 0.572, Target precision 0.460, Target recall 0.761, ROC-AUC 0.866, PR-AUC/average precision 0.644, and balanced accuracy 0.789. The ROC-AUC, PR-AUC, and balanced-accuracy means exceed their stated baselines; accuracy (0.808) is descriptive only.

### Environment and code traceability

The test runs used **Python 3.11.16**. The current recorded Git commit, source fingerprint, and test result are listed in `results/environment_traceability.md`. No `prehardware-v1.0` release tag exists. The test inventory is eight main UT-01–UT-08 tests plus three parameterized numeric timing-shift cases; it is not an 18-test suite.

## Development Status

The repository currently contains the initial project structure and reproducible Python development environment.

The pre-hardware work currently includes public-dataset inspection and preprocessing, ERP analysis, LDA evaluation, subject-dependent grouped cross-validation, LOSO evaluation, and synthetic P300 software checks. Synthetic and public-dataset results do not replace validation on physical EEG hardware.

---

# Setup Guide

This section describes how to set up the project on Windows.

## Requirements

- Windows 10/11
- Git
- Python 3.11
- Pipenv

## 1. Verify Python

Open PowerShell or Git Bash and run:

```bash
python --version
```

The required version is:

```text
Python 3.11.x
```

If Python 3.11 is not the default Python version, verify it with:

```bash
py -3.11 --version
```

## 2. Clone the Repository

```bash
git clone <repository-url>
cd eeg-p300-recognition-assessment
```

## 3. Install Pipenv

Using Python 3.11:

```bash
py -3.11 -m pip install pipenv
```

Verify the installation:

```bash
pipenv --version
```

## 4. Create the Python Environment

Create the project's virtual environment using Python 3.11:

```bash
pipenv --python 3.11
```

Check the virtual environment:

```bash
pipenv --venv
```

Check the Python interpreter:

```bash
pipenv --py
```

## 5. Install Dependencies

Install the dependencies defined in the project:

```bash
pipenv install
```

The main Python dependencies include:

- NumPy
- SciPy
- pandas
- MNE-Python
- scikit-learn
- matplotlib
- MOABB

For development and testing:

```bash
pipenv install --dev pytest
```

## 6. Activate the Environment

```bash
pipenv shell
```

Alternatively, commands can be executed through Pipenv without entering the shell:

```bash
pipenv run python --version
```

## 7. Verify the Environment

Check the Python version:

```bash
pipenv run python --version
```

Expected:

```text
Python 3.11.x
```

Verify the required packages:

```bash
pipenv run python -c "import numpy, scipy, pandas, mne, sklearn, matplotlib, moabb; print('Environment PASS')"
```

Expected:

```text
Environment PASS
```

## 8. Record Exact Package Versions

Display the exact versions installed in the project environment:

```bash
pipenv run python -c "import numpy, scipy, pandas, mne, sklearn, matplotlib, moabb; print('Python:', __import__('sys').version.split()[0]); print('NumPy:', numpy.__version__); print('SciPy:', scipy.__version__); print('pandas:', pandas.__version__); print('MNE:', mne.__version__); print('scikit-learn:', sklearn.__version__); print('matplotlib:', matplotlib.__version__); print('MOABB:', moabb.__version__)"
```

Display all installed packages:

```bash
pipenv run pip freeze
```

## 9. Reproducibility and Lock File

The project uses `Pipfile.lock` to record the locked dependency versions.

Update the lock file when dependencies change:

```bash
pipenv lock
```

Install the exact locked dependencies:

```bash
pipenv sync
```

Verify the environment again:

```bash
pipenv run python -c "import numpy, scipy, pandas, mne, sklearn, matplotlib, moabb; print('Environment PASS')"
```

## 10. Testing

Run the project's automated tests:

```bash
pipenv run pytest
```

## 11. Git Workflow

Check the repository status:

```bash
git status
```

Add changes:

```bash
git add .
```

Commit changes:

```bash
git commit -m "Update project setup"
```

Verify the working tree:

```bash
git status
```

A clean repository should show:

```text
nothing to commit, working tree clean
```
