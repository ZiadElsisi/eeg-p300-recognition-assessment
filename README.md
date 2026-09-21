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

## Development Status

The repository currently contains the initial project structure and reproducible Python development environment.

The EEG acquisition, preprocessing, feature extraction, machine-learning, backend, frontend, database, and reporting components will be developed incrementally.

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
