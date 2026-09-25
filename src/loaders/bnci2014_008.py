"""
This module provides functions to load the BNCI2014-008 dataset.

When This module is called it checks for the Dataset in  [/ data / moabb ] directory and returns it
or downloads it if it's not already there .
"""

from pathlib import Path
import mne
from moabb.datasets import BNCI2014_008

## Setting the DIR configuration :

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "moabb"


def load_bnci2014_008(subjects=None):

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    mne.set_config(
        "MNE_DATA",
        str(DATA_DIR.resolve()),
        set_env=True,
    )

    dataset = BNCI2014_008()

    if subjects is None:
        subjects = dataset.subject_list

    data = dataset.get_data(
        subjects=subjects
    )

    return dataset, data

