import mne
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from src.Utility.config import (SEED,FOLDS,find_project_root,NUM_SUBJECTS,TRIALS_PER_BLOCK,N_TRIALS,TEST_SIZE)

def get_block_ids(mat_path, epochs):
    """
    Assigns each EEG epoch to its corresponding experimental block.

    Limitations / Assumptions:
    - trial_starts must contain the sample index at which each trial begins.
    - trial_starts must be sorted in ascending order.
    - trial_starts and epochs.events[:, 0] must use the same sample coordinate system.
    - Trial-start indices must be zero-based (MATLAB indices must be converted beforehand).
    - Each block must contain a fixed number of consecutive trials.
    - trials_per_block must be a positive integer.
    - Epochs must not occur before the first trial start.
    - Every epoch is assigned to the most recent trial start preceding or
      equal to its event sample.

    Returns:
        block_ids: Zero-based block ID for each epoch.
    """
    # Loads trial-start sample positions from the original MATLAB dataset, identifies the most recent trial preceding each EEG epoch, and determines the corresponding block using the number of trials per block.
    mat = loadmat(mat_path, simplify_cells=True)  ## loads matlab file into python

    # the -1 is because Matlab doesn't Counts from sample Zero
    # np.asarray(mat["data"]["trial"]) return the sample Where Every Trial Starts
    trial_starts = np.asarray(mat["data"]["trial"]).ravel().astype(int) - 1
    
    if len(trial_starts) == 0:
        raise ValueError("No trial starts found")

    if np.any(np.diff(trial_starts) <= 0):
        raise ValueError("Trial starts must be strictly increasing")

    if TRIALS_PER_BLOCK <= 0:
        raise ValueError("TRIALS_PER_BLOCK must be positive")
    if len(trial_starts) != N_TRIALS:
        raise ValueError(f"Expected {N_TRIALS} trials, found {len(trial_starts)}")


    # which trial does each epoch belong to?
    ## Why -1 : We want the index of the most recent trial start that occurred before the event.
    trial_ids = np.searchsorted(trial_starts, epochs.events[:, 0], side="right") - 1
    if np.any(trial_ids < 0):
        raise ValueError("Some epochs occur before the first trial")

    # which block does each epoch belong to?
    block_ids = trial_ids // TRIALS_PER_BLOCK
    return block_ids


class EpochManager:

    def __init__(self):
        self.is_split = False
        self.train_dir = find_project_root() / "data" / "split_epochs" / "train"
        self.test_dir = find_project_root() / "data" / "split_epochs" / "test"

    def load_epochs(self, subject_id, split="train"):
        if not self.is_split:
            raise RuntimeError("the epochs are not yet split")
        directory = self.train_dir if split == "train" else self.test_dir

        return mne.read_epochs(
            directory / f"subject_{subject_id}_{split}-epo.fif",
            preload=True,
            verbose=False
        )

    def make_holdout_evaluation_splits(self):
        # Create output directories

        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Process all 8 subjects
        for i in range(1, NUM_SUBJECTS+1):
            # ------------------------------------------------------------
            # Load preprocessed epochs
            # ------------------------------------------------------------
            epochs_path = (
                    find_project_root()
                    / "data"
                    / "preprocessed"
                    / f"subject_{i}_clean-epo.fif"
            )

            epochs = mne.read_epochs(
                epochs_path,
                preload=True,
                verbose=False
            )

            # ------------------------------------------------------------
            # Read original trial boundaries
            # ------------------------------------------------------------
            raw_path = (
                    find_project_root()
                    / "data"
                    / "moabb"
                    / "MNE-bnci-data"
                    / "~bci"
                    / "database"
                    / "008-2014"
                    / f"A{i:02d}.mat"
            )

            block_ids = get_block_ids(mat_path=raw_path, epochs=epochs)

            # ------------------------------------------------------------
            # Validate groups
            # ------------------------------------------------------------
            assert len(block_ids) == len(epochs), (
                "Group count does not match epoch count."
            )

            # ------------------------------------------------------------
            # Split epochs
            # ------------------------------------------------------------
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=TEST_SIZE,
                random_state=SEED
            )

            train_idx, test_idx = next(
                splitter.split(
                    X=epochs.events,
                    groups=block_ids
                )
            )

            training_epochs = epochs[train_idx]
            testing_epochs = epochs[test_idx]

            # ------------------------------------------------------------
            # Save training and testing epochs
            # ------------------------------------------------------------
            training_epochs.save(
                self.train_dir / f"subject_{i}_train-epo.fif",
                overwrite=True,
                verbose=False
            )

            testing_epochs.save(
                self.test_dir / f"subject_{i}_test-epo.fif",
                overwrite=True,
                verbose=False
            )

        print("All subjects' epochs have been split and saved.")
        self.is_split = True

    def make_cv_splits(self,epochs, labels, mat_path, n_splits=FOLDS, seed=SEED):
        """
        Returns:
            for a certain Subject
            splits:    list of (train_idx, test_idx) pairs
        """
        # Step 1:get block ids
        block_ids = get_block_ids(mat_path=mat_path, epochs=epochs)

        # Step 2: split, keeping each block entirely in train or entirely in test
        # The data is divided into 5 folds. In each iteration:
        # 1 fold is used for validation.
        # The remaining 4 folds are used for training.
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # evaluate the labels
        labels = np.asarray(labels)
        if len(labels) != len(epochs):
            raise ValueError("labels and epochs must have the same length")
        ## np.zeros(len(labels)) is place Holder For the Needed Features (X)
        ## it returns array of splits
        # each split has two arrays --> [train idx] , [test idx]
        splits = list(cv.split(np.zeros(len(labels)), labels, block_ids))

        # Step 3: safety check, no block may appear on both sides
        for train_idx, test_idx in splits:
            if set(block_ids[train_idx]) & set(block_ids[test_idx]):
                raise AssertionError("Block leakage detected")

        return splits







