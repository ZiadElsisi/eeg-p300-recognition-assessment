import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedGroupKFold

TRIALS_PER_BLOCK = 5
N_TRIALS = 35  # 7 blocks x 5 trials


def make_block_splits(epochs, labels, mat_path, n_splits=5, seed=42):
    """
    Returns:
        block_ids: block number (0-6) for every epoch
        splits:    list of (train_idx, test_idx) pairs
    """
    labels = np.asarray(labels)
    if len(labels) != len(epochs):
        raise ValueError("labels and epochs must have the same length")

    mat = loadmat(mat_path, simplify_cells=True) ## loads matlab file into python

     # the -1 is because Matlab doesn't Counts from sample Zero
    # np.asarray(mat["data"]["trial"]) return the sample Where Every Trial Starts
    trial_starts = np.asarray(mat["data"]["trial"]).ravel().astype(int) - 1

    if len(trial_starts) != N_TRIALS:
        raise ValueError(f"Expected {N_TRIALS} trials, found {len(trial_starts)}")

    # Step 2: which trial does each epoch belong to?
    ## Why -1 : We want the index of the most recent trial start that occurred before the event.
    trial_ids = np.searchsorted(trial_starts, epochs.events[:, 0], side="right") - 1
    if np.any(trial_ids < 0):
        raise ValueError("Some epochs occur before the first trial")

    # Step 3: which block does each epoch belong to?
    block_ids = trial_ids // TRIALS_PER_BLOCK

    # Step 4: split, keeping each block entirely in train or entirely in test
    # The data is divided into 5 folds. In each iteration:
    # 1 fold is used for validation.
    # The remaining 4 folds are used for training.
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    ## np.zeros(len(labels)) is place Holder For the Needed Features (X)
    ## it returns array of splits
    # eachh split has two arrays --> [train idx] , [test idx]
    splits = list(cv.split(np.zeros(len(labels)), labels, block_ids))

    # Step 5: safety check, no block may appear on both sides
    for train_idx, test_idx in splits:
        if set(block_ids[train_idx]) & set(block_ids[test_idx]):
            raise AssertionError("Block leakage detected")

    return block_ids, splits


