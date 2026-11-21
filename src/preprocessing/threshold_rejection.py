import logging

import numpy as np
import pandas as pd
import mne
from autoreject import AutoReject
from autoreject import get_rejection_threshold
logger = logging.getLogger(__name__)

def apply_global_rejection(
    epochs: mne.Epochs,
    cv: int = 5,
    random_state: int = 42,
    logger: logging.Logger | None = None
):
    logger = logger or logging.getLogger(__name__)
    logger.info("Starting global artifact rejection.")
    logger.info("Original epochs: %d", len(epochs))

    eeg_picks = mne.pick_types(
        epochs.info, eeg=True, exclude=[]
    )

    original_selection = np.asarray(epochs.selection).copy()
    channel_names = [epochs.ch_names[i] for i in eeg_picks]

    reject = get_rejection_threshold(
        epochs,
        ch_types=["eeg"],
        cv=cv,
        random_state=random_state
    )

    threshold = reject["eeg"]

    logger.info(
        "Estimated global EEG threshold: %.2f µV",
        threshold * 1e6
    )

    data = epochs.get_data(picks=eeg_picks)
    p2p = np.ptp(data, axis=2)

    epochs_clean = epochs.copy()
    epochs_clean.drop_bad(reject=reject, verbose=False)

    remaining = set(epochs_clean.selection)

    removed_count = 0

    for i, selection_id in enumerate(original_selection):

        removed = selection_id not in remaining

        if removed:
            removed_count += 1

            bad_channels = [
                channel_names[j]
                for j in range(len(channel_names))
                if p2p[i, j] > threshold
            ]

            logger.warning(
                "Epoch rejected | Index: %d | Bad channels: %s | "
                "Maximum P2P: %.2f µV",
                i,
                bad_channels,
                np.max(p2p[i]) * 1e6
            )

    logger.info(
        "Global rejection completed | Original: %d | "
        "Removed: %d | Retained: %d | Rejection rate: %.2f%%",
        len(epochs),
        removed_count,
        len(epochs_clean),
        removed_count / len(epochs) * 100
    )

    return epochs_clean, len(epochs_clean)



def apply_full_autoreject(
    epochs: mne.Epochs,
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1 ,
 logger: logging.Logger | None = None
):
    logger = logger or logging.getLogger(__name__)

    logger.info("Starting full AutoReject.")
    logger.info("Original epochs: %d", len(epochs))

    eeg_picks = mne.pick_types(
        epochs.info, eeg=True, exclude=[]
    )

    original_selection = np.asarray(epochs.selection).copy()
    channel_names = [epochs.ch_names[i] for i in eeg_picks]

    ar = AutoReject(
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        picks=eeg_picks
    )

    epochs_clean, reject_log = ar.fit_transform(
        epochs.copy(),
        return_log=True
    )

    removed_count = 0
    interpolated_count = 0

    for i, selection_id in enumerate(original_selection):

        labels = reject_log.labels[i]
        removed = bool(reject_log.bad_epochs[i])

        bad_channels = [
            channel_names[j]
            for j in range(len(channel_names))
            if labels[j] == 1
        ]

        interpolated_channels = [
            channel_names[j]
            for j in range(len(channel_names))
            if labels[j] == 2
        ]

        if removed:
            removed_count += 1

            logger.warning(
                "Epoch rejected | Index: %d | Bad channels: %s",
                i,
                bad_channels
            )

        elif interpolated_channels:
            interpolated_count += 1

            logger.info(
                "Channels interpolated | Epoch: %d | Channels: %s",
                i,
                interpolated_channels
            )

    logger.info(
        "AutoReject completed | Original: %d | Removed: %d | "
        "Retained: %d | Interpolated epochs: %d | "
        "Rejection rate: %.2f%%",
        len(epochs),
        removed_count,
        len(epochs_clean),
        interpolated_count,
        removed_count / len(epochs) * 100
    )

    logger.info("Selected interpolation: %s", ar.n_interpolate_)
    logger.info("Selected consensus: %s", ar.consensus_)

    return epochs_clean,len(epochs_clean),
