"""Combined project tests: LOSO evaluation and pre-hardware Step 9."""

# ---------------------------------------------------------------------------
# Pre-hardware Step 9 pipeline tests
# ---------------------------------------------------------------------------

"""Automated checks corresponding to pre-hardware guide UT-01–UT-07."""

import sys
from pathlib import Path

import numpy as np
import mne

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders import bnci2014_008 as loader
from src.preprocessing import pipeline
from src.simulation.synthetic_p300 import generate_synthetic_p300


def _make_raw_with_events():
    """Small deterministic recording with the project's channels and markers."""
    sfreq = 256.0
    n_samples = int(8 * sfreq)
    rng = np.random.default_rng(17)
    eeg = rng.normal(0.0, 3e-6, size=(len(pipeline.EEG_CHANNELS), n_samples))
    target_stim = np.zeros(n_samples)
    flash_stim = np.zeros(n_samples)
    event_samples = [512, 768, 1024, 1280]
    event_codes = [1, 2, 1, 2]
    flash_codes = [1, 2, 3, 4]
    for sample, event_code, flash_code in zip(
        event_samples, event_codes, flash_codes
    ):
        target_stim[sample] = event_code
        flash_stim[sample] = flash_code

    info = mne.create_info(
        pipeline.EEG_CHANNELS + ["Target stim", "Flash stim"],
        sfreq,
        ch_types=["eeg"] * len(pipeline.EEG_CHANNELS) + ["stim", "stim"],
    )
    data = np.vstack([eeg, target_stim, flash_stim])
    return mne.io.RawArray(data, info, verbose=False)


def test_ut01_loader_returns_requested_subject_metadata_and_events(monkeypatch):
    raw = _make_raw_with_events()

    class FakeDataset:
        subject_list = [1]

        def get_data(self, subjects):
            assert subjects == [1]
            return {1: {"session": {"run": raw}}}

    monkeypatch.setattr(loader, "BNCI2014_008", FakeDataset)
    monkeypatch.setattr(loader.mne, "set_config", lambda *args, **kwargs: None)

    dataset, subject_data = loader.load_bnci2014_008(subjects=[1])
    loaded = subject_data[1]["session"]["run"]
    events = mne.find_events(loaded, stim_channel="Target stim", verbose=False)

    assert dataset.subject_list == [1]
    assert loaded.info["sfreq"] == 256.0
    assert len(loaded.ch_names) == 10
    assert events[:, 2].tolist() == [1, 2, 1, 2]


def test_ut02_filtering_keeps_expected_trial_count():
    epochs = pipeline.preprocess_eeg(_make_raw_with_events())

    assert len(epochs) == 4
    assert epochs.events[:, 2].tolist() == [1, 2, 1, 2]


def test_ut03_epoch_window_has_expected_sample_count():
    epochs = pipeline.preprocess_eeg(_make_raw_with_events())
    one_sample_seconds = 1.0 / epochs.info["sfreq"]

    # MNE snaps requested boundaries to the 256 Hz sample grid; ±200/800 ms
    # are not exact sample indices at this rate.
    assert abs(epochs.tmin - pipeline.TMIN) <= one_sample_seconds
    assert abs(epochs.tmax - pipeline.TMAX) <= one_sample_seconds
    assert len(epochs.times) == 257


def test_ut04_baseline_is_approximately_zero_after_correction():
    epochs = pipeline.preprocess_eeg(_make_raw_with_events())

    residual = pipeline._verify_baseline(epochs)
    assert residual <= 1e-10


def test_ut05_synthetic_target_has_positive_p300_window_difference():
    data, labels, times = generate_synthetic_p300(seed=42)
    pz_index = ("Fz", "Cz", "Pz", "P3", "P4").index("Pz")
    window = (times >= 0.300) & (times <= 0.500)
    target = data[labels == 1, pz_index][:, window].mean()
    nontarget = data[labels == 0, pz_index][:, window].mean()

    assert target > nontarget
    assert target - nontarget > 2.0


def test_ut06_synthetic_no_p300_control_has_no_target_window_effect():
    data, labels, times = generate_synthetic_p300(
        seed=42,
        include_p300=False,
    )
    pz_index = ("Fz", "Cz", "Pz", "P3", "P4").index("Pz")
    window = (times >= 0.300) & (times <= 0.500)
    target = data[labels == 1, pz_index][:, window].mean()
    nontarget = data[labels == 0, pz_index][:, window].mean()

    assert abs(target - nontarget) < 1.0


def test_ut07_fixed_seed_reproduces_grouped_splits_and_fold_metrics():
    import pandas as pd
    from src.evaluation.subject_dependent_5fold import (
        evaluate_fold,
        make_grouped_splits,
    )

    rng = np.random.default_rng(23)
    labels = np.tile([2, 1, 1, 1], 20)
    groups = np.repeat(np.arange(20), 4)
    features = rng.normal(size=(len(labels), 8))
    features[labels == 2, 0] += 1.5
    features = pd.DataFrame(features)

    first_splits = make_grouped_splits(labels, groups, seed=42)
    second_splits = make_grouped_splits(labels, groups, seed=42)
    assert len(first_splits) == 5

    first_metrics = []
    second_metrics = []
    for (train_a, test_a), (train_b, test_b) in zip(first_splits, second_splits):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)
        assert not set(groups[train_a]).intersection(groups[test_a])
        first_metrics.append(evaluate_fold(features, labels, train_a, test_a)[0])
        second_metrics.append(evaluate_fold(features, labels, train_b, test_b)[0])
    assert first_metrics == second_metrics
