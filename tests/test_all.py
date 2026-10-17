"""Combined project tests: LOSO evaluation and pre-hardware Step 9."""

# ---------------------------------------------------------------------------
# LOSO evaluation tests
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.evaluation import loso  # noqa: E402

SUBJECTS = [1, 2, 3, 4, 5, 6]
N_PER_SUBJECT = {1: 120, 2: 110, 3: 130, 4: 100, 5: 125, 6: 115}
N_FEATURES = 6          # features 0..4 are signal/noise, last column = sentinel
SENTINEL_COL = N_FEATURES - 1


def sentinel_value(subject: int) -> float:
    return 1000.0 + subject


@pytest.fixture(scope="module")
def synthetic():
    """Imbalanced (~1:4) Target/NonTarget data from 6 subjects."""
    rng = np.random.default_rng(0)
    X_parts, y_parts, id_parts = [], [], []
    for s in SUBJECTS:
        n = N_PER_SUBJECT[s]
        y = np.where(rng.random(n) < 0.2, loso.TARGET_LABEL,
                     loso.NONTARGET_LABEL)
        X = rng.normal(size=(n, N_FEATURES))
        X[y == loso.TARGET_LABEL, 0] += 2.0          # learnable P300-like shift
        X[:, SENTINEL_COL] = sentinel_value(s)       # marks the subject
        X_parts.append(X)
        y_parts.append(y)
        id_parts.append(np.full(n, s))
    return (np.vstack(X_parts), np.concatenate(y_parts),
            np.concatenate(id_parts))


class SpyModel:
    """Real pipeline that records which subjects reach fit()/predict()."""
    fit_sentinels: list = []
    predict_sentinels: list = []
    fit_sizes: list = []

    def __init__(self):
        self.model = loso.build_model()

    @property
    def classes_(self):
        return self.model.classes_

    def fit(self, X, y):
        type(self).fit_sentinels.append(set(np.unique(X[:, SENTINEL_COL])))
        type(self).fit_sizes.append(len(X))
        self.model.fit(X, y)
        return self

    def predict(self, X):
        type(self).predict_sentinels.append(set(np.unique(X[:, SENTINEL_COL])))
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def reset_spy():
    SpyModel.fit_sentinels = []
    SpyModel.predict_sentinels = []
    SpyModel.fit_sizes = []


# 1. fold count -------------------------------------------------------
def test_number_of_folds_equals_number_of_subjects(synthetic):
    _, _, ids = synthetic
    assert len(loso.make_loso_folds(ids)) == len(SUBJECTS)


# 2. each subject held out exactly once -------------------------------
def test_every_subject_held_out_exactly_once(synthetic):
    _, _, ids = synthetic
    held_out = [f["held_out_subject"] for f in loso.make_loso_folds(ids)]
    assert sorted(held_out) == SUBJECTS
    assert len(set(held_out)) == len(held_out)


# 3 + 4. no overlap, correct train/test composition -------------------
def test_train_test_composition(synthetic):
    _, _, ids = synthetic
    for fold in loso.make_loso_folds(ids):
        held = fold["held_out_subject"]
        train_idx, test_idx = fold["train_idx"], fold["test_idx"]

        assert held not in fold["train_subjects"]
        assert held not in set(ids[train_idx])
        assert np.intersect1d(train_idx, test_idx).size == 0
        assert set(ids[test_idx]) == {held}
        assert set(ids[train_idx]) == set(SUBJECTS) - {held}
        assert fold["train_subjects"] == [s for s in SUBJECTS if s != held]
        # every row is used exactly once per fold
        assert len(train_idx) + len(test_idx) == len(ids)


# 5. metrics exist for every held-out subject -------------------------
def test_metrics_for_every_held_out_subject(synthetic):
    X, y, ids = synthetic
    df = loso.run_loso(X, y, ids)
    assert list(df["held_out_subject"]) == SUBJECTS
    for col in ["fold", "held_out_subject", "train_subjects", "n_train",
                "n_test", *loso.METRICS]:
        assert col in df.columns
    assert not df[loso.METRICS].isna().any().any()
    for s, n_test in N_PER_SUBJECT.items():
        row = df[df["held_out_subject"] == s].iloc[0]
        assert row["n_test"] == n_test
        assert row["n_train"] == sum(N_PER_SUBJECT.values()) - n_test
        assert row["tn"] + row["fp"] + row["fn"] + row["tp"] == n_test
    # signal was injected, so LOSO should beat chance
    assert df["roc_auc"].mean() > 0.8


# 6. mean / std correctness -------------------------------------------
def test_mean_and_std_across_subjects(synthetic):
    X, y, ids = synthetic
    df = loso.run_loso(X, y, ids)
    summary = loso.summarize(df).set_index("metric")
    for m in loso.METRICS:
        values = df[m].to_numpy()
        assert summary.loc[m, "mean"] == pytest.approx(values.mean())
        assert summary.loc[m, "std"] == pytest.approx(values.std(ddof=1))
    # statistics are over the 6 subjects, not over trials
    assert len(df) == len(SUBJECTS)


# 7 + 9 (UT-08). held-out subject never reaches training --------------
def test_held_out_subject_never_seen_in_training(synthetic):
    X, y, ids = synthetic
    reset_spy()
    df = loso.run_loso(X, y, ids, model_factory=SpyModel)

    assert len(SpyModel.fit_sentinels) == len(SUBJECTS)
    for fold_no, held in enumerate(SUBJECTS):
        fit_seen = SpyModel.fit_sentinels[fold_no]
        pred_seen = SpyModel.predict_sentinels[fold_no]

        # test_subject_id NEVER appears in training data
        assert sentinel_value(held) not in fit_seen
        assert fit_seen == {sentinel_value(s) for s in SUBJECTS if s != held}
        # predict() only ever sees the held-out subject
        assert pred_seen == {sentinel_value(held)}
        assert SpyModel.fit_sizes[fold_no] == (
            sum(N_PER_SUBJECT.values()) - N_PER_SUBJECT[held]
        )

    # recorded training lists never contain the held-out subject
    for _, row in df.iterrows():
        train = [int(s) for s in row["train_subjects"].split(",")]
        assert row["held_out_subject"] not in train
        assert len(train) == len(SUBJECTS) - 1


# integrity guard rejects a leaking fold ------------------------------
def test_leaking_fold_is_rejected(synthetic):
    X, y, ids = synthetic
    fold = loso.make_loso_folds(ids)[0]

    leaking = dict(fold)
    leaking["train_idx"] = np.concatenate([fold["train_idx"],
                                           fold["test_idx"][:5]])
    with pytest.raises(loso.LeakageError):
        loso.evaluate_fold(leaking, X, y, ids)

    wrong_list = dict(fold)
    wrong_list["train_subjects"] = SUBJECTS[:]   # includes held-out subject
    with pytest.raises(loso.LeakageError):
        loso.evaluate_fold(wrong_list, X, y, ids)

    mixed_test = dict(fold)
    mixed_test["test_idx"] = np.concatenate([fold["test_idx"],
                                             fold["train_idx"][:5]])
    with pytest.raises(loso.LeakageError):
        loso.evaluate_fold(mixed_test, X, y, ids)


# ROC-AUC must use the Target probability column ----------------------
def test_roc_auc_uses_target_probability_column(synthetic):
    class ReversedClassOrderModel:
        """classes_ = [2, 1] so the Target column is 0, not 1."""
        classes_ = np.array([loso.TARGET_LABEL, loso.NONTARGET_LABEL])

        def fit(self, X, y):
            return self

        def _p_target(self, X):
            return 1.0 / (1.0 + np.exp(-X[:, 0]))

        def predict_proba(self, X):
            p = self._p_target(X)
            return np.column_stack([p, 1.0 - p])      # [P(2), P(1)]

        def predict(self, X):
            return np.where(self._p_target(X) > 0.5,
                            loso.TARGET_LABEL, loso.NONTARGET_LABEL)

    rng = np.random.default_rng(1)
    ids = np.repeat([1, 2, 3], 40)
    y = np.tile([1, 1, 1, 2], 30)
    X = np.zeros((len(y), 2))
    X[:, 0] = np.where(y == 2, 3.0, -3.0) + rng.normal(0, 0.1, len(y))

    df = loso.run_loso(X, y, ids, model_factory=ReversedClassOrderModel)
    assert (df["roc_auc"] == 1.0).all()


# determinism (no randomness in LOSO) ---------------------------------
def test_results_are_reproducible(synthetic):
    X, y, ids = synthetic
    a = loso.run_loso(X, y, ids)
    b = loso.run_loso(X, y, ids)
    pd.testing.assert_frame_equal(a, b)


# subject discovery from file names -----------------------------------
def test_discover_subjects_from_filenames(tmp_path):
    for s in [3, 1, 2, 10]:
        (tmp_path / f"subject_{s}_clean-epo.fif").touch()
    (tmp_path / "subject_x_clean-epo.fif").touch()
    (tmp_path / "notes.txt").touch()
    assert loso.discover_subjects(tmp_path) == [1, 2, 3, 10]


# saved files ----------------------------------------------------------
def test_results_saved_with_expected_columns(synthetic, tmp_path):
    X, y, ids = synthetic
    df = loso.run_loso(X, y, ids)
    summary = loso.summarize(df)
    fold_path, summary_path = loso.save_results(df, summary, tmp_path)

    fold_csv = pd.read_csv(fold_path)
    assert list(fold_csv.columns[:10]) == [
        "fold", "held_out_subject", "train_subjects", "n_train", "n_test",
        "accuracy", "precision", "recall", "f1", "roc_auc",
    ]
    assert len(fold_csv) == len(SUBJECTS)
    first = fold_csv.iloc[0]
    assert str(first["train_subjects"]) == "2,3,4,5,6"

    summary_csv = pd.read_csv(summary_path)
    assert list(summary_csv.columns) == ["metric", "mean", "std"]
    assert list(summary_csv["metric"]) == loso.METRICS

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
