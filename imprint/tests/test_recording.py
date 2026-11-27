"""Tests for the shared analysis-facing Recording contract."""

import numpy as np
import pytest

from imprint.core import ChannelInfo, ChannelRole, EventLabel, Recording, StimulusEvent


def channels(*names):
    return tuple(ChannelInfo(name, ChannelRole.BRAIN, "V") for name in names)


def make_recording(data=None, **kwargs):
    if data is None:
        data = np.zeros((2, 100), dtype=np.float64)
    return Recording(data=data, sampling_rate_hz=250, channels=channels("Fz", "Pz"), **kwargs)


def test_construction_dimensions_duration_unit_and_channel_names():
    recording = make_recording()

    assert recording.data.shape == (2, 100)
    assert recording.n_channels == 2
    assert recording.n_samples == 100
    assert recording.sampling_rate_hz == 250
    assert recording.duration_s == pytest.approx(0.4)
    assert recording.channel_names == ("Fz", "Pz")
    assert recording.signal_unit == "V"


def test_data_is_zero_copy_and_read_only():
    values = np.zeros((2, 100))
    recording = make_recording(values)

    assert np.shares_memory(recording.data, values)
    assert not recording.data.flags.writeable
    with pytest.raises(ValueError):
        recording.data[0, 0] = 1.0


@pytest.mark.parametrize("data", [np.zeros(10), np.zeros((2, 3, 4))])
def test_invalid_dimensions_are_rejected(data):
    with pytest.raises(ValueError, match="two dimensions"):
        make_recording(data)


def test_non_array_and_non_numeric_data_are_rejected():
    with pytest.raises(TypeError, match="NumPy array"):
        make_recording([[0.0]])
    with pytest.raises(TypeError, match="real numeric"):
        make_recording(np.array([["x"]]))
    with pytest.raises(TypeError, match="real numeric"):
        make_recording(np.array([[1 + 2j]]))


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_invalid_sampling_rate_is_rejected(bad):
    with pytest.raises(ValueError, match="sampling_rate_hz"):
        Recording(np.zeros((2, 3)), bad, channels("Fz", "Pz"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_signal_values_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        make_recording(np.array([[value], [0.0]]))


def test_channel_count_and_duplicate_names_are_rejected():
    with pytest.raises(ValueError, match="channel dimension"):
        Recording(np.zeros((2, 3)), 250, channels("Fz"))
    with pytest.raises(ValueError, match="unique"):
        Recording(np.zeros((2, 3)), 250, channels("Fz", "Fz"))


def test_channel_metadata_is_validated():
    with pytest.raises(ValueError, match="non-empty"):
        ChannelInfo(" ", ChannelRole.BRAIN)
    with pytest.raises(ValueError, match="ChannelRole"):
        ChannelInfo("Fz", "brain")
    with pytest.raises(ValueError, match="canonical volts"):
        ChannelInfo("Fz", ChannelRole.BRAIN, "uV")
    with pytest.raises(TypeError, match="tuple of ChannelInfo"):
        Recording(np.zeros((1, 3)), 250, [ChannelInfo("Fz", ChannelRole.BRAIN)])


def test_event_samples_must_be_within_recording():
    valid = StimulusEvent("P01", "T01", "B01", EventLabel.PROBE, 0.0, 0)
    assert make_recording(events=(valid,)).events == (valid,)
    outside = StimulusEvent("P01", "T02", "B01", EventLabel.TARGET, 0.4, 100)
    with pytest.raises(ValueError, match="outside recording bounds"):
        make_recording(events=(outside,))


@pytest.mark.parametrize("key", ["condition", "participant_condition", "ground_truth", "countermeasure_label"])
def test_analysis_facing_recording_rejects_evaluation_only_metadata(key):
    with pytest.raises(ValueError, match="evaluation-only"):
        make_recording(metadata={"nested": {key: "hidden label"}})
