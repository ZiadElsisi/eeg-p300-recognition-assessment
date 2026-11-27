"""Tests for the shared source contract and synthetic adapter."""

import inspect

import numpy as np
import pytest

from imprint.config import load_config
from imprint.core import ChannelInfo, ChannelRole, DataSource, Recording
from imprint.data.synthetic import SyntheticDataSource


class DummySource(DataSource):
    def _load(self) -> Recording:
        return Recording(
            np.zeros((1, 4)),
            4,
            (ChannelInfo("Fz", ChannelRole.BRAIN),),
            metadata={"source_id": "dummy"},
        )


class InvalidDummySource(DataSource):
    def _load(self):
        return np.zeros((1, 4))


def test_concrete_source_implements_contract_and_returns_shared_recording():
    recording = DummySource().load()
    assert isinstance(recording, Recording)
    assert recording.n_channels == 1


def test_source_boundary_rejects_invalid_return_type():
    with pytest.raises(TypeError, match="must return Recording"):
        InvalidDummySource().load()


def test_source_contract_has_no_dataset_or_synthetic_dependencies():
    import imprint.core.interfaces as interfaces

    source = inspect.getsource(interfaces)
    assert "synthetic" not in source.lower()
    assert "erp" not in source.lower()
    assert "openbci" not in source.lower()


def test_synthetic_source_loads_configured_recording():
    config = load_config(overrides={"synthetic": {"duration_s": 2.0}})
    recording = SyntheticDataSource(config).load()

    assert isinstance(recording, Recording)
    assert recording.data.shape == (config.signal.total_channel_count, 500)
    assert recording.sampling_rate_hz == config.signal.sampling_rate
    assert recording.duration_s == pytest.approx(2.0)
    assert recording.signal_unit == "V"
    assert recording.channel_names[:6] == config.signal.brain_channels
    assert all(ch.role is ChannelRole.BRAIN for ch in recording.channels[:6])
    assert all(ch.role is ChannelRole.EYE for ch in recording.channels[6:])
    assert recording.events == ()
    assert "condition" not in recording.metadata


def test_synthetic_source_reproducibility_and_configured_shape():
    config = load_config(overrides={
        "signal": {"sampling_rate": 200.0},
        "synthetic": {"duration_s": 1.5},
    })
    left = SyntheticDataSource(config).load()
    right = SyntheticDataSource(config).load()

    assert left.data.shape == (8, 300)
    np.testing.assert_array_equal(left.data, right.data)


def test_synthetic_source_rejects_event_out_of_range():
    from imprint.core import EventLabel, StimulusEvent

    config = load_config(overrides={"synthetic": {"duration_s": 1.0}})
    event = StimulusEvent("P01", "T01", "B01", EventLabel.PROBE, 1.0, 250)
    with pytest.raises(ValueError, match="outside recording bounds"):
        SyntheticDataSource(config, events=(event,)).load()
