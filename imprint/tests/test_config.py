import json

import pytest

from imprint.config import default_config_path, load_config
from imprint.data.synthetic import generate_recording


def test_default_toml_loads_all_sections_and_approved_values():
    config = load_config()

    assert default_config_path().name == "default.toml"
    assert config.signal.brain_channel_count == 6
    assert config.signal.eye_channel_count == 2
    assert config.signal.total_channel_count == 8
    assert config.signal.eye_channels is None
    assert config.signal.signal_unit == "V"
    assert config.filtering.low_cutoff_hz == pytest.approx(0.1)
    assert config.filtering.high_cutoff_hz == pytest.approx(30.0)
    assert config.epoching.epoch_start_seconds == pytest.approx(-0.2)
    assert config.epoching.epoch_end_seconds == pytest.approx(1.5)
    assert config.epoching.baseline_start_seconds == pytest.approx(-0.2)
    assert config.epoching.baseline_end_seconds == pytest.approx(0.0)
    assert config.artifact_rejection.peak_to_peak_threshold_uv == pytest.approx(150.0)
    assert config.p300.channel == "Pz"
    assert config.p300.measurement_start_ms == pytest.approx(300.0)
    assert config.p300.measurement_end_ms == pytest.approx(600.0)
    assert config.bootstrap.comparison_groups == ("Probe", "Irrelevance")
    assert config.bootstrap.iterations == 1000
    assert config.bootstrap.minimum_probe_epochs == 10
    assert config.bootstrap.recognition_threshold == pytest.approx(0.90)


def test_session_totals_are_derived_and_consistent():
    config = load_config()

    assert config.session.picture_count == 12
    assert config.session.total_trials == 480
    assert config.session.trials_per_block == 120
    assert config.session.block_count == 4


def test_undecided_values_remain_unset():
    config = load_config()

    assert config.signal.eye_channels is None
    assert config.signal.eye_device_inputs is None
    assert config.filtering.filter_order is None
    assert config.bootstrap.resampling_method is None
    assert config.blink_correction.fisher_significance_level is None


def test_nested_overrides_change_effective_settings_and_generator_behavior():
    config = load_config(overrides={"synthetic": {"duration_s": 2.0}})

    assert config.synthetic.duration_s == 2.0
    assert generate_recording(config).shape == (8, 500)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"signal": {"sampling_rate": 0}}, "sampling_rate must be positive"),
        ({"signal": {"brain_channels": ["Fz", "Fz", "Pz", "P3", "P4", "Oz"]}}, "duplicate names"),
        ({"signal": {"eye_channels": ["Fz", "VEOG"]}}, "must not overlap"),
        ({"signal": {"total_channel_count": 9}}, "must equal brain plus eye"),
        ({"epoching": {"epoch_start_seconds": 2.0}}, "must be earlier"),
        ({"epoching": {"baseline_start_seconds": -0.3}}, "must lie within"),
        ({"filtering": {"high_cutoff_hz": 125.0}}, "below the Nyquist"),
        ({"bootstrap": {"iterations": 0}}, "bootstrap.iterations must be a positive integer"),
        ({"bootstrap": {"recognition_threshold": 1.1}}, "between 0 and 1"),
        ({"session": {"inter_picture_interval_min_seconds": 4.0}}, "no greater than its maximum"),
        ({"session": {"block_count": 7}}, "must divide the derived total"),
    ],
)
def test_invalid_overrides_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        load_config(overrides=overrides)


def test_unknown_configuration_settings_are_rejected():
    with pytest.raises(ValueError, match="Unknown setting.*signal"):
        load_config(overrides={"signal": {"mystery": 1}})


def test_effective_config_can_be_serialized(tmp_path):
    config = load_config(overrides={"synthetic": {"duration_s": 3.0}})
    path = config.save_effective_config(tmp_path / "runs" / "effective.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["synthetic"]["duration_s"] == 3.0
    assert data["session"]["total_trials"] == 480
    assert data["session"]["trials_per_block"] == 120
    assert data["overrides"] == {"synthetic": {"duration_s": 3.0}}


def test_override_capture_can_be_disabled():
    config = load_config(overrides={"synthetic": {"duration_s": 2.0}, "reproducibility": {"capture_overrides": False}})

    assert config.synthetic.duration_s == 2.0
    assert config.to_dict()["overrides"] == {}
