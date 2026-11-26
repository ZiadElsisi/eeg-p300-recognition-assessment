
"""Tests for synthetic participant conditions and event schedules."""

from collections import Counter

import pytest

from imprint.config import GeneratorConfig
from imprint.data.synthetic.events import EventLabel, StimulusEvent
from imprint.data.synthetic.participant import (
    Participant,
    ParticipantCondition,
)
from imprint.data.synthetic.schedule import generate_event_schedule


@pytest.fixture
def config():
    """Create a short EEG recording configuration for tests."""

    return GeneratorConfig(
        n_channels=8,
        sampling_rate=250.0,
        duration_s=10.0,
        seed=42,
    )


@pytest.fixture
def participant():
    """Create a test participant."""

    return Participant(
        participant_id="P001",
        condition=ParticipantCondition.KNOWS,
    )


def test_participant_conditions():
    """Check the available participant conditions."""

    assert ParticipantCondition.KNOWS.value == "KNOWS"
    assert ParticipantCondition.DOES_NOT_KNOW.value == "DOES_NOT_KNOW"


def test_participant_rejects_empty_id():
    """Reject participants without an identifier."""

    with pytest.raises(ValueError):
        Participant(
            participant_id="",
            condition=ParticipantCondition.KNOWS,
        )


def test_event_rejects_invalid_label():
    """Reject event labels outside the defined categories."""

    with pytest.raises(ValueError):
        StimulusEvent(
            participant_id="P001",
            trial_id="P001-trial-0001",
            label="UNKNOWN",
            onset_time_s=1.0,
            sample_index=250,
        )


def test_schedule_event_counts(config, participant):
    """Check that the requested number of each event is generated."""

    events = generate_event_schedule(
        cfg=config,
        participant=participant,
        n_probe=3,
        n_irrelevance=4,
        n_target=2,
        seed=123,
    )

    counts = Counter(event.label for event in events)

    assert len(events) == 9
    assert counts[EventLabel.PROBE] == 3
    assert counts[EventLabel.IRRELEVANCE] == 4
    assert counts[EventLabel.TARGET] == 2


def test_schedule_sample_indices(config, participant):
    """Check conversion from event time to EEG sample index."""

    events = generate_event_schedule(
        cfg=config,
        participant=participant,
        n_probe=2,
        n_irrelevance=1,
        n_target=1,
        first_event_onset_s=1.0,
        inter_event_interval_s=1.0,
        seed=42,
    )

    for event in events:
        expected_index = round(
            event.onset_time_s * config.sampling_rate
        )

        assert event.sample_index == expected_index


def test_schedule_is_reproducible(config, participant):
    """Check that the same seed generates the same schedule."""

    first_schedule = generate_event_schedule(
        cfg=config,
        participant=participant,
        n_probe=3,
        n_irrelevance=3,
        n_target=2,
        seed=100,
    )

    second_schedule = generate_event_schedule(
        cfg=config,
        participant=participant,
        n_probe=3,
        n_irrelevance=3,
        n_target=2,
        seed=100,
    )

    assert first_schedule == second_schedule


def test_schedule_uses_config_seed_by_default(config, participant):
    """Check that the configuration seed is used when no seed is given."""

    first_schedule = generate_event_schedule(
        cfg=config,
        participant=participant,
        n_probe=2,
        n_irrelevance=2,
        n_target=1,
    )

    second_schedule = generate_event_schedule(
        cfg=config,
        participant=participant,
        n_probe=2,
        n_irrelevance=2,
        n_target=1,
        seed=config.seed,
    )

    assert first_schedule == second_schedule


def test_schedule_keeps_condition_separate(config, participant):
    """Check that event records do not contain participant condition."""

    events = generate_event_schedule(
        cfg=config,
        participant=participant,
        n_probe=1,
        n_irrelevance=1,
        n_target=1,
    )

    for event in events:
        assert event.participant_id == participant.participant_id
        assert not hasattr(event, "condition")


def test_schedule_rejects_zero_events(config, participant):
    """Reject an empty event schedule."""

    with pytest.raises(ValueError):
        generate_event_schedule(
            cfg=config,
            participant=participant,
            n_probe=0,
            n_irrelevance=0,
            n_target=0,
        )


def test_schedule_rejects_negative_event_count(config, participant):
    """Reject negative event counts."""

    with pytest.raises(ValueError):
        generate_event_schedule(
            cfg=config,
            participant=participant,
            n_probe=-1,
            n_irrelevance=2,
            n_target=1,
        )


def test_schedule_rejects_events_outside_recording(config, participant):
    """Reject schedules that extend beyond the recording duration."""

    with pytest.raises(ValueError):
        generate_event_schedule(
            cfg=config,
            participant=participant,
            n_probe=5,
            n_irrelevance=5,
            n_target=5,
            first_event_onset_s=1.0,
            inter_event_interval_s=1.0,
        )


def test_schedule_rejects_events_at_same_sample(config, participant):
    """Reject event timings that map to the same EEG sample."""

    with pytest.raises(ValueError):
        generate_event_schedule(
            cfg=config,
            participant=participant,
            n_probe=2,
            n_irrelevance=0,
            n_target=0,
            first_event_onset_s=1.0,
            inter_event_interval_s=0.001,
        )


def test_schedule_rejects_negative_seed(config, participant):
    """Reject negative random seeds."""

    with pytest.raises(ValueError):
        generate_event_schedule(
            cfg=config,
            participant=participant,
            n_probe=1,
            n_irrelevance=1,
            n_target=1,
            seed=-5,
        )
