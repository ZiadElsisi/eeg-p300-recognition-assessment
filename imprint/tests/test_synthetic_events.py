
"""Tests for synthetic participant conditions and event schedules."""

from collections import Counter

import pytest

from imprint.config import GeneratorConfig, load_config
from imprint.data.synthetic.events import EventLabel, StimulusEvent
from imprint.data.synthetic.participant import (
    Participant,
    ParticipantCohort,
    ParticipantCondition,
    assign_participant_conditions,
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


def test_condition_assignment_is_seeded_and_respects_requested_counts():
    ids = [f"P{index:03d}" for index in range(8)]
    counts = {
        ParticipantCondition.KNOWS: 3,
        ParticipantCondition.DOES_NOT_KNOW: 5,
    }

    first = assign_participant_conditions(ids, counts, seed=417)
    second = assign_participant_conditions(ids, counts, seed=417)

    assert first == second
    assert [person.participant_id for person in first.participants] == ids
    assert Counter(person.condition for person in first.participants) == counts
    assert first.seed == 417


def test_condition_assignment_requires_exact_counts():
    with pytest.raises(ValueError, match="sum to the number"):
        assign_participant_conditions(
            ["P001", "P002"],
            {ParticipantCondition.KNOWS: 1},
            seed=1,
        )


def test_condition_assignment_rejects_duplicate_participants():
    with pytest.raises(ValueError, match="unique"):
        assign_participant_conditions(
            ["P001", "P001"],
            {ParticipantCondition.KNOWS: 2},
            seed=1,
        )


def test_participant_cohort_round_trips_as_separate_truth_file(tmp_path):
    cohort = assign_participant_conditions(
        ["P001", "P002", "P003", "P004"],
        {
            ParticipantCondition.KNOWS: 2,
            ParticipantCondition.DOES_NOT_KNOW: 2,
        },
        seed=29,
    )
    truth_path = tmp_path / "evaluation" / "participant_conditions.json"

    saved_path = cohort.save(truth_path)
    loaded_cohort = ParticipantCohort.load(saved_path)

    assert loaded_cohort == cohort
    assert loaded_cohort.seed == 29
    assert loaded_cohort.to_dict()["schema"] == "imprint.synthetic-participant-conditions.v1"


def test_participant_cohort_rejects_unknown_schema():
    with pytest.raises(ValueError, match="Expected cohort schema"):
        ParticipantCohort.from_dict({"schema": "unknown", "seed": 1, "participants": []})


def test_event_rejects_invalid_label():
    """Reject event labels outside the defined categories."""

    with pytest.raises(ValueError):
        StimulusEvent(
            participant_id="P001",
            trial_id="P001-trial-0001",
            block_id="P001-block-0001",
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


def test_schedule_assigns_a_single_default_block(config, participant):
    events = generate_event_schedule(
        cfg=config,
        participant=participant,
        n_probe=2,
        n_irrelevance=2,
        n_target=1,
    )

    assert {event.block_id for event in events} == {"P001-block-0001"}


def test_schedule_assigns_configured_blocks_by_trial_order(config, participant):
    events = generate_event_schedule(
        cfg=config,
        participant=participant,
        n_probe=3,
        n_irrelevance=3,
        n_target=2,
        trials_per_block=3,
    )

    assert [event.block_id for event in events] == [
        "P001-block-0001",
        "P001-block-0001",
        "P001-block-0001",
        "P001-block-0002",
        "P001-block-0002",
        "P001-block-0002",
        "P001-block-0003",
        "P001-block-0003",
    ]


@pytest.mark.parametrize("trials_per_block", [0, -1, 1.5, True])
def test_schedule_rejects_invalid_block_size(config, participant, trials_per_block):
    with pytest.raises(ValueError, match="trials_per_block"):
        generate_event_schedule(
            cfg=config,
            participant=participant,
            n_probe=1,
            n_irrelevance=1,
            n_target=1,
            trials_per_block=trials_per_block,
        )


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


def test_full_config_drives_approved_session_counts_timing_and_blocks(participant):
    config = load_config()

    events = generate_event_schedule(cfg=config, participant=participant)
    repeated = generate_event_schedule(cfg=config, participant=participant)

    assert events == repeated
    assert len(events) == config.session.total_trials == 480
    counts = Counter(event.label for event in events)
    assert counts == {
        EventLabel.PROBE: 80,
        EventLabel.IRRELEVANCE: 320,
        EventLabel.TARGET: 80,
    }
    assert Counter(event.block_id for event in events) == {
        f"P001-block-{block:04d}": 120 for block in range(1, 5)
    }
    intervals = [
        right.onset_time_s - left.onset_time_s
        for left, right in zip(events, events[1:])
    ]
    assert all(
        config.session.inter_picture_interval_min_seconds <= interval
        <= config.session.inter_picture_interval_max_seconds
        for interval in intervals
    )
    assert all(
        event.sample_index == round(event.onset_time_s * config.signal.sampling_rate)
        for event in events
    )
