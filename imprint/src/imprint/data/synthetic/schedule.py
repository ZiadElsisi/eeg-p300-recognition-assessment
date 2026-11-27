
"""Generate reproducible stimulus event schedules."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from ...config import load_config
from ...config.settings import ImprintConfig, SyntheticSettings
from .events import EventLabel, StimulusEvent
from .participant import Participant


def generate_event_schedule(
    cfg: SyntheticSettings | ImprintConfig,
    participant: Participant,
    n_probe: int | None = None,
    n_irrelevance: int | None = None,
    n_target: int | None = None,
    first_event_onset_s: float | None = None,
    inter_event_interval_s: float | None = None,
    trials_per_block: int | None = None,
    seed: int | None = None,
) -> list[StimulusEvent]:
    """Generate a reproducible schedule of stimulus events.

    Event labels are shuffled, while event timings remain evenly spaced.
    Participant condition is not copied into the event records.
    """

    experiment = cfg.session if isinstance(cfg, ImprintConfig) else None
    session_defaults = experiment or load_config().session
    synthetic_cfg = cfg.synthetic if isinstance(cfg, ImprintConfig) else cfg
    if not isinstance(synthetic_cfg, SyntheticSettings):
        raise ValueError("cfg must be SyntheticSettings or ImprintConfig")

    # Validate the recording configuration.
    synthetic_cfg.validate()

    if not isinstance(participant, Participant):
        raise ValueError("participant must be a Participant instance")

    counts_provided = (n_probe is not None, n_irrelevance is not None, n_target is not None)
    if experiment is not None and not any(counts_provided):
        n_probe = experiment.probe_pictures_per_set * experiment.picture_set_count * experiment.repetitions_per_picture
        n_irrelevance = experiment.irrelevance_pictures_per_set * experiment.picture_set_count * experiment.repetitions_per_picture
        n_target = experiment.target_pictures_per_set * experiment.picture_set_count * experiment.repetitions_per_picture
    elif not all(counts_provided):
        raise ValueError("provide all three event counts, or pass ImprintConfig and omit all three")

    assert n_probe is not None and n_irrelevance is not None and n_target is not None

    # Validate event counts.
    counts = {
        "n_probe": n_probe,
        "n_irrelevance": n_irrelevance,
        "n_target": n_target,
    }

    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be a non-negative integer")

        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    total_events = n_probe + n_irrelevance + n_target

    if total_events == 0:
        raise ValueError("At least one event must be requested")

    if trials_per_block is not None and (
        isinstance(trials_per_block, bool)
        or not isinstance(trials_per_block, Integral)
        or trials_per_block <= 0
    ):
        raise ValueError("trials_per_block must be a positive integer or None")

    # Validate event timing parameters.
    if first_event_onset_s is None:
        first_event_onset_s = session_defaults.initial_delay_seconds
    if (
        isinstance(first_event_onset_s, bool)
        or not isinstance(first_event_onset_s, Real)
        or not math.isfinite(float(first_event_onset_s))
        or first_event_onset_s < 0
    ):
        raise ValueError("first_event_onset_s must be finite and >= 0")

    if inter_event_interval_s is not None and (
        isinstance(inter_event_interval_s, bool)
        or not isinstance(inter_event_interval_s, Real)
        or not math.isfinite(float(inter_event_interval_s))
        or inter_event_interval_s <= 0
    ):
        raise ValueError("inter_event_interval_s must be finite and > 0")

    # Use the configuration seed by default.
    actual_seed = synthetic_cfg.seed if seed is None else seed

    if isinstance(actual_seed, bool) or not isinstance(actual_seed, Integral):
        raise ValueError("seed must be an integer")

    if actual_seed < 0:
        raise ValueError("seed must be >= 0")

    # Build the requested event labels.
    labels = (
        [EventLabel.PROBE] * int(n_probe)
        + [EventLabel.IRRELEVANCE] * int(n_irrelevance)
        + [EventLabel.TARGET] * int(n_target)
    )

    # Shuffle event order reproducibly.
    rng = np.random.default_rng(int(actual_seed))
    rng.shuffle(labels)

    if trials_per_block is None and experiment is not None:
        trials_per_block = experiment.trials_per_block

    if experiment is not None and inter_event_interval_s is None:
        intervals = rng.uniform(
            experiment.inter_picture_interval_min_seconds,
            experiment.inter_picture_interval_max_seconds,
            size=max(0, total_events - 1),
        )
    else:
        fixed_interval = (
            session_defaults.legacy_fixed_inter_event_interval_seconds
            if inter_event_interval_s is None
            else float(inter_event_interval_s)
        )
        intervals = np.full(max(0, total_events - 1), fixed_interval, dtype=np.float64)

    session_end_limit = (
        experiment.maximum_schedule_duration_seconds
        if experiment is not None
        else synthetic_cfg.duration_realized_s
    )

    events: list[StimulusEvent] = []
    used_sample_indices: set[int] = set()

    for index, label in enumerate(labels):
        # Calculate the event onset time.
        onset_time_s = float(first_event_onset_s) + float(intervals[:index].sum())

        if onset_time_s >= session_end_limit:
            raise ValueError(
                "Event onset exceeds the recording duration"
            )

        # Convert event time to the nearest EEG sample.
        sample_index = int(round(onset_time_s * synthetic_cfg.sampling_rate))

        if experiment is None and sample_index >= synthetic_cfg.n_samples:
            raise ValueError(
                "Event sample index is outside the recording"
            )

        if sample_index in used_sample_indices:
            raise ValueError(
                "Two events cannot occupy the same sample index"
            )

        used_sample_indices.add(sample_index)

        # Create a unique trial identifier.
        trial_id = f"{participant.participant_id}-trial-{index + 1:04d}"
        block_number = (
            1 if trials_per_block is None else index // int(trials_per_block) + 1
        )
        block_id = f"{participant.participant_id}-block-{block_number:04d}"

        event = StimulusEvent(
            participant_id=participant.participant_id,
            trial_id=trial_id,
            block_id=block_id,
            label=label,
            onset_time_s=onset_time_s,
            sample_index=sample_index,
        )

        events.append(event)

    return events
