
"""Generate reproducible stimulus event schedules."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from ...config import GeneratorConfig
from .events import EventLabel, StimulusEvent
from .participant import Participant


def generate_event_schedule(
    cfg: GeneratorConfig,
    participant: Participant,
    n_probe: int,
    n_irrelevance: int,
    n_target: int,
    first_event_onset_s: float = 1.0,
    inter_event_interval_s: float = 1.0,
    seed: int | None = None,
) -> list[StimulusEvent]:
    """Generate a reproducible schedule of stimulus events.

    Event labels are shuffled, while event timings remain evenly spaced.
    Participant condition is not copied into the event records.
    """

    # Validate the recording configuration.
    cfg.validate()

    if not isinstance(participant, Participant):
        raise ValueError("participant must be a Participant instance")

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

    # Validate event timing parameters.
    if (
        isinstance(first_event_onset_s, bool)
        or not isinstance(first_event_onset_s, Real)
        or not math.isfinite(float(first_event_onset_s))
        or first_event_onset_s < 0
    ):
        raise ValueError("first_event_onset_s must be finite and >= 0")

    if (
        isinstance(inter_event_interval_s, bool)
        or not isinstance(inter_event_interval_s, Real)
        or not math.isfinite(float(inter_event_interval_s))
        or inter_event_interval_s <= 0
    ):
        raise ValueError("inter_event_interval_s must be finite and > 0")

    # Use the configuration seed by default.
    actual_seed = cfg.seed if seed is None else seed

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

    events: list[StimulusEvent] = []
    used_sample_indices: set[int] = set()

    for index, label in enumerate(labels):
        # Calculate the event onset time.
        onset_time_s = (
            float(first_event_onset_s)
            + index * float(inter_event_interval_s)
        )

        if onset_time_s >= cfg.duration_realized_s:
            raise ValueError(
                "Event onset exceeds the recording duration"
            )

        # Convert event time to the nearest EEG sample.
        sample_index = int(round(onset_time_s * cfg.sampling_rate))

        if sample_index >= cfg.n_samples:
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

        event = StimulusEvent(
            participant_id=participant.participant_id,
            trial_id=trial_id,
            label=label,
            onset_time_s=onset_time_s,
            sample_index=sample_index,
        )

        events.append(event)

    return events
