"""P300 measurement (Task 17).

Measures the mean Pz amplitude in a configured post-stimulus window for every
retained epoch. It makes NO recognition decision and never pools categories.

Conventions
-----------
* Epoch array: ``(n_epochs, n_channels, n_times)`` (architecture 5.3).
* Time vector: seconds relative to stimulus onset (onset = 0).
* Window: half-open ``[start_s, end_s)`` -- start INCLUSIVE, end EXCLUSIVE.
* Unit: the epochs' canonical unit (volts). Output field is ``p300_amplitude_v``.
  No conversion is done here; use ``to_microvolts`` at a reporting boundary.

ASSUMPTIONS to reconcile with the real repo (see completion report):
* ``epochs`` exposes: data, times_s, channel_names, sampling_rate_hz, unit,
  events (one per epoch, with trial_id/stimulus_id/event_label/block_id/
  sample_index), participant_id, optional recording_id and channel_types.
* ``epochs`` holds retained epochs only (Task 16 output).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

_TOL = 1e-9
_EXPECTED_CH_TYPE = "eeg"
_VOLT_UNITS = {"v", "volt", "volts"}


class P300MeasurementError(ValueError):
    """Invalid input for P300 measurement (replace with the project's domain error base)."""


@dataclass(frozen=True)
class P300Config:
    """Mirror of the centralized analysis config fields (merge into config/settings.py)."""
    p300_channel: str
    p300_window_start_s: float
    p300_window_end_s: float

    @classmethod
    def from_settings(cls, p300: Any, signal: Any = None) -> "P300Config":
        """Build from centralized ``ImprintConfig.p300`` (ms -> s), optionally
        cross-checked against ``ImprintConfig.signal.primary_p300_channel``."""
        if signal is not None and signal.primary_p300_channel != p300.channel:
            raise P300MeasurementError(
                f"p300.channel {p300.channel!r} disagrees with "
                f"signal.primary_p300_channel {signal.primary_p300_channel!r}"
            )
        return cls(
            p300_channel=p300.channel,
            p300_window_start_s=p300.measurement_start_ms / 1000.0,
            p300_window_end_s=p300.measurement_end_ms / 1000.0,
        )

    def __post_init__(self) -> None:
        if not self.p300_channel:
            raise P300MeasurementError("p300_channel must be non-empty")
        if not (np.isfinite(self.p300_window_start_s) and np.isfinite(self.p300_window_end_s)):
            raise P300MeasurementError("P300 window bounds must be finite")
        if self.p300_window_start_s >= self.p300_window_end_s:
            raise P300MeasurementError(
                f"P300 window start ({self.p300_window_start_s}) must be < end ({self.p300_window_end_s})"
            )


@dataclass(frozen=True)
class P300Measurement:
    participant_id: str | None
    recording_id: str | None
    trial_id: str
    stimulus_id: str
    event_type: Any            # project EventType enum value (PROBE/IRRELEVANCE/TARGET)
    block_id: str | None
    event_sample_index: int | None
    channel: str
    window_start_s: float
    window_end_s: float        # exclusive
    n_samples: int
    p300_amplitude_v: float    # volts


@dataclass(frozen=True)
class P300Exclusion:
    participant_id: str | None
    recording_id: str | None
    trial_id: str | None
    stimulus_id: str | None
    event_type: Any
    reason: str


@dataclass(frozen=True)
class P300MeasurementSet:
    measurements: tuple[P300Measurement, ...]
    exclusions: tuple[P300Exclusion, ...]
    unit: str = "V"
    n_input_epochs: int = 0

    def __post_init__(self) -> None:
        if len(self.measurements) + len(self.exclusions) != self.n_input_epochs:
            raise P300MeasurementError("measurements + exclusions must equal input epoch count")


def to_microvolts(amplitude_v: float) -> float:
    """Explicit reporting-boundary conversion (V -> uV)."""
    return amplitude_v * 1e6


class _EpochsLike(Protocol):  # documentation of the assumed contract
    data: np.ndarray
    times_s: np.ndarray
    channel_names: Sequence[str]
    sampling_rate_hz: float
    events: Sequence[Any]


def _select_channel(epochs: Any, name: str) -> int:
    names = list(epochs.channel_names)
    n = names.count(name)
    if n == 0:
        raise P300MeasurementError(f"Required channel {name!r} not found in {names}")
    if n > 1:
        raise P300MeasurementError(f"Channel {name!r} appears {n} times; must be unique")
    idx = names.index(name)
    types = getattr(epochs, "channel_types", None)
    if types is not None:
        if len(types) != len(names):
            raise P300MeasurementError("channel_types length does not match channel_names")
        if str(types[idx]).lower() != _EXPECTED_CH_TYPE:
            raise P300MeasurementError(
                f"Channel {name!r} has type {types[idx]!r}, expected {_EXPECTED_CH_TYPE!r}"
            )
    return idx


def _window_mask(times: np.ndarray, fs: float, start: float, end: float) -> np.ndarray:
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
        raise P300MeasurementError("Time vector must be 1-D, finite, with >= 2 samples")
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise P300MeasurementError("Time vector must be strictly increasing")
    if not (np.isfinite(fs) and fs > 0):
        raise P300MeasurementError("sampling_rate_hz must be positive and finite")
    if not np.allclose(dt, 1.0 / fs, rtol=1e-6, atol=1e-12):
        raise P300MeasurementError("Time vector spacing is inconsistent with sampling_rate_hz")
    if start < times[0] - _TOL or end > times[-1] + 1.0 / fs + _TOL:
        raise P300MeasurementError(
            f"Window [{start}, {end}) outside epoch range [{times[0]}, {times[-1]}]"
        )
    mask = (times >= start - _TOL) & (times < end - _TOL)
    if not mask.any():
        raise P300MeasurementError(f"No samples fall inside window [{start}, {end})")
    return mask


def _meta(ev: Any, name: str, default: Any = None) -> Any:
    return getattr(ev, name, default)


def measure_p300(epochs: Any, config: P300Config) -> P300MeasurementSet:
    """One Pz mean-amplitude measurement per input epoch (or an explicit exclusion record)."""
    data = np.asarray(epochs.data)
    if data.ndim != 3:
        raise P300MeasurementError(f"Epoch data must be (n_epochs, n_channels, n_times); got {data.shape}")
    n_ep, n_ch, n_t = data.shape
    if n_ch != len(epochs.channel_names):
        raise P300MeasurementError("Channel dimension does not match channel_names")
    times = np.asarray(epochs.times_s, dtype=float)
    if times.shape != (n_t,):
        raise P300MeasurementError("Time vector length does not match epoch n_times")
    if len(epochs.events) != n_ep:
        raise P300MeasurementError("Event metadata count does not match number of epochs")

    unit = getattr(epochs, "unit", None)
    if unit is not None and str(unit).strip().lower() not in _VOLT_UNITS:
        raise P300MeasurementError(f"Unsupported signal unit {unit!r}; expected volts ('V')")

    ch = _select_channel(epochs, config.p300_channel)
    mask = _window_mask(times, float(epochs.sampling_rate_hz),
                        config.p300_window_start_s, config.p300_window_end_s)
    n_samples = int(mask.sum())

    pid = getattr(epochs, "participant_id", None)
    rid = getattr(epochs, "recording_id", None)
    out: list[P300Measurement] = []
    excl: list[P300Exclusion] = []

    for i in range(n_ep):
        ev = epochs.events[i]
        trial_id, stim_id = _meta(ev, "trial_id"), _meta(ev, "stimulus_id")
        label = _meta(ev, "event_label")

        def exclude(reason: str) -> None:
            excl.append(P300Exclusion(pid, rid, trial_id, stim_id, label, reason))

        if trial_id is None or stim_id is None or label is None:
            exclude("missing event metadata (trial_id/stimulus_id/event_label)")
            continue
        seg = data[i, ch, mask]  # fancy-index copy; source array untouched
        if not np.all(np.isfinite(seg)):
            exclude("non-finite values in measurement window")
            continue
        out.append(P300Measurement(
            participant_id=pid, recording_id=rid, trial_id=trial_id, stimulus_id=stim_id,
            event_type=label, block_id=_meta(ev, "block_id"),
            event_sample_index=_meta(ev, "sample_index"), channel=config.p300_channel,
            window_start_s=config.p300_window_start_s, window_end_s=config.p300_window_end_s,
            n_samples=n_samples, p300_amplitude_v=float(seg.mean()),
        ))

    return P300MeasurementSet(tuple(out), tuple(excl), "V", n_ep)
