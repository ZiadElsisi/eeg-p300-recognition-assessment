"""Canonical, analysis-facing EEG recording model.

Signal arrays use ``(channels, samples)`` order and volts (``V``). Construction
does not copy a valid NumPy array: the recording takes ownership of its
writeability flag and marks the array read-only. Callers must not mutate the
underlying memory through another alias after handoff. Reading ``data`` is a
zero-copy operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .events import StimulusEvent


class ChannelRole(str, Enum):
    """Broad channel category, independent of source-specific hardware names."""

    BRAIN = "brain"
    EYE = "eye"
    OTHER = "other"


@dataclass(frozen=True)
class ChannelInfo:
    """Metadata for one signal row."""

    name: str
    role: ChannelRole
    unit: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("channel.name must be a non-empty string")
        if not isinstance(self.role, ChannelRole):
            raise ValueError("channel.role must be a ChannelRole value")
        if self.unit is not None and self.unit != "V":
            raise ValueError("channel.unit must be canonical volts ('V') when specified")


_FORBIDDEN_METADATA_KEYS = {
    "condition", "participant_condition", "knowledge_status", "ground_truth",
    "groundtruth", "countermeasure", "countermeasure_label", "evaluation_label",
}


def _validate_analysis_metadata(value: Any, path: str = "metadata") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.casefold() in _FORBIDDEN_METADATA_KEYS:
                raise ValueError(f"{path} must not contain evaluation-only field {key!r}")
            _validate_analysis_metadata(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_analysis_metadata(child, f"{path}[{index}]")


@dataclass(frozen=True)
class Recording:
    """Shared continuous EEG representation for all Imprint data sources.

    ``data`` is a two-dimensional real numeric NumPy array in volts, shaped
    ``(n_channels, n_samples)``. It is kept zero-copy and made read-only.
    Metadata is shallowly read-only; values should be treated as descriptive
    provenance and must not contain participant condition or other evaluation
    truth.
    """

    data: np.ndarray
    sampling_rate_hz: float
    channels: tuple[ChannelInfo, ...]
    events: tuple[StimulusEvent, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a NumPy array")
        if self.data.ndim != 2:
            raise ValueError("data must have exactly two dimensions: (channels, samples)")
        if self.data.shape[0] <= 0 or self.data.shape[1] <= 0:
            raise ValueError("data must contain at least one channel and one sample")
        if not np.issubdtype(self.data.dtype, np.number) or np.issubdtype(self.data.dtype, np.complexfloating):
            raise TypeError("data must have a real numeric dtype")
        if not np.isfinite(self.data).all():
            raise ValueError("data must contain only finite values")
        if isinstance(self.sampling_rate_hz, bool) or not isinstance(self.sampling_rate_hz, Real):
            raise ValueError("sampling_rate_hz must be a positive finite number")
        if not math.isfinite(float(self.sampling_rate_hz)) or self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be a positive finite number")

        if not isinstance(self.channels, tuple) or any(not isinstance(ch, ChannelInfo) for ch in self.channels):
            raise TypeError("channels must be a tuple of ChannelInfo values")
        if self.data.shape[0] != len(self.channels):
            raise ValueError("data channel dimension must match the number of channel metadata entries")
        names = tuple(channel.name for channel in self.channels)
        if len(names) != len(set(names)):
            raise ValueError("channel names must be unique")

        if not isinstance(self.events, tuple) or any(not isinstance(event, StimulusEvent) for event in self.events):
            raise TypeError("events must be a tuple of StimulusEvent values")
        for event in self.events:
            if event.sample_index >= self.data.shape[1]:
                raise ValueError(
                    f"event sample_index {event.sample_index} is outside recording bounds "
                    f"[0, {self.data.shape[1] - 1}]"
                )

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        _validate_analysis_metadata(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        self.data.setflags(write=False)

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / float(self.sampling_rate_hz)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)

    @property
    def signal_unit(self) -> str:
        """Canonical unit for ``data``: volts."""
        return "V"
