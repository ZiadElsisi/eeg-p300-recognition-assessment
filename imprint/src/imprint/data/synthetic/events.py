
"""Event definitions and event data structures."""

from dataclasses import dataclass
from enum import Enum
import math


class EventLabel(str, Enum):
    """Available stimulus categories in the CIT experiment."""

    PROBE = "PROBE"
    IRRELEVANCE = "IRRELEVANCE"
    TARGET = "TARGET"


@dataclass(frozen=True)
class StimulusEvent:
    """Represent one stimulus event in a continuous EEG recording."""

    participant_id: str
    trial_id: str
    label: EventLabel
    onset_time_s: float
    sample_index: int

    def __post_init__(self) -> None:
        """Validate event information."""

        if not isinstance(self.participant_id, str) or not self.participant_id.strip():
            raise ValueError("participant_id cannot be empty")

        if not isinstance(self.trial_id, str) or not self.trial_id.strip():
            raise ValueError("trial_id cannot be empty")

        if not isinstance(self.label, EventLabel):
            raise ValueError("label must be an EventLabel value")

        if (
            isinstance(self.onset_time_s, bool)
            or not isinstance(self.onset_time_s, (int, float))
            or not math.isfinite(self.onset_time_s)
            or self.onset_time_s < 0
        ):
            raise ValueError("onset_time_s must be a finite non-negative number")

        if (
            isinstance(self.sample_index, bool)
            or not isinstance(self.sample_index, int)
            or self.sample_index < 0
        ):
            raise ValueError("sample_index must be a non-negative integer")
