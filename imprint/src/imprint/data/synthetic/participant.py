
"""Participant information and experimental conditions."""

from dataclasses import dataclass
from enum import Enum


class ParticipantCondition(str, Enum):
    """Knowledge condition assigned to a participant."""

    KNOWS = "KNOWS"
    DOES_NOT_KNOW = "DOES_NOT_KNOW"
    ## in real data we have to add "unknown" as a condition.


@dataclass(frozen=True)
class Participant:
    """Represent a participant and their experimental condition."""

    participant_id: str
    condition: ParticipantCondition

    def __post_init__(self) -> None:
        """Validate participant information."""

        if not isinstance(self.participant_id, str):
            raise ValueError("participant_id must be a string")

        if not self.participant_id.strip():
            raise ValueError("participant_id cannot be empty")

        if not isinstance(self.condition, ParticipantCondition):
            raise ValueError(
                "condition must be a ParticipantCondition value"
            )
