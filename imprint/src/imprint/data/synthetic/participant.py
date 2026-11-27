
"""Participant conditions and their reproducible evaluation-only record."""

import json
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


_COHORT_SCHEMA = "imprint.synthetic-participant-conditions.v1"


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


@dataclass(frozen=True)
class ParticipantCohort:
    """Separate, serializable assignment of synthetic participant conditions.

    This object contains evaluation truth. Keep it separate from recordings,
    stimulus events, and participant-analysis inputs.
    """

    participants: tuple[Participant, ...]
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.participants, tuple) or not self.participants:
            raise ValueError("participants must be a non-empty tuple")
        if any(not isinstance(person, Participant) for person in self.participants):
            raise ValueError("participants must contain Participant instances")
        participant_ids = [person.participant_id for person in self.participants]
        if len(participant_ids) != len(set(participant_ids)):
            raise ValueError("participant IDs must be unique")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        """Return a versioned representation containing evaluation truth."""
        return {
            "schema": _COHORT_SCHEMA,
            "seed": int(self.seed),
            "participants": [
                {
                    "participant_id": person.participant_id,
                    "condition": person.condition.value,
                }
                for person in self.participants
            ],
        }

    def save(self, path: str | Path) -> Path:
        """Persist this evaluation-only assignment as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def from_dict(cls, data: object) -> "ParticipantCohort":
        """Restore an assignment from its versioned serialized form."""
        if not isinstance(data, dict) or data.get("schema") != _COHORT_SCHEMA:
            raise ValueError(f"Expected cohort schema {_COHORT_SCHEMA!r}")

        seed = data.get("seed")
        records = data.get("participants")
        if not isinstance(records, list):
            raise ValueError("participants must be a list")

        participants: list[Participant] = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("each participant record must be an object")
            try:
                condition = ParticipantCondition(record.get("condition"))
            except (TypeError, ValueError) as exc:
                raise ValueError("participant condition is unsupported") from exc
            participants.append(
                Participant(
                    participant_id=record.get("participant_id"),
                    condition=condition,
                )
            )

        return cls(participants=tuple(participants), seed=seed)

    @classmethod
    def load(cls, path: str | Path) -> "ParticipantCohort":
        """Load a previously persisted evaluation-only assignment."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def assign_participant_conditions(
    participant_ids: Sequence[str],
    condition_counts: Mapping[ParticipantCondition, int],
    *,
    seed: int,
) -> ParticipantCohort:
    """Assign an exact condition balance to IDs reproducibly.

    IDs retain their input order; the condition allocation is shuffled using
    ``seed``. Counts must sum to the number of IDs so no assignment is implicit.
    """
    if isinstance(participant_ids, (str, bytes)):
        raise ValueError("participant_ids must be a sequence of unique IDs")
    try:
        ids = tuple(participant_ids)
    except TypeError as exc:
        raise ValueError("participant_ids must be a sequence of unique IDs") from exc

    if not ids:
        raise ValueError("at least one participant ID is required")
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("participant IDs must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("participant IDs must be unique")
    if not isinstance(condition_counts, Mapping):
        raise ValueError("condition_counts must map ParticipantCondition to counts")
    if any(not isinstance(condition, ParticipantCondition) for condition in condition_counts):
        raise ValueError("condition_counts keys must be ParticipantCondition values")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    counts: dict[ParticipantCondition, int] = {}
    for condition in ParticipantCondition:
        count = condition_counts.get(condition, 0)
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("condition counts must be non-negative integers")
        counts[condition] = int(count)

    if sum(counts.values()) != len(ids):
        raise ValueError("condition counts must sum to the number of participant IDs")

    assignments = [
        condition
        for condition in ParticipantCondition
        for _ in range(counts[condition])
    ]
    order = np.random.default_rng(int(seed)).permutation(len(ids))
    participants = tuple(
        Participant(participant_id=participant_id, condition=assignments[index])
        for participant_id, index in zip(ids, order)
    )
    return ParticipantCohort(participants=participants, seed=int(seed))
