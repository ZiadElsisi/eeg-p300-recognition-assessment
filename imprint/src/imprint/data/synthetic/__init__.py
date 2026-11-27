"""Deterministic synthetic EEG signal generation."""

from ...config import GeneratorConfig
from .generator import generate_base_recording, generate_components, generate_recording
from .source import SyntheticDataSource
from .participant import (
    Participant,
    ParticipantCohort,
    ParticipantCondition,
    assign_participant_conditions,
)

__all__ = [
    "GeneratorConfig",
    "generate_base_recording",
    "generate_components",
    "generate_recording",
    "SyntheticDataSource",
    "Participant",
    "ParticipantCohort",
    "ParticipantCondition",
    "assign_participant_conditions",
]
