"""Deterministic synthetic EEG signal generation."""

from ...config import GeneratorConfig
from .generator import generate_base_recording, generate_components, generate_recording

__all__ = [
    "GeneratorConfig",
    "generate_base_recording",
    "generate_components",
    "generate_recording",
]
