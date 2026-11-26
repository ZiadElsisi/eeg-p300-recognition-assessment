"""Orchestration: deterministic base recording + component composition."""

from __future__ import annotations

import numpy as np

from .components import (
    alpha_rhythm,
    brain_noise,
    component_rngs,
    line_noise,
    slow_drift,
)
from ...config import GeneratorConfig


def generate_base_recording(cfg: GeneratorConfig) -> np.ndarray:
    """Return zero-valued continuous data: (channels, samples), float64, volts."""
    cfg.validate()
    return np.zeros((cfg.n_channels, cfg.n_samples), dtype=np.float64)


def generate_components(cfg: GeneratorConfig) -> dict[str, np.ndarray]:
    """Return each signal component separately, in volts."""
    cfg.validate()
    rngs = component_rngs(cfg)
    return {
        "brain_noise": brain_noise(cfg, rngs["brain_noise"]),
        "alpha": alpha_rhythm(cfg, rngs["alpha"]),
        "line_noise": line_noise(cfg, rngs["line_noise"]),
        "drift": slow_drift(cfg, rngs["drift"]),
    }


def generate_recording(cfg: GeneratorConfig | None = None) -> np.ndarray:
    """Return synthetic continuous EEG: (channels, samples), float64, volts.

    This function returns signal data only. Participant condition and other
    evaluation labels must be generated and returned separately by a source
    adapter; they are never embedded in this array.
    """
    cfg = cfg or GeneratorConfig()
    cfg.validate()
    data = generate_base_recording(cfg)
    for comp in generate_components(cfg).values():
        if comp.shape != data.shape:
            raise ValueError(f"component shape {comp.shape} does not match recording shape {data.shape}")
        data += comp
    return data
