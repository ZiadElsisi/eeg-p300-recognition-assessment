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
from ...config.loader import load_config
from ...config.settings import ImprintConfig, SyntheticSettings


ConfigInput = SyntheticSettings | ImprintConfig


def _synthetic_settings(cfg: ConfigInput | None) -> SyntheticSettings:
    if cfg is None:
        return load_config().synthetic
    if isinstance(cfg, ImprintConfig):
        return cfg.synthetic
    if isinstance(cfg, SyntheticSettings):
        return cfg
    raise TypeError("cfg must be an ImprintConfig or SyntheticSettings")


def generate_base_recording(cfg: ConfigInput) -> np.ndarray:
    """Return zero-valued continuous data: (channels, samples), float64, volts."""
    cfg = _synthetic_settings(cfg)
    cfg.validate()
    return np.zeros((cfg.n_channels, cfg.n_samples), dtype=np.float64)


def generate_components(cfg: ConfigInput) -> dict[str, np.ndarray]:
    """Return each signal component separately, in volts."""
    cfg = _synthetic_settings(cfg)
    cfg.validate()
    rngs = component_rngs(cfg)
    return {
        "brain_noise": brain_noise(cfg, rngs["brain_noise"]),
        "alpha": alpha_rhythm(cfg, rngs["alpha"]),
        "line_noise": line_noise(cfg, rngs["line_noise"]),
        "drift": slow_drift(cfg, rngs["drift"]),
    }


def generate_recording(cfg: ConfigInput | None = None) -> np.ndarray:
    """Return synthetic continuous EEG: (channels, samples), float64, volts.

    This function returns signal data only. Participant condition and other
    evaluation labels must be generated and returned separately by a source
    adapter; they are never embedded in this array.
    """
    cfg = _synthetic_settings(cfg)
    cfg.validate()
    data = generate_base_recording(cfg)
    for comp in generate_components(cfg).values():
        if comp.shape != data.shape:
            raise ValueError(f"component shape {comp.shape} does not match recording shape {data.shape}")
        data += comp
    return data
