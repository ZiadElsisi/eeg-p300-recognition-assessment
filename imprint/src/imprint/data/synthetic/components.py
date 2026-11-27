"""Background EEG components (adapted from the supplied IMPRINT-07 code).

The configurable amplitudes are expressed in microvolts. Component arrays are
converted to the architecture's canonical unit, volts, before they are returned.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ...config.settings import SyntheticSettings

FloatArray = NDArray[np.float64]
_COMPONENTS = ("brain_noise", "alpha", "line_noise", "drift")
_UV_TO_V = 1e-6


def time_axis(cfg: SyntheticSettings) -> FloatArray:
    """Time of each sample in seconds, shape ``(n_samples,)``."""
    return np.arange(cfg.n_samples, dtype=np.float64) / cfg.sampling_rate


def component_rngs(cfg: SyntheticSettings) -> dict[str, np.random.Generator]:
    """Create independent RNG streams so toggling one component leaves others stable."""
    children = np.random.SeedSequence(cfg.seed).spawn(len(_COMPONENTS))
    return {name: np.random.default_rng(child) for name, child in zip(_COMPONENTS, children)}


def brain_noise(cfg: SyntheticSettings, rng: np.random.Generator) -> FloatArray:
    """Approximate 1/f background with exact configured per-channel standard deviation."""
    n, n_channels = cfg.n_samples, cfg.n_channels
    if cfg.brain_noise_uv == 0 or n < 2:
        return np.zeros((n_channels, n), dtype=np.float64)

    white = rng.standard_normal((n_channels, n))
    spectrum = np.fft.rfft(white, axis=1)
    frequencies = np.fft.rfftfreq(n, d=1.0 / cfg.sampling_rate)
    scale = np.zeros_like(frequencies)
    scale[1:] = 1.0 / np.sqrt(frequencies[1:])
    pink = np.fft.irfft(spectrum * scale, n=n, axis=1)
    standard_deviation = pink.std(axis=1, keepdims=True)
    standard_deviation[standard_deviation == 0] = 1.0
    return pink / standard_deviation * (cfg.brain_noise_uv * _UV_TO_V)


def alpha_rhythm(cfg: SyntheticSettings, rng: np.random.Generator) -> FloatArray:
    """Generate a shared-band alpha frequency with channel-specific phase and envelope."""
    n, n_channels = cfg.n_samples, cfg.n_channels
    if cfg.alpha_uv == 0:
        return np.zeros((n_channels, n), dtype=np.float64)

    frequency = rng.uniform(*cfg.alpha_band_hz)
    phase = rng.uniform(0, 2 * np.pi, size=(n_channels, 1))
    gain = rng.uniform(*cfg.alpha_channel_gain_range, size=(n_channels, 1))
    envelope_frequency = rng.uniform(*cfg.alpha_envelope_frequency_range_hz)
    envelope_phase = rng.uniform(0, 2 * np.pi, size=(n_channels, 1))
    time = time_axis(cfg)[None, :]
    envelope = 1.0 + cfg.alpha_envelope_modulation * np.sin(
        2 * np.pi * envelope_frequency * time + envelope_phase
    )
    return cfg.alpha_uv * _UV_TO_V * gain * envelope * np.sin(2 * np.pi * frequency * time + phase)


def line_noise(cfg: SyntheticSettings, rng: np.random.Generator) -> FloatArray:
    """Generate configured mains interference with a per-channel gain."""
    n, n_channels = cfg.n_samples, cfg.n_channels
    if cfg.line_noise_uv == 0:
        return np.zeros((n_channels, n), dtype=np.float64)

    phase = rng.uniform(0, 2 * np.pi, size=(n_channels, 1))
    gain = rng.uniform(*cfg.line_channel_gain_range, size=(n_channels, 1))
    time = time_axis(cfg)[None, :]
    return cfg.line_noise_uv * _UV_TO_V * gain * np.sin(
        2 * np.pi * cfg.line_freq_hz * time + phase
    )


def slow_drift(cfg: SyntheticSettings, rng: np.random.Generator) -> FloatArray:
    """Generate three low-frequency sinusoids, normalized to configured peak amplitude."""
    n, n_channels = cfg.n_samples, cfg.n_channels
    if cfg.drift_uv == 0:
        return np.zeros((n_channels, n), dtype=np.float64)

    n_waves = cfg.drift_component_count
    frequencies = rng.uniform(*cfg.drift_band_hz, size=(n_channels, n_waves, 1))
    phases = rng.uniform(0, 2 * np.pi, size=(n_channels, n_waves, 1))
    weights = rng.uniform(*cfg.drift_component_weight_range, size=(n_channels, n_waves, 1))
    time = time_axis(cfg)[None, None, :]
    drift = (weights * np.sin(2 * np.pi * frequencies * time + phases)).sum(axis=1)
    peak = np.abs(drift).max(axis=1, keepdims=True)
    peak[peak == 0] = 1.0
    return drift / peak * (cfg.drift_uv * _UV_TO_V)
