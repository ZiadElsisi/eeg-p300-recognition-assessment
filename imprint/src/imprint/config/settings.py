"""Typed settings shared by Imprint configuration and source adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real


def _is_int(value: object) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        type(value).__module__ == "numpy" and type(value).__name__.startswith(("int", "uint"))
    )


def _is_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _check_band(name: str, band: object) -> None:
    try:
        valid_length = len(band) == 2  # type: ignore[arg-type]
    except TypeError:
        valid_length = False
    if not valid_length:
        raise ValueError(f"{name} must contain exactly two values (low, high)")
    low, high = band  # type: ignore[misc]
    if not (_is_real(low) and _is_real(high)):
        raise ValueError(f"{name} endpoints must be finite numbers")
    if low < 0:
        raise ValueError(f"{name} lower bound must be >= 0")
    if low >= high:
        raise ValueError(f"{name} must satisfy low < high, got {tuple(band)}")  # type: ignore[arg-type]


@dataclass(frozen=True)
class GeneratorConfig:
    """Validated settings for synthetic background EEG.

    Amplitude parameters use microvolts, matching the supplied synthetic
    component model. The generator converts output arrays to canonical volts.
    ``n_samples`` is rounded from rate × requested duration; use
    ``duration_realized_s`` for the exact duration represented by the samples.

    This is a background-signal model. It does not create CIT event labels,
    participant conditions, or a physiologically realistic scalp topography.
    """

    n_channels: int = 8
    sampling_rate: float = 250.0
    duration_s: float = 10.0
    seed: int = 42

    brain_noise_uv: float = 8.0
    alpha_uv: float = 10.0
    alpha_band_hz: tuple[float, float] = (8.0, 12.0)
    line_noise_uv: float = 5.0
    line_freq_hz: float = 50.0
    drift_uv: float = 30.0
    drift_band_hz: tuple[float, float] = (0.05, 0.5)

    @property
    def n_samples(self) -> int:
        return int(round(self.sampling_rate * self.duration_s))

    @property
    def duration_realized_s(self) -> float:
        return self.n_samples / self.sampling_rate

    def validate(self) -> None:
        if not _is_int(self.n_channels):
            raise ValueError("n_channels must be an integer")
        if not _is_int(self.seed):
            raise ValueError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")

        for name in (
            "sampling_rate", "duration_s", "brain_noise_uv", "alpha_uv",
            "line_noise_uv", "line_freq_hz", "drift_uv",
        ):
            if not _is_real(getattr(self, name)):
                raise ValueError(f"{name} must be a finite number")

        _check_band("alpha_band_hz", self.alpha_band_hz)
        _check_band("drift_band_hz", self.drift_band_hz)

        if self.n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be > 0")
        if self.duration_s <= 0 or self.n_samples < 1:
            raise ValueError("duration_s must give at least one sample")
        for name in ("brain_noise_uv", "alpha_uv", "line_noise_uv", "drift_uv"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

        nyquist = self.sampling_rate / 2
        if self.line_noise_uv > 0:
            if self.line_freq_hz <= 0:
                raise ValueError("line_freq_hz must be > 0")
            if self.line_freq_hz >= nyquist:
                raise ValueError(f"line_freq_hz ({self.line_freq_hz}) must be below Nyquist ({nyquist})")
        if self.alpha_uv > 0 and self.alpha_band_hz[1] >= nyquist:
            raise ValueError("alpha band must be below Nyquist")
        if self.drift_uv > 0 and self.drift_band_hz[1] >= nyquist:
            raise ValueError("drift band must be below Nyquist")
