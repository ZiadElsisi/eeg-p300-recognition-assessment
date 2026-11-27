"""Validated, immutable configuration models for Imprint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _names(name: str, values: tuple[str, ...]) -> None:
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty names")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicate names")


@dataclass(frozen=True)
class SignalSettings:
    sampling_rate: float
    brain_channel_count: int
    eye_channel_count: int
    total_channel_count: int
    signal_unit: str
    brain_channels: tuple[str, ...]
    primary_p300_channel: str
    eye_channels: tuple[str, ...] | None = None
    eye_device_inputs: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if _finite_real("signal.sampling_rate", self.sampling_rate) <= 0:
            raise ValueError("signal.sampling_rate must be positive")
        _positive_int("signal.brain_channel_count", self.brain_channel_count)
        _positive_int("signal.eye_channel_count", self.eye_channel_count)
        _positive_int("signal.total_channel_count", self.total_channel_count)
        _names("signal.brain_channels", self.brain_channels)
        if self.brain_channel_count != len(self.brain_channels):
            raise ValueError("signal.brain_channel_count must match signal.brain_channels")
        if self.total_channel_count != self.brain_channel_count + self.eye_channel_count:
            raise ValueError("signal.total_channel_count must equal brain plus eye channel counts")
        if self.signal_unit != "V":
            raise ValueError("signal.signal_unit must be canonical volts ('V')")
        if self.primary_p300_channel not in self.brain_channels:
            raise ValueError("signal.primary_p300_channel must be one of the brain channels")
        for key, channels in (("eye_channels", self.eye_channels), ("eye_device_inputs", self.eye_device_inputs)):
            if channels is not None:
                if len(channels) != self.eye_channel_count:
                    raise ValueError(f"signal.{key} must contain exactly {self.eye_channel_count} entries")
                _names(f"signal.{key}", channels)
        if self.eye_channels and set(self.brain_channels).intersection(self.eye_channels):
            raise ValueError("signal.brain_channels and signal.eye_channels must not overlap")


@dataclass(frozen=True)
class FilteringSettings:
    filter_type: str
    low_cutoff_hz: float
    high_cutoff_hz: float
    zero_phase: bool
    filter_order: int | None = None

    def validate(self, sampling_rate: float) -> None:
        low = _finite_real("filtering.low_cutoff_hz", self.low_cutoff_hz)
        high = _finite_real("filtering.high_cutoff_hz", self.high_cutoff_hz)
        if self.filter_type.lower() != "butterworth":
            raise ValueError("filtering.filter_type must be 'butterworth'")
        if low <= 0 or low >= high:
            raise ValueError("filtering cutoffs must satisfy 0 < low_cutoff_hz < high_cutoff_hz")
        if high >= sampling_rate / 2:
            raise ValueError("filtering.high_cutoff_hz must be below the Nyquist frequency")
        if not isinstance(self.zero_phase, bool):
            raise ValueError("filtering.zero_phase must be a boolean")
        if self.filter_order is not None:
            _positive_int("filtering.filter_order", self.filter_order)


@dataclass(frozen=True)
class EpochSettings:
    epoch_start_seconds: float
    epoch_end_seconds: float
    baseline_start_seconds: float
    baseline_end_seconds: float

    def __post_init__(self) -> None:
        start = _finite_real("epoching.epoch_start_seconds", self.epoch_start_seconds)
        end = _finite_real("epoching.epoch_end_seconds", self.epoch_end_seconds)
        baseline_start = _finite_real("epoching.baseline_start_seconds", self.baseline_start_seconds)
        baseline_end = _finite_real("epoching.baseline_end_seconds", self.baseline_end_seconds)
        if start >= end:
            raise ValueError("epoching.epoch_start_seconds must be earlier than epoch_end_seconds")
        if baseline_start >= baseline_end:
            raise ValueError("epoching.baseline_start_seconds must be earlier than baseline_end_seconds")
        if baseline_start < start or baseline_end > end:
            raise ValueError("epoching baseline interval must lie within the epoch interval")


@dataclass(frozen=True)
class ArtifactRejectionSettings:
    peak_to_peak_threshold_uv: float
    apply_to_brain_channels: bool
    inconclusive_on_poor_recording: bool

    def __post_init__(self) -> None:
        if _finite_real("artifact_rejection.peak_to_peak_threshold_uv", self.peak_to_peak_threshold_uv) <= 0:
            raise ValueError("artifact_rejection.peak_to_peak_threshold_uv must be positive")
        if not isinstance(self.apply_to_brain_channels, bool):
            raise ValueError("artifact_rejection.apply_to_brain_channels must be a boolean")
        if not isinstance(self.inconclusive_on_poor_recording, bool):
            raise ValueError("artifact_rejection.inconclusive_on_poor_recording must be a boolean")


@dataclass(frozen=True)
class P300Settings:
    channel: str
    measurement_start_ms: float
    measurement_end_ms: float
    measurement: str

    def __post_init__(self) -> None:
        start = _finite_real("p300.measurement_start_ms", self.measurement_start_ms)
        end = _finite_real("p300.measurement_end_ms", self.measurement_end_ms)
        if start >= end:
            raise ValueError("p300 measurement start must be earlier than its end")
        if self.measurement != "mean_amplitude":
            raise ValueError("p300.measurement must be 'mean_amplitude'")


@dataclass(frozen=True)
class BootstrapSettings:
    iterations: int
    recognition_threshold: float
    minimum_probe_epochs: int
    comparison_groups: tuple[str, ...]
    equalize_samples: bool
    resampling_method: str | None = None

    def __post_init__(self) -> None:
        _positive_int("bootstrap.iterations", self.iterations)
        _positive_int("bootstrap.minimum_probe_epochs", self.minimum_probe_epochs)
        threshold = _finite_real("bootstrap.recognition_threshold", self.recognition_threshold)
        if not 0 <= threshold <= 1:
            raise ValueError("bootstrap.recognition_threshold must be between 0 and 1")
        if self.comparison_groups != ("Probe", "Irrelevance"):
            raise ValueError("bootstrap.comparison_groups must be ['Probe', 'Irrelevance']")
        if not isinstance(self.equalize_samples, bool) or not self.equalize_samples:
            raise ValueError("bootstrap.equalize_samples must be true for the primary comparison")
        if self.resampling_method is not None and not self.resampling_method.strip():
            raise ValueError("bootstrap.resampling_method must be non-empty when specified")


@dataclass(frozen=True)
class BlinkCorrectionSettings:
    method: str
    use_eye_channels: bool
    compare_probe_irrelevance: bool
    blink_ratio_difference_threshold: float
    require_significant_fisher_test: bool
    remove_category_average_response: bool
    fisher_significance_level: float | None = None

    def __post_init__(self) -> None:
        if self.method != "regression":
            raise ValueError("blink_correction.method must be 'regression'")
        for name in ("use_eye_channels", "compare_probe_irrelevance", "require_significant_fisher_test", "remove_category_average_response"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"blink_correction.{name} must be a boolean")
        threshold = _finite_real("blink_correction.blink_ratio_difference_threshold", self.blink_ratio_difference_threshold)
        if not 0 <= threshold <= 1:
            raise ValueError("blink_correction.blink_ratio_difference_threshold must be between 0 and 1")
        if self.fisher_significance_level is not None:
            level = _finite_real("blink_correction.fisher_significance_level", self.fisher_significance_level)
            if not 0 < level < 1:
                raise ValueError("blink_correction.fisher_significance_level must be between 0 and 1")


@dataclass(frozen=True)
class CountermeasureSettings:
    irrelevance_comparison_threshold: float
    reaction_time_difference_threshold_ms: float
    compare_against_remaining_irrelevance: bool
    deception_flag_on_detection: bool

    def __post_init__(self) -> None:
        threshold = _finite_real("countermeasure.irrelevance_comparison_threshold", self.irrelevance_comparison_threshold)
        if not 0 <= threshold <= 1:
            raise ValueError("countermeasure.irrelevance_comparison_threshold must be between 0 and 1")
        if _finite_real("countermeasure.reaction_time_difference_threshold_ms", self.reaction_time_difference_threshold_ms) <= 0:
            raise ValueError("countermeasure.reaction_time_difference_threshold_ms must be positive")
        for name in ("compare_against_remaining_irrelevance", "deception_flag_on_detection"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"countermeasure.{name} must be a boolean")


@dataclass(frozen=True)
class SessionSettings:
    probe_pictures: tuple[str, ...]
    picture_set_count: int
    probe_pictures_per_set: int
    irrelevance_pictures_per_set: int
    target_pictures_per_set: int
    repetitions_per_picture: int
    block_count: int
    picture_duration_ms: float
    inter_picture_interval_min_seconds: float
    inter_picture_interval_max_seconds: float
    approximate_stimulus_duration_minutes: float
    initial_delay_seconds: float = 1.0
    legacy_fixed_inter_event_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        _names("session.probe_pictures", self.probe_pictures)
        for field_name in (
            "picture_set_count", "probe_pictures_per_set", "irrelevance_pictures_per_set",
            "target_pictures_per_set", "repetitions_per_picture", "block_count",
        ):
            _positive_int(f"session.{field_name}", getattr(self, field_name))
        if self.probe_pictures_per_set * self.picture_set_count != len(self.probe_pictures):
            raise ValueError("session.probe_pictures must match probe_pictures_per_set × picture_set_count")
        if _finite_real("session.picture_duration_ms", self.picture_duration_ms) <= 0:
            raise ValueError("session.picture_duration_ms must be positive")
        low = _finite_real("session.inter_picture_interval_min_seconds", self.inter_picture_interval_min_seconds)
        high = _finite_real("session.inter_picture_interval_max_seconds", self.inter_picture_interval_max_seconds)
        if low <= 0 or low > high:
            raise ValueError("session interval minimum must be positive and no greater than its maximum")
        if _finite_real("session.approximate_stimulus_duration_minutes", self.approximate_stimulus_duration_minutes) <= 0:
            raise ValueError("session.approximate_stimulus_duration_minutes must be positive")
        if _finite_real("session.initial_delay_seconds", self.initial_delay_seconds) < 0:
            raise ValueError("session.initial_delay_seconds must be non-negative")
        if _finite_real("session.legacy_fixed_inter_event_interval_seconds", self.legacy_fixed_inter_event_interval_seconds) <= 0:
            raise ValueError("session.legacy_fixed_inter_event_interval_seconds must be positive")
        if self.total_trials % self.block_count:
            raise ValueError("session.block_count must divide the derived total trial count")

    @property
    def picture_count(self) -> int:
        return self.picture_set_count * (
            self.probe_pictures_per_set
            + self.irrelevance_pictures_per_set
            + self.target_pictures_per_set
        )

    @property
    def total_trials(self) -> int:
        return self.picture_count * self.repetitions_per_picture

    @property
    def trials_per_block(self) -> int:
        return self.total_trials // self.block_count

    @property
    def maximum_schedule_duration_seconds(self) -> float:
        """Maximum nominal session span from the configured upper interval bound."""
        return (
            self.initial_delay_seconds
            + max(0, self.total_trials - 1) * self.inter_picture_interval_max_seconds
            + self.picture_duration_ms / 1000.0
        )


@dataclass(frozen=True)
class SyntheticSettings:
    """Synthetic generator settings, with acquisition/seed inherited centrally."""

    n_channels: int
    sampling_rate: float
    duration_s: float
    seed: int
    brain_noise_uv: float
    alpha_uv: float
    alpha_band_hz: tuple[float, float]
    line_noise_uv: float
    line_freq_hz: float
    drift_uv: float
    drift_band_hz: tuple[float, float]
    alpha_envelope_frequency_range_hz: tuple[float, float]
    alpha_channel_gain_range: tuple[float, float]
    alpha_envelope_modulation: float
    line_channel_gain_range: tuple[float, float]
    drift_component_count: int
    drift_component_weight_range: tuple[float, float]

    @property
    def n_samples(self) -> int:
        return int(self.sampling_rate * self.duration_s)

    @property
    def duration_realized_s(self) -> float:
        return self.n_samples / self.sampling_rate

    def validate(self) -> None:
        _positive_int("synthetic.n_channels", self.n_channels)
        if _finite_real("synthetic.sampling_rate", self.sampling_rate) <= 0:
            raise ValueError("synthetic.sampling_rate must be positive")
        if _nonnegative_int("reproducibility.default_seed", self.seed) != self.seed:
            raise ValueError("reproducibility.default_seed must be a non-negative integer")
        if _finite_real("synthetic.duration_s", self.duration_s) <= 0 or self.n_samples < 1:
            raise ValueError("synthetic.duration_s must produce at least one sample")
        for name in ("brain_noise_uv", "alpha_uv", "line_noise_uv", "drift_uv"):
            if _finite_real(f"synthetic.{name}", getattr(self, name)) < 0:
                raise ValueError(f"synthetic.{name} must be non-negative")
        for name, band in (
            ("alpha_band_hz", self.alpha_band_hz),
            ("drift_band_hz", self.drift_band_hz),
            ("alpha_envelope_frequency_range_hz", self.alpha_envelope_frequency_range_hz),
            ("alpha_channel_gain_range", self.alpha_channel_gain_range),
            ("line_channel_gain_range", self.line_channel_gain_range),
            ("drift_component_weight_range", self.drift_component_weight_range),
        ):
            if len(band) != 2 or not all(math.isfinite(float(value)) for value in band):
                raise ValueError(f"synthetic.{name} must contain two finite frequencies")
            if band[0] < 0 or band[0] >= band[1]:
                raise ValueError(f"synthetic.{name} must satisfy 0 <= low < high")
            if name in ("alpha_band_hz", "drift_band_hz", "alpha_envelope_frequency_range_hz") and band[1] >= self.sampling_rate / 2:
                raise ValueError(f"synthetic.{name} must be below Nyquist")
        modulation = _finite_real("synthetic.alpha_envelope_modulation", self.alpha_envelope_modulation)
        if not 0 <= modulation <= 1:
            raise ValueError("synthetic.alpha_envelope_modulation must be between 0 and 1")
        _positive_int("synthetic.drift_component_count", self.drift_component_count)
        line_frequency = _finite_real("synthetic.line_frequency_hz", self.line_freq_hz)
        if self.line_noise_uv > 0 and not 0 < line_frequency < self.sampling_rate / 2:
            raise ValueError("synthetic.line_frequency_hz must be below Nyquist when line noise is enabled")


@dataclass(frozen=True)
class ReproducibilitySettings:
    default_seed: int
    capture_effective_config: bool
    capture_overrides: bool
    capture_code_revision: bool
    capture_software_versions: bool
    capture_source_identity: bool

    def __post_init__(self) -> None:
        _nonnegative_int("reproducibility.default_seed", self.default_seed)
        for name in (
            "capture_effective_config", "capture_overrides", "capture_code_revision",
            "capture_software_versions", "capture_source_identity",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"reproducibility.{name} must be a boolean")


@dataclass(frozen=True)
class ImprintConfig:
    signal: SignalSettings
    filtering: FilteringSettings
    epoching: EpochSettings
    artifact_rejection: ArtifactRejectionSettings
    p300: P300Settings
    bootstrap: BootstrapSettings
    blink_correction: BlinkCorrectionSettings
    countermeasure: CountermeasureSettings
    session: SessionSettings
    synthetic: SyntheticSettings
    reproducibility: ReproducibilitySettings
    keep_proxy_labels_explicit: bool
    reporting_output_directory: str
    reporting_format: str
    overrides_json: str = "{}"

    def __post_init__(self) -> None:
        self.filtering.validate(self.signal.sampling_rate)
        if self.p300.channel != self.signal.primary_p300_channel:
            raise ValueError("p300.channel must match signal.primary_p300_channel")
        if self.p300.channel not in self.signal.brain_channels:
            raise ValueError("p300.channel must be one of signal.brain_channels")
        if self.synthetic.n_channels != self.signal.total_channel_count:
            raise ValueError("synthetic channel count must match signal.total_channel_count")
        if self.synthetic.sampling_rate != self.signal.sampling_rate:
            raise ValueError("synthetic sampling rate must match signal.sampling_rate")
        if self.synthetic.seed != self.reproducibility.default_seed:
            raise ValueError("synthetic seed must come from reproducibility.default_seed")
        if not isinstance(self.keep_proxy_labels_explicit, bool):
            raise ValueError("evaluation.keep_proxy_labels_explicit must be a boolean")
        if not self.reporting_output_directory.strip():
            raise ValueError("reporting.output_directory must be non-empty")
        if not self.reporting_format.strip():
            raise ValueError("reporting.format must be non-empty")
        self.synthetic.validate()

    def to_dict(self) -> dict[str, Any]:
        """Return the effective configuration, including values derived at load time."""
        result = {
            "signal": asdict(self.signal),
            "filtering": asdict(self.filtering),
            "epoching": asdict(self.epoching),
            "artifact_rejection": asdict(self.artifact_rejection),
            "p300": asdict(self.p300),
            "bootstrap": asdict(self.bootstrap),
            "blink_correction": asdict(self.blink_correction),
            "countermeasure": asdict(self.countermeasure),
            "session": asdict(self.session),
            "synthetic": asdict(self.synthetic),
            "reproducibility": asdict(self.reproducibility),
            "evaluation": {"keep_proxy_labels_explicit": self.keep_proxy_labels_explicit},
            "reporting": {
                "output_directory": self.reporting_output_directory,
                "format": self.reporting_format,
            },
        }
        result["signal"]["channel_count"] = self.signal.total_channel_count
        result["session"]["picture_count"] = self.session.picture_count
        result["session"]["total_trials"] = self.session.total_trials
        result["session"]["trials_per_block"] = self.session.trials_per_block
        result["overrides"] = (
            json.loads(self.overrides_json)
            if self.reproducibility.capture_overrides
            else {}
        )
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save_effective_config(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json() + "\n", encoding="utf-8")
        return output_path
