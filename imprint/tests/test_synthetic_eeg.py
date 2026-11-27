import dataclasses
from pathlib import Path
import sys

import numpy as np
import pytest

# Make direct execution work with the repository's src layout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imprint.data.synthetic import (
    GeneratorConfig,
    generate_base_recording,
    generate_components,
    generate_recording,
)

FS = GeneratorConfig().sampling_rate
UV = 1e-6  # Generator output is in volts; config amplitudes are in microvolts.


def psd(x, fs=FS):
    """Periodogram of one channel -> (freqs, power)."""
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    return np.fft.rfftfreq(len(x), 1 / fs), spec


def welch(x, fs=FS, nperseg=512):
    """Minimal Welch PSD (Hann, 50% overlap) -> (freqs, power)."""
    step = nperseg // 2
    win = np.hanning(nperseg)
    segs = [
        np.abs(np.fft.rfft((x[i:i + nperseg] - x[i:i + nperseg].mean()) * win)) ** 2
        for i in range(0, len(x) - nperseg + 1, step)
    ]
    return np.fft.rfftfreq(nperseg, 1 / fs), np.mean(segs, axis=0)


def only(**enabled):
    """Config with every component off except the ones given."""
    base = dict(brain_noise_uv=0, alpha_uv=0, line_noise_uv=0, drift_uv=0)
    base.update(enabled)
    return GeneratorConfig(**base)


# ---------------- IMPRINT-06 ----------------
def test_default_shape_is_8_by_2500():
    x = generate_recording(GeneratorConfig())
    assert x.shape == (8, 2500)
    assert x.dtype == np.float64
    assert np.isfinite(x).all()


def test_sampling_rate_and_duration_change_shape():
    assert generate_recording(GeneratorConfig(sampling_rate=500, duration_s=2)).shape == (8, 1000)
    assert generate_recording(GeneratorConfig(sampling_rate=125, duration_s=4)).shape == (8, 500)


def test_channel_count_configurable():
    assert generate_recording(GeneratorConfig(n_channels=4)).shape[0] == 4


def test_base_recording_is_zeros():
    assert not generate_base_recording(GeneratorConfig()).any()


def test_same_seed_is_identical():
    a = generate_recording(GeneratorConfig(seed=42))
    b = generate_recording(GeneratorConfig(seed=42))
    np.testing.assert_array_equal(a, b)


def test_different_seed_differs():
    a = generate_recording(GeneratorConfig(seed=1))
    b = generate_recording(GeneratorConfig(seed=2))
    assert not np.allclose(a, b)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_channels=0),
        dict(n_channels=8.5),
        dict(n_channels=True),
        dict(sampling_rate=0),
        dict(sampling_rate=float("nan")),
        dict(sampling_rate=float("inf")),
        dict(duration_s=0),
        dict(duration_s=float("nan")),
        dict(duration_s=0.001),  # rounds to 0 samples
        dict(sampling_rate=80),  # 50 Hz mains above Nyquist (40 Hz)
        dict(alpha_uv=-1),
        dict(brain_noise_uv=float("nan")),
        dict(drift_uv=float("inf")),
        dict(line_freq_hz=0),
        dict(line_freq_hz=float("nan")),
        dict(alpha_band_hz=(12.0, 8.0)),
        dict(alpha_band_hz=(8.0, 8.0)),
        dict(alpha_band_hz=(8.0,)),
        dict(alpha_band_hz=(-1.0, 8.0)),
        dict(alpha_band_hz=(8.0, float("nan"))),
        dict(drift_band_hz=(0.5, 0.05)),
        dict(drift_band_hz=(0.05, 200.0)),  # above Nyquist
        dict(seed=-1),
        dict(seed=1.5),
        dict(seed=True),
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        generate_recording(GeneratorConfig(**kwargs))


def test_low_sampling_rate_ok_when_line_noise_disabled():
    cfg = GeneratorConfig(sampling_rate=80, line_noise_uv=0)
    x = generate_recording(cfg)
    assert x.shape == (8, 800)
    assert np.isfinite(x).all()


def test_numpy_integer_types_accepted():
    cfg = GeneratorConfig(n_channels=np.int64(4), seed=np.int64(3))
    assert generate_recording(cfg).shape[0] == 4


# ---------------- sample-count truncation / short recordings ----------------
def test_non_integer_product_uses_integer_sample_count_and_exposes_realized_duration():
    cfg = GeneratorConfig(sampling_rate=333, duration_s=0.01, line_noise_uv=0)
    assert cfg.n_samples == 3  # int(3.33)
    assert cfg.duration_realized_s == pytest.approx(3 / 333)
    assert generate_recording(cfg).shape == (8, 3)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_very_short_recordings_are_finite(n):
    cfg = GeneratorConfig(sampling_rate=250, duration_s=n / 250)
    x = generate_recording(cfg)
    assert x.shape == (8, n)
    assert np.isfinite(x).all()


# ---------------- IMPRINT-07 ----------------
def test_components_sum_to_recording():
    cfg = GeneratorConfig(seed=7)
    total = sum(generate_components(cfg).values())
    np.testing.assert_allclose(generate_recording(cfg), total)


@pytest.mark.parametrize("disabled", ["brain_noise", "alpha", "line_noise", "drift"])
def test_disabling_one_component_does_not_change_others(disabled):
    cfg = GeneratorConfig(seed=7)
    full = generate_components(cfg)
    off = generate_components(dataclasses.replace(cfg, **{f"{disabled}_uv": 0}))
    assert not off[disabled].any()
    for name in full:
        if name != disabled:
            np.testing.assert_array_equal(full[name], off[name])


@pytest.mark.parametrize("seed", range(5))
def test_alpha_peak_is_in_8_to_12_hz(seed):
    x = generate_recording(only(alpha_uv=10, seed=seed))
    for ch in x:
        f, p = psd(ch)
        assert 8 <= f[np.argmax(p)] <= 12


@pytest.mark.parametrize("seed", range(5))
def test_line_noise_peak_is_exactly_50_hz(seed):
    x = generate_recording(only(line_noise_uv=5, seed=seed))
    for ch in x:
        f, p = psd(ch)
        assert f[np.argmax(p)] == pytest.approx(50.0)


@pytest.mark.parametrize("seed", range(5))
def test_drift_energy_is_below_1_hz(seed):
    # 60 s so that the slowest components (0.05 Hz) span several cycles.
    x = generate_recording(only(drift_uv=30, duration_s=60, seed=seed))
    for ch in x:
        f, p = psd(ch)
        assert p[f < 1].sum() / p.sum() > 0.95


@pytest.mark.parametrize("seed", range(5))
def test_brain_noise_std_is_exact(seed):
    x = generate_recording(only(brain_noise_uv=8, seed=seed))
    np.testing.assert_allclose(x.std(axis=1), 8 * UV, rtol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_brain_noise_welch_slope_is_about_minus_one(seed):
    x = generate_recording(only(brain_noise_uv=8, duration_s=120, seed=seed))
    slopes = []
    for ch in x:
        f, p = welch(ch)
        m = (f >= 1) & (f <= 40)
        slopes.append(np.polyfit(np.log10(f[m]), np.log10(p[m]), 1)[0])
    assert np.mean(slopes) == pytest.approx(-1.0, abs=0.3)


def test_brain_noise_low_vs_high_band_power():
    x = generate_recording(only(brain_noise_uv=8, seed=3))
    f, p = psd(x[0])
    low = p[(f >= 1) & (f < 4)].mean()
    high = p[(f >= 30) & (f < 40)].mean()
    assert low > 5 * high


# ---------------- amplitude semantics ----------------
def test_drift_uv_is_exact_peak():
    x = generate_recording(only(drift_uv=30, seed=3))
    np.testing.assert_allclose(np.abs(x).max(axis=1), 30 * UV, rtol=1e-9)


def test_line_noise_uv_is_nominal_amplitude_with_gain_range():
    x = generate_recording(only(line_noise_uv=5, seed=3))
    peaks = np.abs(x).max(axis=1)
    assert (peaks <= 5 * 1.3 * UV + 1e-9).all()
    assert (peaks >= 5 * 0.7 * 0.8 * UV).all()  # 0.8: coarse sampling of a 50 Hz sine


def test_alpha_uv_is_nominal_scale_with_bounded_gain():
    x = generate_recording(only(alpha_uv=10, seed=3))
    assert np.abs(x).max() <= 10 * 1.5 * 1.3 * UV + 1e-9


def test_amplitudes_are_in_realistic_range():
    x = generate_recording(GeneratorConfig(seed=0))
    assert np.abs(x).max() < 150 * UV  # 150 microvolts expressed in canonical volts


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve())]))
