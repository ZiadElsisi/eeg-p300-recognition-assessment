
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from src.simulation.synthetic_p300 import (
    DEFAULT_CHANNELS,
    DEFAULT_AMPLITUDE_JITTER_FRACTION,
    DEFAULT_LATENCY_JITTER_SECONDS,
    DEFAULT_P300_AMPLITUDE,
    DEFAULT_P300_LATENCY_SECONDS,
    P300_GAUSSIAN_SIGMA_SECONDS,
    generate_synthetic_p300,
    get_channel_weights,
)
from src.simulation.timing_validation import estimate_p300_latency_ms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "synthetic_p300"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


reference_data, reference_labels, reference_times = generate_synthetic_p300(
    channels=DEFAULT_CHANNELS,
    seed=42,
    noise_std=5.0,
    trigger_shift_ms=0.0,
)
reference_pz = DEFAULT_CHANNELS.index("Pz")
reference_latency_ms = estimate_p300_latency_ms(
    reference_times,
    reference_data[reference_labels == 1, reference_pz].mean(axis=0),
    reference_data[reference_labels == 0, reference_pz].mean(axis=0),
)

# The six required scenarios.
# Noise values are illustrative simulation choices, not project requirements.
SCENARIOS = [
    {
        "name": "clean_p300",
        "noise_std": 3.0,
        "include_p300": True,
        "shift_ms": 0.0,
    },
    {
        "name": "noisy_p300",
        "noise_std": 12.0,
        "include_p300": True,
        "shift_ms": 0.0,
    },
    {
        "name": "no_p300_control",
        "noise_std": 5.0,
        "include_p300": False,
        "shift_ms": 0.0,
    },
    {
        "name": "shift_4ms",
        "noise_std": 5.0,
        "include_p300": True,
        "shift_ms": 4.0,
    },
    {
        "name": "shift_8ms",
        "noise_std": 5.0,
        "include_p300": True,
        "shift_ms": 8.0,
    },
    {
        "name": "shift_12ms",
        "noise_std": 5.0,
        "include_p300": True,
        "shift_ms": 12.0,
    },
]

summary = []

for scenario in SCENARIOS:
    name = scenario["name"]

    X, y, times = generate_synthetic_p300(
        n_target=60,
        n_nontarget=240,
        sfreq=250,
        channels=DEFAULT_CHANNELS,
        seed=42,
        noise_std=scenario["noise_std"],
        include_p300=scenario["include_p300"],
        trigger_shift_ms=scenario["shift_ms"],
    )

    assert X.shape == (300, len(DEFAULT_CHANNELS), 250), (
        f"Unexpected data shape for {name}: {X.shape}"
    )
    assert np.sum(y == 1) == 60
    assert np.sum(y == 0) == 240

    # Locate Pz.
    pz_index = DEFAULT_CHANNELS.index("Pz")

    target_epochs = X[y == 1, pz_index, :]
    nontarget_epochs = X[y == 0, pz_index, :]

    target_average = target_epochs.mean(axis=0)
    nontarget_average = nontarget_epochs.mean(axis=0)

    detected_latency_ms = None
    detected_shift_ms = None
    if scenario["include_p300"]:
        detected_latency_ms = estimate_p300_latency_ms(
            times, target_average, nontarget_average
        )
        detected_shift_ms = detected_latency_ms - reference_latency_ms
        if scenario["shift_ms"]:
            assert abs(detected_shift_ms - scenario["shift_ms"]) <= 2.0, (
                f"{name}: detected shift {detected_shift_ms:.2f} ms, "
                f"expected {scenario['shift_ms']:.2f} ms (±2 ms)."
            )

    # Plot class-average waveforms.
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        times * 1000,
        target_average,
        label="Target",
    )
    ax.plot(
        times * 1000,
        nontarget_average,
        label="NonTarget",
    )

    ax.axvline(0, linestyle="--", label="Stimulus onset")
    ax.axvspan(300, 500, alpha=0.15, label="300–500 ms window")

    ax.set_title(f"Synthetic P300: {name}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (arbitrary units)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=150)
    plt.close(fig)

    # Record the mean response in the project's P300 analysis window.
    window = (times >= 0.300) & (times <= 0.500)

    result = {
        "scenario": name,
        "shape": list(X.shape),
        "sampling_frequency_hz": 250,
        "epoch_time_range_seconds": [-0.2, 0.8],
        "target_trials": int(np.sum(y == 1)),
        "nontarget_trials": int(np.sum(y == 0)),
        "class_labels": {"Target": 1, "NonTarget": 0},
        "noise_std": scenario["noise_std"],
        "p300_injected": scenario["include_p300"],
        "simulated_shift_ms": scenario["shift_ms"],
        "generation_parameters": {
            "background_noise": {
                "type": "zero-mean Gaussian white noise",
                "standard_deviation_arbitrary_units": scenario["noise_std"],
            },
            "p300": {
                "waveform": "positive Gaussian",
                "nominal_center_ms": DEFAULT_P300_LATENCY_SECONDS * 1000,
                "peak_amplitude_arbitrary_units": DEFAULT_P300_AMPLITUDE,
                "gaussian_sigma_ms": P300_GAUSSIAN_SIGMA_SECONDS * 1000,
                "latency_jitter": {
                    "distribution": "normal, zero mean",
                    "standard_deviation_ms": DEFAULT_LATENCY_JITTER_SECONDS * 1000,
                },
                "amplitude_jitter": {
                    "distribution": "normal, zero mean",
                    "standard_deviation_fraction": DEFAULT_AMPLITUDE_JITTER_FRACTION,
                },
                "relative_channel_weights": get_channel_weights(DEFAULT_CHANNELS),
            },
            "trigger_shift_applied_to": "target P300 latency",
        },
        "reference_latency_ms": reference_latency_ms,
        "detected_p300_latency_ms": detected_latency_ms,
        "detected_shift_ms": detected_shift_ms,
        "target_mean_300_500": float(
            target_average[window].mean()
        ),
        "nontarget_mean_300_500": float(
            nontarget_average[window].mean()
        ),
    }

    summary.append(result)

    print(f"PASS: {name}")
    print(f"  Data shape: {X.shape}")
    if detected_shift_ms is not None:
        print(
            f"  Detected P300 shift: {detected_shift_ms:.2f} ms "
            f"(injected {scenario['shift_ms']:.2f} ms)"
        )
    print(
        "  Mean amplitude difference (300–500 ms):",
        round(
            result["target_mean_300_500"]
            - result["nontarget_mean_300_500"],
            3,
        ),
    )

with open(
    OUTPUT_DIR / "simulation_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, indent=2)

print(f"\nAll six scenarios generated in: {OUTPUT_DIR}")
