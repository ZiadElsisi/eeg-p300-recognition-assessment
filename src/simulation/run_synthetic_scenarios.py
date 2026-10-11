
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from src.simulation.synthetic_p300 import generate_synthetic_p300


OUTPUT_DIR = Path("../../results/synthetic_p300")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        channels=("Fz", "Cz", "Pz", "P3", "P4"),
        seed=42,
        noise_std=scenario["noise_std"],
        include_p300=scenario["include_p300"],
        trigger_shift_ms=scenario["shift_ms"],
    )

    assert X.shape == (300, 5, 250), (
        f"Unexpected data shape for {name}: {X.shape}"
    )
    assert np.sum(y == 1) == 60
    assert np.sum(y == 0) == 240

    # Locate Pz.
    pz_index = ("Fz", "Cz", "Pz", "P3", "P4").index("Pz")

    target_epochs = X[y == 1, pz_index, :]
    nontarget_epochs = X[y == 0, pz_index, :]

    target_average = target_epochs.mean(axis=0)
    nontarget_average = nontarget_epochs.mean(axis=0)

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
        "target_trials": int(np.sum(y == 1)),
        "nontarget_trials": int(np.sum(y == 0)),
        "noise_std": scenario["noise_std"],
        "p300_injected": scenario["include_p300"],
        "simulated_shift_ms": scenario["shift_ms"],
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
