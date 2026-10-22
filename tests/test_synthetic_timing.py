"""Automated numerical checks for synthetic trigger-shift scenarios."""

import numpy as np
import pytest

from src.simulation.synthetic_p300 import DEFAULT_CHANNELS, generate_synthetic_p300
from src.simulation.timing_validation import estimate_p300_latency_ms


def _estimated_pz_latency(shift_ms: float) -> float:
    data, labels, times = generate_synthetic_p300(
        seed=42,
        noise_std=5.0,
        trigger_shift_ms=shift_ms,
    )
    pz = DEFAULT_CHANNELS.index("Pz")
    target_average = data[labels == 1, pz].mean(axis=0)
    nontarget_average = data[labels == 0, pz].mean(axis=0)
    return estimate_p300_latency_ms(times, target_average, nontarget_average)


@pytest.mark.parametrize("injected_shift_ms", [4.0, 8.0, 12.0])
def test_synthetic_p300_latency_recovers_injected_shift(injected_shift_ms):
    reference_latency_ms = _estimated_pz_latency(0.0)
    shifted_latency_ms = _estimated_pz_latency(injected_shift_ms)
    detected_shift_ms = shifted_latency_ms - reference_latency_ms

    assert abs(detected_shift_ms - injected_shift_ms) <= 2.0
