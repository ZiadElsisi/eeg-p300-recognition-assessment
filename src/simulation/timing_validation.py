"""Numerical timing checks for synthetic P300 latency shifts."""

import numpy as np
from scipy.optimize import curve_fit


def estimate_p300_latency_ms(times, target_average, nontarget_average) -> float:
    """Fit a Gaussian to the Target-minus-NonTarget response around P300."""
    fit_window = (times >= 0.240) & (times <= 0.550)
    fit_times = times[fit_window]
    difference = target_average[fit_window] - nontarget_average[fit_window]

    def gaussian(time, offset, amplitude, latency, width):
        return offset + amplitude * np.exp(
            -0.5 * ((time - latency) / width) ** 2
        )

    parameters, _ = curve_fit(
        gaussian,
        fit_times,
        difference,
        p0=(0.0, max(float(difference.max()), 0.1), 0.350, 0.055),
        bounds=((-np.inf, 0.0, 0.250, 0.025),
                (np.inf, np.inf, 0.550, 0.120)),
        maxfev=10000,
    )
    return float(parameters[2] * 1000.0)
