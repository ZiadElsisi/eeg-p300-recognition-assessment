
import numpy as np

DEFAULT_CHANNELS = (
    "Fz", "Cz", "Pz", "P3", "P4", "P7/T5", "P8/T6", "Fp1"
)
DEFAULT_NOISE_STD = 5.0
DEFAULT_P300_AMPLITUDE = 8.0
DEFAULT_P300_LATENCY_SECONDS = 0.350
DEFAULT_LATENCY_JITTER_SECONDS = 0.020
DEFAULT_AMPLITUDE_JITTER_FRACTION = 0.15
P300_GAUSSIAN_SIGMA_SECONDS = 0.050


def get_channel_weights(channels=DEFAULT_CHANNELS):
    """Return the relative P300 amplitude assigned to each channel."""
    parietal_names = {"Pz", "P3", "P4", "P7", "P8", "P7/T5", "P8/T6"}
    return {
        ch: (
            1.0 if ch == "Pz" else
            0.7 if ch in parietal_names else
            0.0 if ch == "Fp1" else
            0.25
        )
        for ch in channels
    }


def generate_synthetic_p300(
    n_target=60,
    n_nontarget=240,
    sfreq=250,
    channels=DEFAULT_CHANNELS,
    noise_std=DEFAULT_NOISE_STD,
    p300_amplitude=DEFAULT_P300_AMPLITUDE,
    p300_latency=DEFAULT_P300_LATENCY_SECONDS,
    latency_jitter=DEFAULT_LATENCY_JITTER_SECONDS,
    amplitude_jitter=DEFAULT_AMPLITUDE_JITTER_FRACTION,
    seed=42,
    trigger_shift_ms=0.0,
    include_p300=True,
):
    """Generate synthetic EEG epochs for software validation.

    Returns
    -------
    X : ndarray, shape (trials, channels, samples)
        Synthetic EEG epochs.
    y : ndarray, shape (trials,)
        1 = Target, 0 = NonTarget.
    times : ndarray
        Epoch time vector in seconds.
    """
    rng = np.random.default_rng(seed)

    tmin, tmax = -0.2, 0.8
    n_samples = int(round((tmax - tmin) * sfreq))
    times = tmin + np.arange(n_samples) / sfreq

    n_total = n_target + n_nontarget
    n_channels = len(channels)

    # Background EEG noise
    X = rng.normal(
        0.0, noise_std,
        size=(n_total, n_channels, n_samples)
    )

    y = np.concatenate([
        np.ones(n_target, dtype=int),
        np.zeros(n_nontarget, dtype=int),
    ])

    # Shuffle trial order reproducibly
    order = rng.permutation(n_total)
    X = X[order]
    y = y[order]

    # Add a P300-like positive Gaussian to target trials
    channel_weights = np.array(list(get_channel_weights(channels).values()))

    if include_p300:
        for i in np.flatnonzero(y == 1):
            latency = p300_latency + rng.normal(
                0.0, latency_jitter
            )

            amplitude = p300_amplitude * (
                    1.0 + rng.normal(0.0, amplitude_jitter)
            )

            latency += trigger_shift_ms / 1000.0

            p300 = amplitude * np.exp(
                -0.5 * ((times - latency) / P300_GAUSSIAN_SIGMA_SECONDS) ** 2
            )

            X[i] += channel_weights[:, None] * p300[None, :]
            
    return X, y, times

