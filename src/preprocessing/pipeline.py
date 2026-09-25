
from pathlib import Path
import mne
import numpy as np
from mne.io.constants import FIFF
from src.loaders.bnci2014_008 import load_bnci2014_008

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"

# The artifact threshold is configurable because the guide says to use the
# "planned amplitude rule", but it does not define the numeric threshold.
EVENT_CHANNEL = "Target stim"

# BNCI2014-008 class codes in the MOABB loader:
#   1 -> NonTarget
#   2 -> Target
EVENT_ID = {
    "NonTarget": 1,
    "Target": 2,
}

# Project-required band-pass.
L_FREQ = 0.1
H_FREQ = 30.0

# The public BNCI2014-008 distribution is already documented as having a
# 50 Hz notch filter. We therefore keep the notch OFF by default to avoid
# applying the same notch twice to an already filtered public signal.

APPLY_NOTCH = False
NOTCH_FREQ = 50.0

# MNE's "reject" threshold is a peak-to-peak threshold.
# IMPORTANT:
# 150 uV is a configurable project value used in this implementation.
# It is NOT a numeric requirement stated in the supplied execution guide.
ARTIFACT_THRESHOLD_UV = 150.0

# Requested project epoch window.
TMIN = -0.2
TMAX = 0.8

EEG_CHANNELS = [
    "Fz",
    "Cz",
    "Pz",
    "Oz",
    "P3",
    "P4",
    "PO7",
    "PO8",
]


# def _print_event_counts(events: np.ndarray, event_id: dict) -> None:
#     """Print Target/NonTarget counts in a readable way."""
#     print("\n=== Event Verification ===")
#
#     for label, code in event_id.items():
#         count = int(np.sum(events[:, 2] == code))
#         print(f"[Record] {label}: {count}")
#
#     unexpected = sorted(set(np.unique(events[:, 2])) - set(event_id.values()))
#     if unexpected:
#         raise ValueError(
#             f"Unexpected event codes found: {unexpected}. "
#             "Check the event channel and dataset metadata."
#         )


def _verify_units(raw: mne.io.BaseRaw) -> None:
    """
    Verify the unit used by MNE for EEG channels.

    MOABB converts BNCI2014-008 source values from microvolts (uV) to
    volts (V) before creating the MNE Raw object.
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False)
    eeg_units = [raw.info["chs"][idx]["unit"] for idx in eeg_picks]

    if not eeg_units:
        raise ValueError("No EEG channels were found.")

    if any(unit != FIFF.FIFF_UNIT_V for unit in eeg_units):
        raise ValueError(
            "EEG channel units are not all volts (V). "
            "Do not apply the artifact threshold until unit handling is verified."
        )

    # print("\n=== Unit Verification ===")
    # print("[Record] MNE EEG unit: volts (V)")
    # print(
    #     f"[Record] Artifact threshold: "
    #     f"{ARTIFACT_THRESHOLD_UV:.1f} uV = "
    #     f"{ARTIFACT_THRESHOLD_UV * 1e-6:.2e} V"
    # )


def _count_artifact_rejections(
    epochs: mne.Epochs, eeg_channel_names: list[str]
) -> tuple[int, int, int]:
    """
    Separate MNE epoch drops into artifact, boundary, and other drops.

    Amplitude rejection causes the rejected EEG channel name(s) to appear
    in the drop log. Boundary-related events are reported separately.
    """
    eeg_names = set(eeg_channel_names)

    artifact_drops = 0
    boundary_drops = 0
    other_drops = 0

    for drop_reasons in epochs.drop_log:
        if not drop_reasons:
            continue

        if any(reason in eeg_names for reason in drop_reasons):
            artifact_drops += 1
        elif "TOO_SHORT" in drop_reasons:
            boundary_drops += 1
        else:
            other_drops += 1

    return artifact_drops, boundary_drops, other_drops


def _verify_baseline(epochs: mne.Epochs) -> float:
    """
    Verify that the pre-stimulus baseline is approximately zero.

    Only EEG channels are checked. Stimulus channels are event markers, not
    physiological signals, so they should never be included in the
    baseline amplitude sanity check.
    """
    baseline_mask = (epochs.times >= TMIN - 1e-12) & (
        epochs.times <= 0.0 + 1e-12
    )

    if not np.any(baseline_mask):
        raise ValueError("No samples were found in the baseline interval.")

    eeg_data = epochs.get_data(picks=EEG_CHANNELS)
    baseline_data = eeg_data[:, :, baseline_mask]

    # Shape: epochs x EEG_channels x baseline_samples
    baseline_means = baseline_data.mean(axis=2)
    max_abs_baseline = float(np.max(np.abs(baseline_means)))

    print("\n=== Baseline Verification ===")
    print(
        "[Record] Maximum absolute baseline mean across "
        f"EEG epoch/channel: {max_abs_baseline:.3e} V"
    )

    # Numerical sanity check. The exact value should be very close to zero
    # after MNE's baseline correction.
    if max_abs_baseline > 1e-10:
        raise ValueError(
            "Baseline correction check failed: residual EEG baseline "
            "mean is larger than the numerical tolerance."
        )

    return max_abs_baseline


def preprocess_eeg(
    raw: mne.io.BaseRaw,
    event_channel: str = EVENT_CHANNEL,
    event_id: dict | None = None,
    artifact_threshold_uv: float = ARTIFACT_THRESHOLD_UV,
    apply_notch: bool = APPLY_NOTCH,
    output_path: str | Path | None = None,
) -> mne.Epochs:
    """
    Build the preprocessing pipeline for one continuous EEG run.

    Pipeline:
        channel selection
            -> event extraction
            -> filtering
            -> epoching
            -> baseline correction
            -> artifact rejection
            -> save clean epochs
    """
    if event_id is None:
        event_id = EVENT_ID.copy()

    if event_channel not in raw.ch_names:
        raise ValueError(
            f'Event channel "{event_channel}" was not found. '
            f"Available channels: {raw.ch_names}"
        )

    # Work on a copy so the original dataset object remains untouched.
    raw_copy = raw.copy()

    # 1. Channel selection
    print("\n[1] Selecting EEG + event channels...")

    missing_eeg = [ch for ch in EEG_CHANNELS if ch not in raw_copy.ch_names]
    if missing_eeg:
        raise ValueError(
            "Expected EEG channels are missing: "
            f"{missing_eeg}. Do not silently substitute channels."
        )

    # We keep the event channel for event extraction, but we will remove
    # it before filtering/epoching so the final Epochs object contains EEG
    # only. This also keeps stimulus codes out of ERP amplitude checks.
    raw_copy.pick(picks=EEG_CHANNELS + [event_channel])

    print(f"[Record] EEG channels ({len(EEG_CHANNELS)}): {EEG_CHANNELS}")
    print(f"[Record] Event channel: {event_channel}")

    # ------------------------------------------------------------------
    # 2. Event extraction
    # ------------------------------------------------------------------
    print("\n[2] Extracting Target/NonTarget events...")

    # IMPORTANT:
    # "Target stim" stores the class code (1/2).
    # "Flash stim" stores which row/column was flashed and is NOT the
    # Target/NonTarget class channel.
    events = mne.find_events(
        raw_copy,
        stim_channel=event_channel,
        shortest_event=1,
        verbose=False,
    )

    # _print_event_counts(events, event_id)

    # After extracting events, keep physiological EEG only.
    raw_eeg = raw_copy.copy().pick(picks=EEG_CHANNELS)

    #  Unit verification
    _verify_units(raw_eeg)

    # Filtering
    print("\n[3] Filtering pipeline...")

    if apply_notch:
        raw_eeg.notch_filter(
            freqs=NOTCH_FREQ,
            method="fir",
            verbose=False,
        )
        print(f"[Record] 50 Hz notch: APPLIED")
    else:
        print(
            "[Record] 50 Hz notch: NOT APPLIED "
            "(dataset is already documented as 50 Hz notch-filtered)"
        )

    # The project guide requires a 0.1-30 Hz band-pass.
    # We use a 4th-order Butterworth IIR filter with zero-phase operation
    # so the filter family/order are explicit and reproducible.

    raw_filtered = raw_eeg.filter(
        l_freq=L_FREQ,
        h_freq=H_FREQ,
        method="iir",
        iir_params={
            "order": 4,
            "ftype": "butter",
        },
        phase="zero",
        verbose=False,
    )

    print("[Record] Band-pass: 0.1-30 Hz")
    print("[Record] Filter: 4th-order Butterworth IIR, zero-phase")

    # 5. Epoching + baseline correction + artifact rejection

    print(
        "\n[4] Epoching, baseline correction, "
        "and artifact rejection..."
    )

    # MNE aligns requested times to the discrete sample grid.
    # At 256 Hz, -0.2 to +0.8 s results in 257 samples when both endpoints
    # are included by MNE.
    reject_criteria = {
        "eeg": artifact_threshold_uv * 1e-6
    }

    epochs = mne.Epochs(
        raw_filtered,
        events,
        event_id=event_id,
        tmin=TMIN,
        tmax=TMAX,
        baseline=(TMIN, 0.0),
        reject=reject_criteria,
        reject_by_annotation=True,
        preload=True,
        verbose=False,
    )

    # Verification records required by the Task 3 guide
    print("\n=== Task 3 Verification Records ===")

    sfreq = float(raw_filtered.info["sfreq"])
    expected_samples = round((TMAX - TMIN) * sfreq) + 1
    actual_samples = len(epochs.times)

    print(f"[Record] Sampling rate: {sfreq:.1f} Hz")
    print(f"[Record] Requested epoch: {TMIN:.3f} to {TMAX:.3f} s")
    print(f"[Record] Expected samples/epoch: {expected_samples}")
    print(f"[Record] Actual samples/epoch: {actual_samples}")
    print(f"[Record] First sample time: {epochs.times[0]:.9f} s")
    print(f"[Record] Last sample time: {epochs.times[-1]:.9f} s")

    if actual_samples != expected_samples:
        raise ValueError("Epoch sample-count verification failed.")

    _verify_baseline(epochs)

    artifact_drops, boundary_drops, other_drops = (
        _count_artifact_rejections(epochs, EEG_CHANNELS)
    )

    total_events = len(events)
    retained_epochs = len(epochs)
    total_drops = total_events - retained_epochs

    artifact_pct = (
        100.0 * artifact_drops / total_events
        if total_events else 0.0
    )

    print(f"[Record] Input events: {total_events}")
    print(f"[Record] Retained epochs: {retained_epochs}")
    print(f"[Record] Total dropped during epoching: {total_drops}")
    print(
        f"[Record] Amplitude-rejected epochs: "
        f"{artifact_drops} ({artifact_pct:.2f}%)"
    )
    print(f"[Record] Boundary drops: {boundary_drops}")
    print(f"[Record] Other drops: {other_drops}")

    # Class balance after preprocessing is useful for Task 4 and beyond.
    print("\n=== Class Counts After Preprocessing ===")
    for label, code in event_id.items():
        count = int(np.sum(epochs.events[:, 2] == code))
        print(f"[Record] {label}: {count}")

    # 6. Save clean epochs
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        epochs.save(output_path, overwrite=True)
        print(f"\n[5] Clean epochs saved to: {output_path}")
    else:
        print(
            "\n[5] Clean epochs were not saved "
            "(output_path=None)."
        )

    print("\n preprocessing finished successfully.")
    return epochs


# def load_bnci_subject(subject_id: int = 1) -> mne.io.BaseRaw:
#     """Load one BNCI2014-008 subject/run through MOABB."""
#     dataset = BNCI2014_008()
#     sessions = dataset.get_data(subjects=[subject_id])
#
#     session_id = next(iter(sessions[subject_id]))
#     run_id = next(iter(sessions[subject_id][session_id]))
#     raw = sessions[subject_id][session_id][run_id]
#
#     # print("\n=== Dataset Sanity Check ===")
#     # print(f"[Record] Subject: {subject_id}")
#     # print(f"[Record] Session: {session_id}")
#     # print(f"[Record] Run: {run_id}")
#     # print(f"[Record] Sampling rate: {raw.info['sfreq']} Hz")
#     # print(f"[Record] Channels: {raw.ch_names}")
#
#     return raw


if __name__ == "__main__":
    print("=== Task 3: BNCI2014-008 Preprocessing Test ===")
    subject_id = 1
    dataset,raw_test_data = load_bnci2014_008(subjects=[subject_id])
    session_id = next(iter(raw_test_data[subject_id]))
    run_id = next(iter(raw_test_data[subject_id][session_id]))

    raw_test_data = raw_test_data[subject_id][session_id][run_id]

    final_epochs = preprocess_eeg(
        raw_test_data,
        output_path=Path.joinpath(PREPROCESSED_DATA_DIR,'subject_01_clean-epo.fif')
    )

    # print(f"\nFinal Epochs shape: {final_epochs.get_data().shape}")
