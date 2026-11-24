from pathlib import Path
import sys
import json
import logging
import mne # an open-source Python library for exploring, visualizing, and analyzing human neurophysiological data
import numpy as np
from mne.io.constants import FIFF
from src.loaders.bnci2014_008 import load_bnci2014_008
from src.Utility import config
from src.preprocessing.Preprocessing_Setting import (
 EVENT_CHANNEL,
 EVENT_ID,
 EEG_CHANNELS,
 L_FREQ,
 H_FREQ,
 APPLY_NOTCH,
 NOTCH_FREQ,
 TMIN,
 TMAX,
 AMPLITUDE_DROP_REASON,
)
from src.Reporting import preprocessing_report

# ============================================================
# LOGGER CONFIGURATION
# ============================================================

# Configure a dedicated logger for this preprocessing module.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout) ## Stream Handler For Logging Inside Cosole
    console_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(console_handler)

logger.propagate = False


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = config.PROJECT_ROOT
PREPROCESSED_DATA_DIR = config.PREPROCESSED_DATA_DIR

# ============================================================
# UNIT VERIFICATION
# ============================================================

def _verify_units(raw: mne.io.BaseRaw) -> None:
    """
    Verify the unit used by MNE for EEG channels.

    MOABB converts BNCI2014-008 source values from microvolts (uV) to
    volts (V) before creating the MNE Raw object.
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False) ## return array of indices ex: [0,2,3]
    eeg_units = [raw.info["chs"][idx]["unit"] for idx in eeg_picks]## return array of units

    if not eeg_units:
        raise ValueError("No EEG channels were found.")

    for unit in eeg_units:
        if unit != FIFF.FIFF_UNIT_V:
            raise ValueError(
                "EEG channel units are not all volts (V). "
                "Do not apply the artifact threshold until unit handling is verified."
            )



# ============================================================
# Artifact Rejection
# ============================================================

def _reject_by_absolute_amplitude(epochs: mne.Epochs, threshold_uv: float) -> int:
    """Drop epochs where any EEG channel exceeds +/- threshold_uv at any sample.

    Unlike MNE's `reject` (peak-to-peak), this checks absolute amplitude.
    Run it on preloaded, baseline-corrected epochs. Returns the drop count.
    """
    if len(epochs) == 0:
        return 0

    threshold_v = threshold_uv * 1e-6  # units were verified to be volts
    data = epochs.get_data(picks=EEG_CHANNELS)  # (epochs, channels, samples)

    peak_per_epoch = np.abs(data).max(axis=(1, 2))
    bad_epoch_indices = np.flatnonzero(peak_per_epoch > threshold_v)

    if bad_epoch_indices.size > 0:
        epochs.drop(bad_epoch_indices, reason=AMPLITUDE_DROP_REASON)

    return int(bad_epoch_indices.size)


# ============================================================
# COUNT ARTIFACT REJECTIONS
# ============================================================

def _count_artifact_rejections(
    epochs: mne.Epochs
) -> tuple[int, int, int]:
    """
    Separate MNE epoch drops into artifact, boundary, and other drops.

    Amplitude rejection is recorded in the drop log with
    AMPLITUDE_DROP_REASON (custom absolute-amplitude rule). Rejected EEG
    channel names (MNE's own reject) are also counted as artifacts.
    Boundary-related events are reported separately.
    """

    artifact_drops = 0
    boundary_drops = 0
    other_drops = 0

    for drop_reasons in epochs.drop_log:
        if not drop_reasons:
            continue

        if AMPLITUDE_DROP_REASON in drop_reasons :
            artifact_drops += 1
        elif "TOO_SHORT" in drop_reasons:
            boundary_drops += 1
        else:
            other_drops += 1

    return artifact_drops, boundary_drops, other_drops


# ============================================================
# BASELINE VERIFICATION
# ============================================================

def _verify_baseline(epochs: mne.Epochs) -> float:
    """
    Verify that the pre-stimulus baseline is approximately zero.

    Only EEG channels are checked. Stimulus channels are event markers, not
    physiological signals, so they should never be included in the
    baseline amplitude sanity check.
    """
    baseline_mask = (epochs.times >= TMIN - 1e-12) & (
        epochs.times <= 0.0 + 1e-12
    ) ## the +- 1e-12 is a safe margin

    if not np.any(baseline_mask):
        raise ValueError("No samples were found in the baseline interval.")

    eeg_data = epochs.get_data(picks=EEG_CHANNELS)
    baseline_data = eeg_data[:, :, baseline_mask]

    # Shape: epochs x EEG_channels x baseline_samples
    baseline_means = baseline_data.mean(axis=2)
    max_abs_baseline = float(np.max(np.abs(baseline_means)))

    logger.info(
        "Baseline verification | Maximum absolute baseline mean: %.3e V",
        max_abs_baseline,
    )

    # Numerical sanity check. The exact value should be very close to zero
    # after MNE's baseline correction.
    if max_abs_baseline > 1e-10:
        raise ValueError(
            "Baseline correction check failed: residual EEG baseline "
            "mean is larger than the numerical tolerance."
        )

    return max_abs_baseline

# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================


def preprocess_eeg(
    raw: mne.io.BaseRaw,
    event_channel: str = EVENT_CHANNEL,
    event_id: dict | None = None,
    artifact_threshold_uv: float | None = None,
    apply_notch: bool = APPLY_NOTCH,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    report_metadata: dict | None = None,
) -> mne.BaseEpochs:
    """
    Build the preprocessing pipeline for one continuous EEG run.

    Pipeline:
        channel selection
            -> event extraction
            -> filtering
            -> epoching
            -> baseline correction
            -> optional artifact rejection
            -> verification
            -> save clean epochs and report
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


    # --------------------------------------------------------
    # 1. Channel selection
    # --------------------------------------------------------
    logger.info("Step 1/5: Selecting EEG and event channels.")

    missing_eeg = [ch for ch in EEG_CHANNELS if ch not in raw_copy.ch_names]
    if missing_eeg:
        raise ValueError(
            "Expected EEG channels are missing: "
            f"{missing_eeg}. Do not silently substitute channels."
        )

    # We keep the event channel for event extraction, but we will remove
    # it before filtering/epoching so the final Epochs object contains EEG
    # only. This also keeps stimulus codes out of ERP Analysis amplitude checks.
    raw_copy.pick(picks=EEG_CHANNELS + [event_channel])

    logger.info("Selected %d EEG channels: %s", len(EEG_CHANNELS), EEG_CHANNELS)
    logger.info("Event channel: %s", event_channel)

    # ------------------------------------------------------------------
    # 2. Event extraction
    # ------------------------------------------------------------------
    logger.info("Step 2/5: Extracting Target/NonTarget events.")

    # IMPORTANT:
    # "Target stim" stores the class code (1/2).
    # "Flash stim" stores which row/column was flashed and is NOT the Target/NonTarget class channel.
    events = mne.find_events(
        raw_copy,
        stim_channel=event_channel,
        shortest_event=1,
        verbose=False,
    )


    # After extracting events, keep physiological EEG only.
    raw_eeg = raw_copy.copy().pick(picks=EEG_CHANNELS)

    #  Unit verification
    _verify_units(raw_eeg)

    # --------------------------------------------------------
    # 3. FILTERING
    # --------------------------------------------------------

    logger.info("Step 3/5: Applying EEG filters.")

    if apply_notch:
        raw_eeg.notch_filter(
            freqs=NOTCH_FREQ,
            method="fir",
            verbose=False,
        )
        logger.info("%.1f Hz notch filter applied.", NOTCH_FREQ)
    else:
        logger.info(
            "%.1f Hz notch filter not applied; the dataset is already "
            "documented as notch-filtered.",
            NOTCH_FREQ,
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

    logger.info(
        "Band-pass filter applied: %.1f-%.1f Hz, 4th-order Butterworth IIR, "
        "zero-phase.",
        L_FREQ,
        H_FREQ,
    )


    # --------------------------------------------------------
    # 4. EPOCHING + BASELINE + artifact rejection (if given threshold)
    # -------------------------------------------------------

    logger.info(
        "Step 4/5: Epoching and baseline correction"
    )

    # MNE aligns requested times to the discrete sample grid.
    # At 256 Hz, -0.2 to +0.8 s results in 257 samples when both endpoints
    # are included by MNE.
    #
    # NOTE: no "reject" argument here. MNE's reject is peak-to-peak, while
    # this project uses an absolute-amplitude rule (> +T or < -T), applied
    # manually right after epoching (see _reject_by_absolute_amplitude).
    epochs = mne.Epochs(
        raw_filtered,
        events,
        event_id=event_id,
        tmin=TMIN,
        tmax=TMAX,
        # baseline=(TMIN, 0.0),
        reject=None,
        reject_by_annotation=True,
        preload=True,
        verbose=False,
    )
    n_amplitude_rejected = 0
    if artifact_threshold_uv is not None:
        n_amplitude_rejected = _reject_by_absolute_amplitude(
            epochs, artifact_threshold_uv
        )
        logger.info(
            "Absolute-amplitude rule (+/- %.1f uV) dropped %d epochs.",
            artifact_threshold_uv,
            n_amplitude_rejected,
        )
    else:
        logger.info("No artifact rejection applied (artifact_threshold_uv=None).")

    if len(epochs) == 0:
        raise ValueError(
            f"No epochs remain from {len(events)} events "
            f"({n_amplitude_rejected} removed by amplitude rule)."
        )



    # --------------------------------------------------------
    # VERIFICATION RECORDS
    # --------------------------------------------------------
    logger.info("Verification records:")

    sfreq = float(raw_filtered.info["sfreq"])
    expected_samples = round((TMAX - TMIN) * sfreq) + 1
    actual_samples = len(epochs.times)

    logger.info("Sampling rate: %.1f Hz", sfreq)
    logger.info("Requested epoch: %.3f to %.3f s", TMIN, TMAX)
    logger.info("Expected samples per epoch: %d", expected_samples)
    logger.info("Actual samples per epoch: %d", actual_samples)
    logger.info("First sample time: %.9f s", epochs.times[0])
    logger.info("Last sample time: %.9f s", epochs.times[-1])

    if actual_samples != expected_samples:
        raise ValueError("Epoch sample-count verification failed.")

    baseline_max_abs_mean_v = _verify_baseline(epochs)

    artifact_drops, boundary_drops, other_drops = (
        _count_artifact_rejections(epochs)
    )

    total_events = len(events)
    retained_epochs = len(epochs)
    total_drops = total_events - retained_epochs

    artifact_pct = (
        100.0 * artifact_drops / total_events
        if total_events else 0.0
    )

    logger.info("Input events: %d", total_events)
    logger.info("Retained epochs: %d", retained_epochs)
    logger.info("Total dropped epochs: %d", total_drops)
    logger.info(
        "Amplitude-rejected epochs: %d (%.2f%%)",
        artifact_drops,
        artifact_pct,
    )
    logger.info("Boundary drops: %d", boundary_drops)
    logger.info("Other drops: %d", other_drops)

    # Class balance after preprocessing is useful for Task 4 and beyond.
    logger.info("Class counts after preprocessing:")
    for label, code in event_id.items():
        count = int(np.sum(epochs.events[:, 2] == code))
        logger.info("%s: %d", label, count)



    # --------------------------------------------------------
    # 5. SAVE CLEAN EPOCHS + Reporting
    # --------------------------------------------------------

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        epochs.save(output_path, overwrite=True)
        logger.info("Step 5/5: Clean epochs saved to: %s", output_path)
    else:
        logger.info(
            "Step 5/5: Clean epochs were not saved (output_path=None)."
        )

    if report_path is not None:
        class_counts = {
            label: int(np.sum(epochs.events[:, 2] == code))
            for label, code in event_id.items()
        }
        counts = {
            "input_events": int(total_events),
            "retained_epochs": int(retained_epochs),
            "total_dropped": int(total_drops),
            "amplitude_rejected_epochs": int(artifact_drops),
            "amplitude_rejected_percent_of_input_events": float(artifact_pct),
            "boundary_drops": int(boundary_drops),
            "other_drops": int(other_drops),
            "retained_by_class": class_counts,
        }
        preprocessing_report.save_report(
            report_path, epochs, counts, event_id, event_channel,
            apply_notch, artifact_threshold_uv,sfreq,
            expected_samples, actual_samples, baseline_max_abs_mean_v,
            output_path, report_metadata,logger
        )
    logger.info("Preprocessing finished successfully.")
    return epochs


if __name__ == "__main__":
    logger.info("Starting BNCI2014-008 preprocessing test.")
    report_dir = PROJECT_ROOT / "results" / "preprocessing" / "BNCI2014-008-Dataset-Inspection"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_records = []

    for subject_id in range(1, 9):
        _, raw_data = load_bnci2014_008(subjects=[subject_id])
        ## The data we capture is the datat from one cause the dataset has one run per subject
        session_id = next(iter(raw_data[subject_id]))
        run_id = next(iter(raw_data[subject_id][session_id]))
        raw = raw_data[subject_id][session_id][run_id]

        report_path = report_dir / f"subject_{subject_id:02d}_preprocessing.json"

        raw_dataset_file = (
                PROJECT_ROOT / "data" / "moabb" / "MNE-bnci-data"
                / "~bci" / "database" / "008-2014" / f"A{subject_id:02d}.mat"
        )

        epochs = preprocess_eeg(
            raw,
            output_path=PREPROCESSED_DATA_DIR / f"subject_{subject_id}_clean-epo.fif",
            report_path=report_path,
            report_metadata={
                "dataset": "BNCI2014-008",
                "subject_id": subject_id,
                "session_id": session_id,
                "run_id": run_id,
                "raw_dataset_file": str(raw_dataset_file),
            },
        )

        report_records.append(
            json.loads(report_path.read_text(encoding="utf-8"))
        )

        logger.info(
            "Subject %02d completed: %d retained epochs.",
            subject_id,
            len(epochs),
        )

    summary_path = report_dir / "preprocessing_summary.json"

    summary_path.write_text(
        json.dumps(report_records, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info("Preprocessing summary saved to: %s", summary_path)