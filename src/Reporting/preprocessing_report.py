import json
from pathlib import Path
import mne
import numpy as np
from src.preprocessing.Preprocessing_Setting import *
from src.Utility import config
import logging

def save_report(
    report_path: str | Path,
    epochs: mne.BaseEpochs,
    counts: dict,
    event_id: dict,
    event_channel: str,
    apply_notch: bool,
    artifact_threshold_uv: float,
    sfreq: float,
    expected_samples: int,
    actual_samples: int,
    baseline_max_abs_mean_v: float,
    output_path: str | Path | None,
    report_metadata: dict | None,
   logger: logging.Logger | None = None
):
    logger = logger or logging.getLogger(__name__)

    """Build the preprocessing report dict and write it to report_path as JSON."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "status": "passed",
        "run": report_metadata or {},
        "provenance": {
            "preprocessing_source_sha256": config.sha256_file(
                Path(__file__).resolve()
            ),
            "raw_dataset_file": (report_metadata or {}).get("raw_dataset_file"),
            "raw_dataset_sha256": config.sha256_file(
                Path(report_metadata["raw_dataset_file"])
            ),
        },
        "software": {
            "mne": mne.__version__,
            "numpy": np.__version__,
        },
        "configuration": {
            "eeg_channels": EEG_CHANNELS,
            "event_channel": event_channel,
            "event_id": event_id,
            "band_pass": {
                "low_hz": L_FREQ,
                "high_hz": H_FREQ,
                "method": "IIR",
                "filter_type": "Butterworth",
                "order": 4,
                "phase": "zero",
            },
            "notch": {
                "applied": bool(apply_notch),
                "frequency_hz": NOTCH_FREQ if apply_notch else None,
            },
            "epoch_window_seconds": [TMIN, TMAX],
            "baseline_seconds": [TMIN, 0.0],
            "artifact_rule": (
                {
                    "type": "absolute_amplitude",
                    "threshold_uv": float(artifact_threshold_uv),
                    "description": (
                        "Reject if any EEG sample on any selected channel "
                        "exceeds +/- threshold after baseline correction."
                    ),
                }
                if artifact_threshold_uv is not None
                else {
                    "type": "none",
                    "description": "No amplitude-based artifact rejection was applied.",
                }
            ),
        "verification": {
            "sampling_rate_hz": sfreq,
            "expected_samples_per_epoch": expected_samples,
            "actual_samples_per_epoch": actual_samples,
            "first_sample_time_seconds": float(epochs.times[0]),
            "last_sample_time_seconds": float(epochs.times[-1]),
            "baseline_max_abs_mean_v": baseline_max_abs_mean_v,
        },
        "counts": counts,
        "saved_epochs": {
            "path": str(output_path) if output_path is not None else None,
            # NOTE: same as original. This hashes the RAW file, not the saved .fif.
            "sha256": (
                config.sha256_file(Path(report_metadata["raw_dataset_file"]))
                if output_path is not None and Path(output_path).exists()
                else None
            ),
        },
    }
    }

    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Preprocessing report saved to: %s", report_path)

