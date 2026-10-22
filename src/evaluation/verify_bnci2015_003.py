"""Run the preprocessing pipeline on all BNCI2015-003 subjects and runs.

MOABB's BNCI2015-003 representation has separate ``Target`` and ``Flash``
stim channels. ``Target`` stores the binary class code (1 NonTarget, 2 Target),
so it is the event source required by this project's binary P300 pipeline.

Run with:
    python -m src.evaluation.verify_bnci2015_003
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import mne
from moabb.datasets import BNCI2015_003

from src.preprocessing.pipeline import preprocess_eeg


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "moabb"
RESULTS_PATH = ROOT / "results" / "bnci2015_003" / "pipeline_verification.csv"


def main() -> None:
    dataset = BNCI2015_003()
    records = []
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    mne.set_config("MNE_DATA", str(DATA_ROOT.resolve()), set_env=True)
    data = dataset.get_data()
    for subject in dataset.subject_list:
        sessions = data[subject]
        for session, runs in sessions.items():
            for run_name, raw in runs.items():
                input_events = mne.find_events(
                    raw, stim_channel="Target", shortest_event=1, verbose=False
                )
                input_event_counts = dict(
                    sorted(Counter(map(int, input_events[:, 2])).items())
                )
                run_record = {
                    "subject": subject,
                    "session": session,
                    "run": run_name,
                    "sampling_rate_hz": float(raw.info["sfreq"]),
                    "input_event_count": len(input_events),
                    "input_event_code_counts": str(input_event_counts),
                    "annotation_count": len(raw.annotations),
                }
                try:
                    epochs = preprocess_eeg(raw, event_channel="Target")
                except Exception as exc:
                    records.append({
                        **run_record,
                        "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    continue
                codes = set(map(int, epochs.events[:, 2]))
                if codes != {1, 2}:
                    status = "FAILED"
                    error = f"expected class codes {{1, 2}}, got {sorted(codes)}"
                else:
                    status = "PASSED"
                    error = ""
                records.append({
                    **run_record,
                    "sampling_rate_hz": float(epochs.info["sfreq"]),
                    "channels": ",".join(epochs.ch_names),
                    "epoch_count": len(epochs),
                    "samples_per_epoch": len(epochs.times),
                    "event_id": str(epochs.event_id),
                    "nontarget_count": int((epochs.events[:, 2] == 1).sum()),
                    "target_count": int((epochs.events[:, 2] == 2).sum()),
                    "status": status,
                    "error": error,
                })

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(RESULTS_PATH, index=False)
    passed = sum(row.get("status") == "PASSED" for row in records)
    failed = sum(row.get("status") == "FAILED" for row in records)
    print(f"Pipeline verification: {passed} passed, {failed} failed of {len(records)} recordings.")
    print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
