"""Pre-specified Pz ERP analysis with Fz reported as exploratory context.

Run from the project root with:
    python -m src.analysis.pz_primary_analysis

The analysis uses the already-cleaned BNCI2014-008 epochs. It reports the
Pz target-minus-nontarget response per subject and as an equal-subject-weighted
grand average, alongside reference and event-grid diagnostics. Fz is marked
secondary/exploratory in the output tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EPOCHS_DIR = ROOT / "data" / "preprocessed"
DATASET_NAME = "BNCI2014-008"
RESULTS_DIR = ROOT / "results" / "ERP" / DATASET_NAME / "channel_comparison"
FIGURES_DIR = ROOT / "figures" / "ERP" / DATASET_NAME / "channel_comparison"
PRIMARY_CHANNEL = "Pz"
EXPLORATORY_CHANNEL = "Fz"
WINDOW = (0.300, 0.500)


def mean_uv(evoked: mne.Evoked, channel: str) -> float:
    cropped = evoked.copy().crop(tmin=WINDOW[0], tmax=WINDOW[1])
    return float(cropped.get_data(picks=[channel]).mean() * 1e6)


def main() -> None:
    epochs_by_subject: dict[int, mne.Epochs] = {}
    for sid in range(1, 9):
        path = EPOCHS_DIR / f"subject_{sid}_clean-epo.fif"
        if path.exists():
            epochs_by_subject[sid] = mne.read_epochs(path, preload=True, verbose=False)
    if not epochs_by_subject:
        raise FileNotFoundError(f"No clean epoch files found in {EPOCHS_DIR}")

    subject_rows = []
    reference_rows = []
    pz_target_evokeds = []
    pz_nontarget_evokeds = []
    fz_target_evokeds = []
    fz_nontarget_evokeds = []

    for sid, epochs in sorted(epochs_by_subject.items()):
        required = {PRIMARY_CHANNEL, EXPLORATORY_CHANNEL}
        if not required.issubset(epochs.ch_names):
            raise ValueError(f"Subject {sid} missing channels: {sorted(required - set(epochs.ch_names))}")
        if not {"Target", "NonTarget"}.issubset(epochs.event_id):
            raise ValueError(f"Subject {sid} is missing Target/NonTarget events.")

        target = epochs["Target"].average()
        nontarget = epochs["NonTarget"].average()
        pz_target_evokeds.append(target.copy().pick([PRIMARY_CHANNEL]))
        pz_nontarget_evokeds.append(nontarget.copy().pick([PRIMARY_CHANNEL]))
        fz_target_evokeds.append(target.copy().pick([EXPLORATORY_CHANNEL]))
        fz_nontarget_evokeds.append(nontarget.copy().pick([EXPLORATORY_CHANNEL]))

        for channel, role in ((PRIMARY_CHANNEL, "pre-specified primary"),
                              (EXPLORATORY_CHANNEL, "exploratory secondary")):
            target_mean = mean_uv(target, channel)
            nontarget_mean = mean_uv(nontarget, channel)
            subject_rows.append({
                "subject": sid,
                "channel": channel,
                "analysis_role": role,
                "target_epochs": len(epochs["Target"]),
                "nontarget_epochs": len(epochs["NonTarget"]),
                "target_mean_300_500_ms_uv": target_mean,
                "nontarget_mean_300_500_ms_uv": nontarget_mean,
                "target_minus_nontarget_uv": target_mean - nontarget_mean,
            })

        nearest_zero_idx = int(np.argmin(np.abs(epochs.times)))
        reference_rows.append({
            "subject": sid,
            "sfreq_hz": float(epochs.info["sfreq"]),
            "epoch_start_s": float(epochs.tmin),
            "epoch_end_s": float(epochs.tmax),
            "zero_sample_time_s": float(epochs.times[nearest_zero_idx]),
            "zero_sample_offset_ms": float(epochs.times[nearest_zero_idx] * 1000),
            "event_onset_sample_count": int(len(epochs.events)),
            "custom_ref_applied": str(epochs.info.get("custom_ref_applied")),
            "active_projectors": int(sum(bool(p.get("active", False)) for p in epochs.info["projs"])),
            "bad_channels": ",".join(epochs.info.get("bads", [])),
            "event_id": str(epochs.event_id),
        })

    pz_t = mne.grand_average(pz_target_evokeds, interpolate_bads=False, drop_bads=False)
    pz_n = mne.grand_average(pz_nontarget_evokeds, interpolate_bads=False, drop_bads=False)
    fz_t = mne.grand_average(fz_target_evokeds, interpolate_bads=False, drop_bads=False)
    fz_n = mne.grand_average(fz_nontarget_evokeds, interpolate_bads=False, drop_bads=False)

    grand_rows = []
    for channel, role, target, nontarget in (
        (PRIMARY_CHANNEL, "pre-specified primary", pz_t, pz_n),
        (EXPLORATORY_CHANNEL, "exploratory secondary", fz_t, fz_n),
    ):
        t_mean = mean_uv(target, channel)
        n_mean = mean_uv(nontarget, channel)
        grand_rows.append({
            "channel": channel,
            "analysis_role": role,
            "subjects": len(epochs_by_subject),
            "target_mean_300_500_ms_uv": t_mean,
            "nontarget_mean_300_500_ms_uv": n_mean,
            "target_minus_nontarget_uv": t_mean - n_mean,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    subject_df = pd.DataFrame(subject_rows)
    grand_df = pd.DataFrame(grand_rows)
    reference_df = pd.DataFrame(reference_rows)
    subject_df.to_csv(RESULTS_DIR / f"{DATASET_NAME}_per_subject_pz_primary_and_fz_exploratory.csv", index=False)
    grand_df.to_csv(RESULTS_DIR / f"{DATASET_NAME}_grand_average_pz_primary_and_fz_exploratory.csv", index=False)
    reference_df.to_csv(RESULTS_DIR / f"{DATASET_NAME}_reference_event_alignment_diagnostics.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, channel, target, nontarget, label in (
        (axes[0], PRIMARY_CHANNEL, pz_t, pz_n, "Primary Pz"),
        (axes[1], EXPLORATORY_CHANNEL, fz_t, fz_n, "Exploratory Fz"),
    ):
        ax.plot(target.times * 1000, target.data[0] * 1e6, label="Target")
        ax.plot(nontarget.times * 1000, nontarget.data[0] * 1e6, label="NonTarget")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvspan(WINDOW[0] * 1000, WINDOW[1] * 1000, alpha=0.15, color="gold")
        ax.set_title(label)
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("Time from event marker (ms)")
    fig.suptitle("BNCI2014-008 ERP: Pz pre-specified; Fz exploratory")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{DATASET_NAME}_pz_primary_fz_exploratory_grand_average.png", dpi=160)
    plt.close(fig)

    pz = subject_df[subject_df["channel"] == PRIMARY_CHANNEL]
    fz = subject_df[subject_df["channel"] == EXPLORATORY_CHANNEL]
    positive_subjects = int((pz["target_minus_nontarget_uv"] > 0).sum())
    fz_positive_subjects = int((fz["target_minus_nontarget_uv"] > 0).sum())
    custom_ref_values = set(reference_df["custom_ref_applied"])
    zero_offsets = sorted(set(reference_df["zero_sample_offset_ms"]))
    report = [
        "Pz primary ERP analysis (BNCI2014-008)",
        "Pz was designated as the pre-specified primary channel. Fz is exploratory.",
        f"Window: {WINDOW[0]*1000:.0f}-{WINDOW[1]*1000:.0f} ms; subjects: {len(epochs_by_subject)}.",
        f"Pz target-minus-nontarget difference was positive in {positive_subjects}/{len(pz)} subjects.",
        f"Exploratory Fz target-minus-nontarget difference was positive in {fz_positive_subjects}/{len(fz)} subjects.",
        f"Stored custom-reference metadata: {', '.join(sorted(custom_ref_values))}.",
        f"Stored epoch event zero-sample offsets (ms): {zero_offsets}.",
        "Reference and zero-sample metadata are in the dataset-prefixed reference-event diagnostic CSV.",
        "The zero sample confirms alignment to the stored epoch event marker; without the original raw trigger stream, it cannot independently verify acquisition-time trigger accuracy.",
        "Reference metadata records what is stored in the epoch files; it cannot reconstruct undocumented reference changes made before those files were saved.",
    ]
    (RESULTS_DIR / f"{DATASET_NAME}_analysis_notes.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print("\nGrand-average results:")
    print(grand_df.to_string(index=False))
    print(f"\nSaved results under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
