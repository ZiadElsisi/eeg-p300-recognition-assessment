# Test Traceability

The suite contains the eight main unit tests (UT-01–UT-08) and only the specifically recommended automated checks: block-group leakage detection and numerical trigger-shift recovery. It does not recreate an 18-test suite.

| ID | Automated check | Test function / location |
|---|---|---|
| UT-01 | Loader returns requested subject data with expected sampling rate, channels, and event codes. | `test_ut01_loader_returns_requested_subject_metadata_and_events` |
| UT-02 | Filtering preserves expected epoch/event count when no documented rejection applies. | `test_ut02_filtering_keeps_expected_trial_count` |
| UT-03 | Epoch interval is -200 to +800 ms on MNE's discrete sample grid. | `test_ut03_epoch_window_has_expected_sample_count` |
| UT-04 | Baseline correction leaves a near-zero pre-stimulus mean. | `test_ut04_baseline_is_approximately_zero_after_correction` |
| UT-05 | Synthetic Target trials have a positive P300-window effect. | `test_ut05_synthetic_target_has_positive_p300_window_difference` |
| UT-06 | Synthetic no-P300 control does not show a comparable Target effect. | `test_ut06_synthetic_no_p300_control_has_no_target_window_effect` |
| UT-07 | Fixed-seed grouped five-fold splits and metrics reproduce; no block group overlaps train/test. | `test_ut07_fixed_seed_reproduces_grouped_splits_and_fold_metrics` |
| UT-08 | LOSO excludes the held-out subject from fitting and reports ROC-AUC, PR-AUC, and balanced accuracy. | `test_ut08_loso_excludes_held_out_subject_and_reports_imbalance_metrics` |
| Recommended timing check | Injected +4, +8, and +12 ms timing shifts are recovered numerically within 2 ms. | `test_synthetic_p300_latency_recovers_injected_shift` (3 parameter cases) |

Latest result: **11 passed** (eight main UTs plus three timing-shift parameter cases). See [`unit_test_report.md`](unit_test_report.md).
