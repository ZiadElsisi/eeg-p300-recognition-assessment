# Environment and Code Traceability

Recorded for the verification run on 2026-10-22.

- Python: `3.11.16`
- Base Git commit: `7ea1d4c397dbcb1cc99a7b369695ab54bebcb83d`
- Working tree: modified; the base commit does not alone identify the code used for these results.
- Release tag: none. `prehardware-v1.0` has not been created.
- Source snapshot SHA-256: `4190a7b2d67e1ebc475520d580bc4f87059a22eafb407dcfbea98823c0ab4fc5`
- Snapshot inputs: all files under `src/`, `tests/`, and `notebooks/` plus `README.md`, `Pipfile`, `Pipfile.lock`, and `.gitignore` (53 files; path and file contents hashed in sorted order).

## Verification results

- Automated tests: 11 passed — the eight main UT-01–UT-08 tests and three numeric synthetic timing-shift cases. See `reports/unit_test_report.md`.
- BNCI2015-003 preprocessing compatibility: 16 of 20 recordings passed. Both recordings for subjects 4 and 5 in each train/test run failed because the current ±70 µV absolute-amplitude rule rejected all 900 epochs. The source event channels still contained 750 NonTarget and 150 Target events. See `results/bnci2015_003/pipeline_verification.csv`.
- BNCI2015-003 CSV SHA-256: `37ebf2fd025a54a95b9fd8454aab505b41c0b696915f46c2f43ee5f46345f924`.

The source snapshot hash identifies the working source inputs independently of Git's commit hash. Once the changes are reviewed and committed, replace the base-commit/dirty-tree entry with the resulting commit and create a release tag only if the team approves that release.
