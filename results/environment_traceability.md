# Environment and Code Traceability

Recorded for the verification run on 2026-10-23.

- Python: `3.11.16`
- Git commit: `71a7e85c77e0be6e5ee6befc13fc8cb12e93362a`
- Source files: recorded against this commit; documentation and the generated test report may subsequently be updated in the working tree.
- Release tag: none. `prehardware-v1.0` has not been created.
- Source snapshot SHA-256: `f9af3bfbb88896970d0c4ead288c8be1196a61364268953fe8ad65836c78a78a`
- Snapshot inputs: all files under `src/`, `tests/`, and `notebooks/` plus `README.md`, `Pipfile`, `Pipfile.lock`, and `.gitignore` (53 files; path and file contents hashed in sorted order).

## Verification results

- Automated tests: 11 passed on 2026-10-23 — the eight main UT-01–UT-08 tests and three numeric synthetic timing-shift cases. See `reports/unit_test_report.md`.
The source snapshot hash independently identifies the listed source inputs. Create a release tag only after the team approves the release.
