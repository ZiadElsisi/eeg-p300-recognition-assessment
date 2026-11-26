# Imprint 38-Task Project Progress Audit

**Audit date:** 2026-11-26  
**Git commit:** `0e61df1` (repository HEAD; working tree has pre-existing staged/unstaged changes)  
**Audit boundary:** Imprint implementation files under `imprint/`, plus the user-provided architecture specification `Imprint_System_Architecture_IMPRINT-02.md` supplied from Downloads as Task 02 evidence. The legacy project outside Imprint was excluded from evidence, test counts, and status decisions. The architecture document is not currently stored inside `imprint/`.  
**Method:** Rechecked all files under `imprint/`, inspected production code, tests, packaging metadata and documentation, and reran only the Imprint suite from `imprint/` using the existing Pipenv environment. Parsed `pyproject.toml` with Python's TOML parser. No dependencies were installed. The README and `pyproject.toml` now include the Task 01 work completed after the initial audit.

## A. Executive summary

| Status | Tasks |
|---|---:|
| DONE | 3 |
| PARTIALLY DONE | 6 |
| NOT DONE | 28 |
| NOT VERIFIED | 1 |
| **Total** | **38** |

**Strict completion:** 3 / 38 = **7.9%**.  
**Estimated implementation score:** (3 DONE + 6 × 0.5) / 38 = **15.8%**. NOT VERIFIED contributes zero under the requested formula.

The Imprint sub-project has a documented architecture baseline, a separated/installable package structure, a typed synthetic-background configuration and generator, plus meaningful tests for that generator. Tasks 01 and 02 are DONE. Task 01 is complete based on the src-layout package metadata, test discovery configuration, and setup documentation; installation itself was not run. Task 02 is DONE based on the supplied architecture specification, although its contracts remain to be implemented. The core scientific workflow is absent: no shared Recording model, source contract, config-file loader, CIT trial generator, preprocessing pipeline, participant analysis, evaluation, report generation, ERP Core adapter, CLI, or end-to-end integration. Task 07 is also DONE, narrowly for the implemented background-component model and its tested behavior; it does not establish physiological realism.

**Main architecture risks:** the declared `default.yaml` is not loaded; the architecture contracts (`core/recording.py` and `core/interfaces.py`) and most package modules are empty; dependency injection and source abstraction are not implemented; there is no code-level ground-truth boundary or analysis/evaluation boundary to verify. The architecture specification is supplied outside the project tree, and should be stored with project documentation for durable local reference.

**Main scientific risks:** the generator creates continuous background signals only. It produces no Probe/Irrelevance/Target events, participant states, event identities, blink/EOG artifacts, P300 response, or evaluation truth. Thus no recognition result can currently be produced. Eight generated channels are generic array rows and lack channel names or a defined six-brain/two-eye split.

## B. Full 38-task status table

| Task | Task name | Status | Evidence | Missing work |
|---|---|---|---|---|
| 01 | Create the new project structure | DONE | `src/imprint/` has responsibility-separated packages; tests are under `tests/`; `pyproject.toml` declares the src layout, runtime/test dependencies, package discovery, and pytest paths; README documents install and test commands. TOML parses and the Imprint suite passes with the configured source path. | Editable installation was not executed, so packaging build/install behavior was not independently exercised. |
| 02 | Define implementation architecture | DONE | User-provided `Imprint_System_Architecture_IMPRINT-02.md` defines package ownership, layered flow, Recording/Event/result contracts, source abstraction, ground-truth boundary, dependency rules, configuration flow, testing strategy, reproducibility, and hardware scope. | Store the approved specification in the Imprint project for durable project-local reference; implementation of its contracts belongs to later tasks and does not block Task 02 completion. |
| 03 | Centralize configuration | PARTIALLY DONE | `configs/default.yaml` declares broad settings; `config/settings.py` defines and validates `GeneratorConfig`. | No YAML loader, complete typed config, config validation against file, dependency injection, or effective-config serialization. Filename is YAML, not the transcript's `configs/default.toml`. |
| 04 | Implement shared Recording model | NOT DONE | `core/recording.py` is empty. | Implement validated data, channel names/types, sampling rate, units, events, and metadata contract. |
| 05 | Implement common source interface | NOT DONE | `core/interfaces.py` and `data/erp_core/__init__.py` contain no source contract or adapter. | Define interface and implement synthetic and ERP Core sources returning compatible recordings. |
| 06 | Build synthetic signal generator | PARTIALLY DONE | `data/synthetic/generator.py`: `generate_base_recording`, `generate_components`, `generate_recording`; dimensions derive from validated sampling rate and duration. | Produces only an unannotated 8-channel continuous array; no Recording instance, trial schedule, events, or participant-level source API. |
| 07 | Add realistic EEG background and noise | DONE | `components.py`: `brain_noise`, `alpha_rhythm`, `line_noise`, `slow_drift`; separate seeded streams; volts output and microvolt inputs documented. `test_synthetic_eeg.py` tests spectra, amplitude, composition, disabling, and reproducibility. | Physiological scope remains approximate: no spatial covariance/topography, channel coupling, individual variability, or source physiology. Completion is for the specified modeled components, not human-like realism. |
| 08 | Add blink and eye-artifact simulation | NOT DONE | Search of `imprint/` found no blink/EOG generator or event model. | Add configurable blink timing/amplitude, eye-channel effects, blink events, and behavioral tests. |
| 09 | Add participant conditions and event labels | NOT DONE | No participant-condition, Probe, Irrelevance, Target, trial, or block generator exists. | Implement repeatable conditions/events and validate their labels and sample positions. |
| 10 | Isolate ground truth | NOT DONE | YAML says truth output should be separate; `generate_recording` returns signal only, but no separate truth object/evaluation API exists. | Establish and enforce a separate evaluation-only truth contract; test that participant-analysis inputs cannot receive/access truth. |
| 11 | Make generation reproducible | PARTIALLY DONE | `component_rngs` uses `SeedSequence(seed).spawn(4)`; same-seed and changed-seed output tests pass. | Reproducibility currently covers background arrays only; no whole-run config, trial schedule, source identity, or persisted seed/provenance. |
| 12 | Test synthetic generator | PARTIALLY DONE | `tests/test_synthetic_eeg.py` tests shape, finiteness, parameter validation, deterministic output, component composition, component-specific spectra, and amplitude semantics. | No event/participant/ground-truth tests, no source/Recording tests, and no end-to-end generator pipeline tests. Three corresponding test modules are empty. |
| 13 | Implement EEG filtering | NOT DONE | `processing/__init__.py` is empty; no Imprint filter implementation. | Add configurable Butterworth forward-backward filtering using Recording sampling rate and behavioral tests. |
| 14 | Implement event-based epoching | NOT DONE | No epoching code exists under `imprint/`. | Extract epochs by event sample index, preserve labels/identity, handle boundaries, and test shapes. |
| 15 | Implement baseline correction | NOT DONE | No baseline processing code exists under `imprint/`. | Implement configurable within-epoch baseline subtraction and tests for interval validity and values. |
| 16 | Implement artifact rejection | NOT DONE | No Imprint artifact rejection implementation exists. | Add configurable brain-channel peak-to-peak rejection, rejected-epoch tracking, category counts, and class-bias checks. |
| 17 | Implement P300 measurement | NOT DONE | No Imprint analysis code exists; no Pz channel metadata is carried by the generator. | Measure configured Pz mean amplitude/window while preserving participant/trial identity and units. |
| 18 | Implement equal-count bootstrap | NOT DONE | No bootstrap function or tests found. | Equalize Probe/Irrelevance in every iteration, define statistic and reproducibility, and test real implementation. |
| 19 | Implement recognition decision | NOT DONE | No participant decision function exists. | Compare Probe against Irrelevance only, apply configured rule per participant, and keep Target outside primary comparison. |
| 20 | Implement inconclusive safeguards | NOT DONE | No decision-quality safeguards exist in code. | Return INCONCLUSIVE for insufficient/poor-quality evidence with explicit reasons; test against false non-recognition. |
| 21 | Implement Target attention check | NOT DONE | Target is only mentioned in YAML; no event or attention-check implementation. | Implement a separate Target usability/attention rule, independent of recognition class. |
| 22 | Implement EOG-based blink regression | NOT DONE | No EOG channels, correction, or fitting strategy exists in code. | Add validated EOG regression, protect P300, and test injected blink recovery and signal preservation. |
| 23 | Implement category-adjusted blink ratio | NOT DONE | No category-adjusted calculation exists. | Define and test the category-conditioned blink ratio with correct denominators and missing-data behavior. |
| 24 | Implement blink countermeasure check | NOT DONE | No blink countermeasure analysis exists. | Implement participant-level check and specified inconclusive/flag behavior. |
| 25 | Implement additional deception checks | NOT DONE | YAML has an empty `countermeasure_checks.rules`; no executable checks exist. | Implement each approved check and validate false-positive/coverage behavior. |
| 26 | Integrate participant analysis pipeline | NOT DONE | `analysis/__init__.py` is empty and there are no processing stages to connect. | Build source-independent per-recording pipeline with no access to truth and participant-isolated decisions. |
| 27 | Generate participant report | NOT DONE | `reporting/__init__.py` is empty; no report function exists. | Produce a participant report including decision, uncertainty, quality, and traceable configuration without exposing ground truth. |
| 28 | Implement study-level evaluation | NOT DONE | `evaluation/__init__.py` is empty; no evaluation functions exist. | Evaluate predictions against separately held truth, preserve denominators, report metrics and uncertainty. |
| 29 | Inspect ERP Core dataset structure | NOT VERIFIED | `data/erp_core/__init__.py` is a one-line package description; no dataset files or inspection artifacts are present in `imprint/`. | Dataset structure/version and required fields cannot be verified from available Imprint files. Inspect a declared ERP Core version and document a mapping. |
| 30 | Implement ERP Core source adapter | NOT DONE | ERP Core package contains no adapter code. | Implement source adapter conforming to shared Recording/source contracts and validate required channels/events. |
| 31 | Implement ERP Core evaluation transformation | NOT DONE | No transformation or proxy-label implementation exists. | Add explicit evaluation-only proxy mapping; mark outputs as proxy labels, never CIT ground truth. |
| 32 | Validate Cyton-compatible channel selection | NOT DONE | YAML lists eight names and optional EOG empty, but there is no validation against a Cyton channel map or channel type model. | Define six brain/two eye channel assumptions, validate names/order/missing-channel policy and Pz availability. |
| 33 | Run synthetic end-to-end integration tests | NOT DONE | No pipeline exists; synthetic tests call only background generator functions. | Generate full synthetic CIT recording, process and analyze through shared production pipeline, then use truth only in evaluation tests. |
| 34 | Persist configuration and code provenance | NOT DONE | YAML requests capture flags; no run persistence/report code exists. | Serialize effective config, overrides, seed, revision, source identity, and relevant software versions with results. |
| 35 | Implement unit and regression tests | PARTIALLY DONE | 72 passing test cases collected from `test_synthetic_eeg.py`; they execute production generator/config code. | Empty `test_config.py`, `test_recording.py`, `test_sources.py`; no pipeline, scientific decision, provenance, integration, or regression coverage. |
| 36 | Implement CLI | NOT DONE | `cli/__init__.py` is empty; no entry point or command exists. | Add documented config-selectable commands for generation and analysis, with clear errors and no duplicated scientific logic. |
| 37 | Write complete documentation | PARTIALLY DONE | `imprint/README.md` now describes package layout, installation, tests, architecture reference, and current implementation boundary. | Still lacks the full configuration reference, implemented scientific methodology, usage examples for a working CIT workflow, and comprehensive limitations documentation. |
| 38 | Final validation and completion report | NOT DONE | No Imprint validation report or runnable end-to-end pipeline exists; this audit is not a project completion report. | Complete executable pipeline, validation, and completion documentation after prerequisite implementation. |

## C. Detailed task-by-task review

The table above records task-specific implementation evidence and missing acceptance criteria. Detail below adds the inspected symbols/tests and execution evidence. “No test” means no relevant executable test was found under `imprint/tests/`; the three empty test modules do not count as behavioral tests.

| Task | Status | Files and symbols inspected | Tests inspected / executed | Implemented and missing; status basis |
|---|---|---|---|---|
| 01 | DONE | `pyproject.toml`; `README.md`; `src/imprint/` packages; `tests/` | Python TOML parser accepted project metadata; `pipenv run pytest -q` from `imprint/`: 72 passed | src-layout package metadata, test discovery/import paths, layout and setup docs are present. Editable install was documented but not run; functional scientific packages remain later-task work. |
| 02 | DONE | User-provided `Imprint_System_Architecture_IMPRINT-02.md`; `core/interfaces.py`, `core/recording.py` checked for implementation state | `test_recording.py`, `test_sources.py` are empty; not required to establish the architecture specification itself | The document specifies responsibilities, dependencies, contracts, data flow, ground-truth separation, source abstraction, and acceptance criteria at sufficient detail for implementation. The code contracts remain empty, which is implementation work for Tasks 03–05, not a reason to mark the architecture task incomplete. The supplied file is currently outside the project tree. |
| 03 | PARTIALLY DONE | `config/settings.py`: `GeneratorConfig.validate`, `_check_band`; `configs/default.yaml` | `test_config.py` empty; validation behavior also exercised by generator parametrized tests | Typed generator-only settings validate correctly. YAML values are not read, most typed sections absent, no schema/config loader or serialization. Requested TOML convention is not followed. |
| 04 | NOT DONE | `core/recording.py` | `test_recording.py` empty | File has no class/function; no shared representation or runtime validation. |
| 05 | NOT DONE | `core/interfaces.py`, synthetic/ERP Core packages | `test_sources.py` empty | No protocol/ABC or source implementations; `generate_recording` is a function returning ndarray, not source contract. |
| 06 | PARTIALLY DONE | `generator.py`: `generate_base_recording`, `generate_components`, `generate_recording`; `GeneratorConfig.n_samples` | Shape, rate/duration, channel count, base zeros, determinism tests passed | Public generator produces dimensioned continuous signals; does not create a Recording or events/source. |
| 07 | DONE | `components.py`: `brain_noise`, `alpha_rhythm`, `line_noise`, `slow_drift`, `component_rngs`; generator summation | Component PSD, amplitude, component-sum, component-isolation tests passed | Each requested background component is implemented, composed, unit-documented, deterministic, and tested. Model lacks inter-channel covariance/topography and remains synthetic approximation. |
| 08 | NOT DONE | Search all `imprint/src/imprint/` | No blink tests/code | No artifact generation, timing, EOG effect, event identification. |
| 09 | NOT DONE | Synthetic generator and YAML | No event tests | No conditions, events, labels, trial/block structure. |
| 10 | NOT DONE | `generate_recording`; YAML truth settings; package boundaries | No ground-truth tests | Signal-only return avoids embedding truth at present, but separate truth interface and enforced analysis/evaluation boundary do not exist; absence is not a tested isolation design. |
| 11 | PARTIALLY DONE | `component_rngs`; `GeneratorConfig.seed` | same/different seed and disabled-component independence tests passed | Seed streams are independent per component, but no reproducible full experiment exists and configuration/provenance is not saved. |
| 12 | PARTIALLY DONE | `tests/test_synthetic_eeg.py` | 72 passed | Extensive tests verify current background generator; no conditions/events, recording contracts, source interchange, or integration. |
| 13 | NOT DONE | `processing/` | No filtering test | Empty package initializer only; no filter implementation. |
| 14 | NOT DONE | `processing/` | No epoch test | No event-aware epoch extraction. |
| 15 | NOT DONE | `processing/` | No baseline test | No correction logic. |
| 16 | NOT DONE | `processing/` | No rejection test | No rejection or class-by-category audit. |
| 17 | NOT DONE | `analysis/` | No P300 test | No Pz signal extraction/measurement. |
| 18 | NOT DONE | `analysis/` | No bootstrap test | No equal-count sampling/statistic. |
| 19 | NOT DONE | `analysis/` | No decision test | No Probe vs Irrelevance decision. |
| 20 | NOT DONE | `analysis/`, YAML outcome declarations | No safeguard test | Outcome strings in configuration are declarations, not implemented logic. |
| 21 | NOT DONE | `analysis/`, YAML `attention_check` | No attention test | Settings are null and no check exists. |
| 22 | NOT DONE | `processing/`, YAML `eog_regression` | No EOG test | No EOG data/correction. P300 preservation cannot be assessed without implementation. |
| 23 | NOT DONE | `analysis/`, YAML blink settings | No ratio test | No formula or category-adjusted counts. |
| 24 | NOT DONE | `analysis/`, YAML | No countermeasure test | No blink behavior check. |
| 25 | NOT DONE | YAML empty rules; `analysis/` | No deception tests | No checks implemented. |
| 26 | NOT DONE | `analysis/`, `processing/`, `data/` | No integration test | No callable pipeline to integrate. |
| 27 | NOT DONE | `reporting/`, YAML output settings | No report test | No report generation or quality/provenance output. |
| 28 | NOT DONE | `evaluation/` | No evaluation test | No truth-aware evaluation component or metrics. |
| 29 | NOT VERIFIED | `data/erp_core/`; files under `imprint/` | No dataset structure tests/artifacts | No ERP Core resource/version is present to inspect; cannot verify actual schema. |
| 30 | NOT DONE | `data/erp_core/` | No adapter tests | No adapter implementation. |
| 31 | NOT DONE | `data/erp_core/`, `evaluation/` | No transformation tests | No proxy conversion or proxy label reporting. |
| 32 | NOT DONE | YAML `recording.channel_names`, `optional_eog_channels` | No channel tests | Names are declarative only; no Cyton mapping or channel type validation. |
| 33 | NOT DONE | `tests/test_synthetic_eeg.py`, all pipeline package dirs | No end-to-end test | Existing tests directly test generator; no shared source-to-result path. |
| 34 | NOT DONE | YAML `reproducibility`; generator | No persistence test | No effective config, Git revision, seed, or source identity saved. |
| 35 | PARTIALLY DONE | Four files in `imprint/tests/` | Only `test_synthetic_eeg.py` contains tests; 72 passed | Tests exercise production code well for generator, but three suites are empty and major modules are untested/unimplemented. |
| 36 | NOT DONE | `cli/` | No CLI tests | No commands or entry point. |
| 37 | PARTIALLY DONE | `imprint/README.md` | Not applicable | README now covers package layout, editable install, test execution, and implementation boundary. It still lacks the detailed configuration guide, scientific methodology, and runnable CIT usage workflow required for complete documentation. |
| 38 | NOT DONE | All Imprint files | Test suite ran; full pipeline not possible | No end-to-end executable pipeline or completion validation exists. |

## D. Architecture audit

| Principle | Finding |
|---|---|
| Package organization | The supplied architecture document assigns clear responsibilities to each package and diagrams their flow. `pyproject.toml` declares the `src/` package layout, package discovery, NumPy runtime dependency, pytest extra, and test import/discovery settings. README explains installation and tests. Directories exist but most are placeholders. The architecture specification itself is currently in Downloads, not stored in the project tree. |
| Dependency direction | Generator depends on typed config, which is appropriate. There are no downstream processing/analysis implementations to verify; architecture cannot be credited based on empty package structure. |
| Shared Recording contract | Absent: `core/recording.py` is empty. Generator returns ndarray with shape but no sampling-rate/channel/event metadata attached. |
| Source abstraction | Absent: `core/interfaces.py` and ERP Core adapter have no implementation. Synthetic function is not source-compatible with an ERP source. |
| Configuration architecture | YAML has comprehensive intended sections, but actual loader absent. `GeneratorConfig` independently defines some generation parameters. Null fields signal unresolved design, and no validation ties YAML to runtime. The intended `configs/default.toml` is not present; actual file is `configs/default.yaml`. |
| Ground-truth isolation | Not established. Current background function returns only signal, but there is no participant analysis or evaluation interface, nor test proving truth cannot be passed into analysis. YAML comments alone do not enforce isolation. |
| Analysis/evaluation separation | Directory names exist, but both are empty. Separation is not operational. |
| Dependency injection | Generator accepts a `GeneratorConfig`, but no application composition or injected settings/source/pipeline exists. |
| Reproducibility | Seeded independent component RNG streams and tested same-seed output work for background arrays. No full-run reproducibility or result provenance. |
| Hardware boundary | No hardware code exists in Imprint, so hardware contamination is not present; Cyton channel validation is also absent. |
| Result traceability | YAML requests configuration/revision capture, but no run or result writer exists. |

## E. Scientific correctness audit

Severity describes the scientific risk if the project were used for its stated CIT purpose.

| Check | Severity | Finding and consequence |
|---|---|---|
| Probe and Irrelevance are the primary comparison | Critical | No such event classes or comparison exist. No CIT recognition claim can currently be computed. |
| Target is an attention check only | High | Target category exists only as a YAML declaration; no attention check or primary-analysis exclusion is implemented. |
| Ground truth does not leak to participant analysis | Critical | No executable truth separation or analysis interface exists, so leakage safety is unverified. |
| Bootstrap equalizes Probe/Irrelevance counts | High | No bootstrap exists; class-count imbalance would invalidate the specified comparison if naively analyzed. |
| Artifact rejection does not create hidden class bias | High | No Imprint rejection exists, so no per-category rejection counts or bias evaluation are available. |
| Sampling rate comes from recording metadata | High | Generator uses configured sampling rate internally, but no Recording contract exists for downstream processing; propagation cannot be verified. |
| Channel identity is validated, not just count | High | Generator has only `n_channels`; YAML lists names but data carries no names and no validator checks identity. Mislabeling could assign the wrong signal to Pz/EOG. |
| Blink regression does not remove P300 | High | No blink correction exists; preservation behavior is untested. |
| Category-adjusted blink calculations are correct | High | No calculation exists. Incorrect denominator/category handling could create false countermeasure signals. |
| Poor recordings become inconclusive, not non-recognition | Critical | YAML declares a policy, but no quality logic or decision function enforces it. Bad-quality subjects could otherwise be falsely classified. |
| ERP Core proxy labels are not CIT ground truth | High | No ERP Core adapter/evaluation transformation exists. YAML comments say proxy labels should be explicit, but code does not enforce that distinction. |
| Synthetic data follows same processing/analysis code as real data | Critical | No processing/analysis pipeline exists; synthetic generator emits only background arrays. Synthetic performance cannot validate intended real-data pipeline. |
| Results trace to configuration and code version | Medium | Configuration requests provenance but nothing persists it; results would not be reproducible/auditable once pipeline is added unless addressed. |
| Six brain channels distinguished from two eye channels | High | Eight channels are generated generically; YAML has eight names but no EOG names and no channel-type validation. EOG/EEG roles are not represented in data. |

## F. Test coverage report

- **Discovered/executed Imprint test cases:** 72.
- **Passed:** 72.
- **Failed:** 0.
- **Skipped:** 0.
- **Command:** `pipenv run pytest -q` from `imprint/` (using Pipenv's existing environment; README documents the equivalent `python -m pytest` invocation).
- `test_synthetic_eeg.py` is the only test module with executable tests. It tests the actual `GeneratorConfig` and synthetic component/generator functions, including invalid settings, deterministic generation, component independence, spectral behavior, unit/amplitude semantics, and summation.
- `test_config.py`, `test_recording.py`, and `test_sources.py` are empty; they provide no tests despite their names.
- **Modules without tests:** Recording/interfaces, config file loading, source adapters, processing, analysis, evaluation, reporting, CLI, ERP Core, channel validation, provenance, and integrated pipeline. Most of these modules are also unimplemented.
- **Tests that do not test production code:** none among the 72 collected test cases; the three empty files test nothing.
- **Important regression gaps:** end-to-end synthetic generation; event and label correctness; truth isolation; class-balanced bootstrap; inconclusive outcomes; artifact rejection bias; EOG correction preserving P300; ERP Core proxy label safeguards; and persisted provenance.

## G. Dependency and integration report

- `imprint/pyproject.toml` declares NumPy as the runtime dependency and pytest as an optional test dependency. No dependency installation was performed during this audit.
- `configs/default.yaml` is currently disconnected from runtime code. `GeneratorConfig` defaults are the effective settings when callers invoke `generate_recording()` without arguments.
- The synthetic generator is the only implemented operational module group. No other production modules call it; no Imprint CLI or application entry point exists.
- Source, processing, analysis, evaluation, reporting, and CLI packages contain only initializers/placeholders. ERP Core package contains only a descriptive initializer.
- No duplicated scientific pipeline logic was found inside Imprint because the scientific pipeline is not yet implemented. There are, however, two configuration representations for the implemented synthetic generator: dataclass defaults and YAML values, with no consistency check.
- Package discovery and pytest's `src` import path are configured in `pyproject.toml`; the documented suite passes from the Imprint directory. Editable installation/build was not run, so that behavior remains unverified in execution.

## H. Recommended next steps

1. **Tasks 03–05 — Implement config loading, Recording, and source contract.** Dependency: Task 02 is complete. Load and validate the default config (including required non-null fields), define typed full-project config, implement validated Recording and source protocol. Acceptance: synthetic source returns a valid Recording; config round-trip/effective config tests pass; source contract tests run against synthetic and a stub ERP Core adapter.
2. **Tasks 08–12 — Expand synthetic source to full CIT trials and isolated truth.** Dependency: Tasks 03–05. Add participant conditions, Probe/Irrelevance/Target labels, trial/block indexing, eye channels/blinks, separate truth object, and full-run seed control. Acceptance: same config/seed reproduces signal/events/truth; changed seed changes stochastic outputs; analysis-facing object cannot access truth; six EEG/two EOG channels have explicit types.
3. **Tasks 13–18 — Build source-independent preprocessing and measurement.** Dependency: validated Recording/events and synthetic source. Add filter, epoching, baseline, artifact rejection/category accounting, Pz metric, and equal-count bootstrap. Acceptance: tests use recordings from multiple sources; sampling rate/channel names come from metadata; rejection effects by category are reported; every bootstrap round has equal Probe/Irrelevance count.
4. **Tasks 19–26 — Implement guarded participant analysis.** Dependency: Task 18 and accepted analysis rules. Implement participant-local comparison, attention check, inconclusive safeguards, blink/countermeasure checks, and pipeline orchestration. Acceptance: Target excluded from primary recognition; invalid/insufficient data yields INCONCLUSIVE; test truth is inaccessible to analysis; participant A data cannot affect participant B decision.
5. **Tasks 27–28 and 34 — Add reports, evaluation, and provenance together.** Dependency: Tasks 19–26. Implement participant report and separate study-level evaluator; save effective config, revision, seeds, source identity and versions. Acceptance: reports distinguish decisions from evaluation metrics, label ERP Core proxy outcomes clearly, and include enough provenance to reproduce a run.
6. **Tasks 29–32 — Inspect and integrate ERP Core/Cyton mappings.** Dependency: Tasks 03–05; do ERP Core dataset inspection before adapter implementation. Acceptance: documented dataset version/schema; adapter returns the shared Recording; proxy transformation is evaluation-only; channel map tests identify six EEG and two EOG channels and reject missing/mislabeled required channels.
7. **Tasks 33, 35–38 — Integration, CLI, documentation, final validation.** Dependency: complete pipeline and adapters. Acceptance: synthetic and ERP Core inputs traverse shared code where scientifically applicable; tests include failure controls and regression cases; CLI invokes library functions; docs match executable features; final report records every unresolved limitation.

## I. Final verdict

- **How much is complete?** Three of 38 tasks are DONE (7.9% strict completion); the weighted implementation estimate is 15.8% because six additional tasks are partial.
- **What runs end-to-end?** Only synthetic background-component generation runs as a complete narrow function (`generate_recording` returns a continuous NumPy array). No CIT experiment or analysis pipeline runs end-to-end.
- **Earliest incomplete dependency blocking the next phase:** Task 03. Task 02 architecture is documented; the central YAML is not loaded or validated by runtime code. Tasks 04–05 then block compatible source data and processing.
- **What should be implemented next?** Implement config loading/validation and typed configuration sections for Task 03, then build the Recording/source contracts (Tasks 04–05) before expanding synthetic trials.
- **Architecturally ready for the next task?** Yes. The supplied architecture is sufficiently specific to proceed to Task 03. Downstream processing and analysis are not ready to implement until Tasks 03–05 establish executable configuration and data/source contracts.
- **Must not be considered complete:** Imprint CIT synthetic data generation, ground-truth isolation, processing, Probe-vs-Irrelevance recognition, inconclusive handling, Target attention checks, EOG correction/countermeasure analysis, ERP Core evaluation, integrated tests, CLI, and project completion. The implemented generator is background EEG only and must not be represented as a functioning CIT pipeline.
