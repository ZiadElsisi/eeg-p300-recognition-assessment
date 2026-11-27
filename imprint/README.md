# Imprint

Imprint is the P300 Concealed Information Test (CIT) sub-project in this repository. It uses the existing repository and Git history; it is not a duplicate of the legacy project or a nested Git repository.

## Project layout

```text
imprint/
├── configs/default.toml  Single default configuration source
├── src/imprint/
│   ├── config/           Typed settings and configuration loading
│   ├── core/             Shared Recording, event, and source contracts
│   ├── data/             Synthetic EEG and future ERP Core source
│   ├── processing/       Source-independent EEG processing
│   ├── analysis/         Participant-level measurements and decisions
│   ├── evaluation/       Study-level evaluation against separate labels
│   └── reporting/        Result and provenance output
├── tests/
└── pyproject.toml
```

These directories define ownership boundaries. Their presence does not mean that later scientific functionality is implemented.

## Install and test

Python 3.11 or newer is required. From this directory:

```sh
python -m pip install -e ".[test]"
python -m pytest
```

## Load and use configuration

`configs/default.toml` is the only default configuration file. Load it through the public API; application modules should receive the resulting object instead of independently reading TOML:

```python
from imprint.config import load_config

config = load_config()
sampling_rate = config.signal.sampling_rate
repetitions = config.session.repetitions_per_picture
epoch_start = config.epoching.epoch_start_seconds
bootstrap_iterations = config.bootstrap.iterations
```

`load_config(path=None, overrides=None)` accepts an optional TOML path and nested Python mapping overrides. For example, `load_config(overrides={"synthetic": {"duration_s": 2.0}})` makes a validated, isolated effective configuration. Save it for a run with `config.save_effective_config("results/effective.json")`; the serialized output includes derived trial totals and, when enabled in the TOML, supplied overrides.

The loader rejects missing or unknown sections and keys, malformed TOML, invalid individual values, and invalid relationships such as mismatched channel counts, baseline outside the epoch, filter cutoffs at or beyond Nyquist, and trial counts that cannot divide evenly among blocks.

## Configuration decisions

The TOML comments identify approved values, existing implementation defaults, inferred choices, and unresolved decisions. The approved experiment settings include six brain channels (`Fz`, `Cz`, `Pz`, `P3`, `P4`, `Oz`) plus two eye channels (eight acquisition channels total), 0.1–30 Hz filtering, −0.2 to 1.5 second epochs, a −0.2 to 0 second baseline, 150 µV peak-to-peak brain-channel rejection, Pz mean amplitude from 300–600 ms, Probe/Irrelevance comparison, a minimum of 10 Probe epochs, 1,000 bootstrap iterations, a 0.90 recognition threshold, and the 12-picture, 480-trial, four-block session design.

The existing synthetic generator values are retained as implementation defaults and labeled provisional; they are not presented as approved physiological parameters. The TOML leaves eye-channel/device names, filter order, bootstrap resampling method, and Fisher-test significance level unset because those decisions are not established. The two-set interpretation of the probe pictures and the approximate duration are marked inferred. No future filtering, epoching, artifact, bootstrap, recognition, or hardware algorithms are implemented by this configuration task.

## Extending configuration

Add a new setting to `configs/default.toml`, its typed dataclass in `src/imprint/config/settings.py`, and the explicit parser in `src/imprint/config/loader.py`. Add value and cross-field validation alongside the dataclass that owns the setting, then test loading, invalid values, and behavior in any existing consumer. Pass the validated top-level object through dependency injection; do not add independent defaults or file reads in consumer modules. Keep units explicit in setting names and comments, and label new values as approved, existing, inferred, or undecided.

## Current implementation boundary

All data sources use `imprint.core.Recording`, whose signal array is `(channels, samples)` in volts. Construction validates dimensions, channel metadata, sampling rate, finite values, event bounds, and excludes evaluation-only labels from metadata. It avoids copying the array and marks the handed-off array read-only; callers must not mutate its backing memory through another alias.

`imprint.core.DataSource` defines the common source boundary. Concrete adapters implement `_load()` and return a `Recording`; calling `load()` checks the returned type. The initial synthetic adapter is available as:

```python
from imprint.config import load_config
from imprint.data.synthetic import SyntheticDataSource

source = SyntheticDataSource(load_config())
recording = source.load()
```

The adapter uses central sampling, channel, duration, and seed settings. Existing synthetic background components are preserved, and same-seed generation is repeatable. Generic `EYE_HORIZONTAL` and `EYE_VERTICAL` labels identify the two planned eye roles only when exact channel names are unset; they are not approved physical input names. The adapter does not create events or participant condition truth.

A shared Recording/source contract and initial synthetic adapter now exist. Full CIT waveform generation, preprocessing, participant analysis, evaluation, report generation, ERP Core adapter, CLI, and end-to-end processing workflow remain future work.
