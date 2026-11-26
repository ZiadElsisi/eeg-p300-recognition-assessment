# Imprint

Imprint is the P300 Concealed Information Test (CIT) sub-project in this repository. It uses the existing repository and Git history; it is not a duplicate of the legacy project or a nested Git repository. The package has its own build metadata and tests under this directory.

## Project layout

```text
imprint/
├── configs/             Default project configuration
├── src/imprint/
│   ├── analysis/         Participant-level measurements and decisions
│   ├── cli/              Command-line and application orchestration
│   ├── config/           Typed settings and configuration loading
│   ├── core/             Shared Recording, event, and source contracts
│   ├── data/
│   │   ├── erp_core/     ERP Core source adapter
│   │   └── synthetic/    Reproducible synthetic EEG source
│   ├── evaluation/       Study-level evaluation against separate labels
│   ├── processing/      Source-independent EEG processing
│   └── reporting/        Result and provenance output
├── tests/                Imprint unit, contract, and integration tests
├── pyproject.toml        Package metadata, dependencies, and test settings
└── README.md
```

These directories define ownership boundaries. Some are scaffolding for later implementation tasks; their presence does not mean their scientific functionality is implemented. The project architecture is defined by the supplied IMPRINT-02 architecture specification.

## Requirements and installation

Python 3.11 or newer is required. From this directory, install the package and its test dependency in editable mode:

```sh
python -m pip install -e ".[test]"
```

Editable installation makes changes under `src/imprint/` available without reinstalling. The package currently declares NumPy as its runtime dependency; optional libraries should be added when their corresponding implementation is introduced.

## Tests

Run the Imprint tests from this directory:

```sh
python -m pytest
```

Pytest is configured to discover only `imprint/tests/` and to import the package from `src/`. The test suite currently covers the implemented synthetic background generator; other test modules are placeholders for subsequent tasks.

## Current implementation boundary

The implemented synthetic generator creates continuous background EEG components. It does not yet generate CIT trial conditions or labels, run preprocessing or participant analysis, evaluate ground truth, or provide a CLI. The architecture specification defines the intended boundaries for those later tasks.
