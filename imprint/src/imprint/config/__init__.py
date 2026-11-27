"""Central configuration API for Imprint."""

from dataclasses import replace
from typing import Any

from .loader import default_config_path, load_config
from .settings import (
    ArtifactRejectionSettings,
    BlinkCorrectionSettings,
    BootstrapSettings,
    CountermeasureSettings,
    EpochSettings,
    FilteringSettings,
    ImprintConfig,
    P300Settings,
    ReproducibilitySettings,
    SessionSettings,
    SignalSettings,
    SyntheticSettings,
)


def GeneratorConfig(**overrides: Any) -> SyntheticSettings:
    """Compatibility constructor backed by the central TOML configuration.

    New code should load ``ImprintConfig`` once and inject ``config.synthetic``.
    Keyword overrides remain available for existing generator callers/tests.
    """
    settings = load_config().synthetic
    if overrides:
        settings = replace(settings, **overrides)
    settings.validate()
    return settings


__all__ = [
    "ArtifactRejectionSettings",
    "BlinkCorrectionSettings",
    "BootstrapSettings",
    "CountermeasureSettings",
    "EpochSettings",
    "FilteringSettings",
    "GeneratorConfig",
    "ImprintConfig",
    "P300Settings",
    "ReproducibilitySettings",
    "SessionSettings",
    "SignalSettings",
    "SyntheticSettings",
    "default_config_path",
    "load_config",
]
