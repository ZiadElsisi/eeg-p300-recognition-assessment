"""Source-neutral interfaces for Imprint data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .recording import Recording


class DataSource(ABC):
    """Template contract for sources that produce validated recordings.

    Concrete adapters implement ``_load``. The public ``load`` boundary checks
    the returned value so downstream code can rely on the shared contract.
    """

    def load(self) -> Recording:
        recording = self._load()
        if not isinstance(recording, Recording):
            raise TypeError(
                f"{type(self).__name__}._load() must return Recording; "
                f"got {type(recording).__name__}"
            )
        return recording

    @abstractmethod
    def _load(self) -> Recording:
        """Obtain one recording and return the canonical shared model."""
