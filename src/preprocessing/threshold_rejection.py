
import logging
import mne

from autoreject import AutoReject, get_rejection_threshold


class ArtifactRejection:
    def __init__(
        self,
        method="autoreject",
        cv=5,
        random_state=42,
        n_jobs=-1,
    ):
        self.method = method
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.reject_ = None
        self.autoreject_ = None
        self.is_fitted_ = False

        self.logger = logging.getLogger(__name__)

    def fit(self, epochs):
        """Learn rejection parameters from training epochs only."""

        self.logger.info(
            "Fitting artifact rejection using %d epochs",
            len(epochs),
        )

        if self.method == "global":
            self.reject_ = get_rejection_threshold(
                epochs,
                ch_types=["eeg"],
                cv=self.cv,
                random_state=self.random_state,
            )

        elif self.method == "autoreject":
            eeg_picks = mne.pick_types(
                epochs.info,
                eeg=True,
                exclude=[],
            )

            self.autoreject_ = AutoReject(
                cv=self.cv,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                picks=eeg_picks,
            )

            self.autoreject_.fit(epochs.copy())

        else:
            raise ValueError(
                f"Unknown rejection method: {self.method}"
            )

        self.is_fitted_ = True

        self.logger.info("Artifact rejection fitting completed.")

        return self

    def transform(self, epochs):
        """Apply previously learned rejection parameters."""

        if not self.is_fitted_:
            raise RuntimeError(
                "Artifact rejection must be fitted first."
            )

        if self.method == "global":
            cleaned = epochs.copy()
            cleaned.drop_bad(
                reject=self.reject_,
                verbose=False,
            )
            reject_log = None

        else:
            cleaned, reject_log = self.autoreject_.transform(
                epochs.copy(),
                return_log=True,
            )

        self.logger.info(
            "Artifact rejection: %d -> %d epochs",
            len(epochs),
            len(cleaned),
        )

        return cleaned, reject_log

    def fit_transform(self, epochs):
        """Fit on epochs and return their cleaned version."""

        return self.fit(epochs).transform(epochs)
