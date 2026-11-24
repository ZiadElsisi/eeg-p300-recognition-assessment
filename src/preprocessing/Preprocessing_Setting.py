
EVENT_CHANNEL = "Target stim"

EVENT_ID = {
    "NonTarget": 1,
    "Target": 2,
}

# Project-required band-pass Filter .
L_FREQ = 0.1
H_FREQ = 30.0

# The public BNCI2014-008 distribution is already documented as having a
# 50 Hz notch filter. We therefore keep the notch OFF by default to avoid applying the same notch twice to an already filtered public signal.

APPLY_NOTCH = False
NOTCH_FREQ = 50.0

# Absolute-amplitude artifact rule:
# an epoch is rejected if ANY EEG channel goes above +ARTIFACT_THRESHOLD_UV
# or below -ARTIFACT_THRESHOLD_UV at ANY sample inside the epoch
# (checked after filtering and baseline correction).
# NOTE: this is NOT MNE's peak-to-peak "reject" criterion, so it is
# implemented manually in _reject_by_absolute_amplitude().
# IMPORTANT:
# 70 uV is a configurable project value used in this implementation.
# It is NOT a numeric requirement stated in the supplied execution guide.
ARTIFACT_THRESHOLD_UV = 70.0 ##

# Label written to epochs.drop_log for amplitude-rejected epochs.
AMPLITUDE_DROP_REASON = "AMPLITUDE"

# Requested project epoch window.
TMIN = -0.2
TMAX = 0.8


EEG_CHANNELS = [
    "Fz",
    "Cz",
    "Pz",
    "Oz",
    "P3",
    "P4",
    "PO7",
    "PO8"]