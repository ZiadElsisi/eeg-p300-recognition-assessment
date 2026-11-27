from dataclasses import dataclass, field
from types import SimpleNamespace as NS
import numpy as np, pytest
from imprint.analysis.p300 import (P300Config, P300MeasurementError, measure_p300, to_microvolts)

# --- throwaway stand-in for the real ProcessedEpochs contract -------------
@dataclass
class FakeEpochs:
    data: np.ndarray
    times_s: np.ndarray
    channel_names: list
    sampling_rate_hz: float
    events: list
    participant_id: str = "P01"
    recording_id: str | None = "R1"
    channel_types: list | None = None
    unit: str | None = None

def make(fs=250.0, names=("Fz", "Cz", "Pz"), labels=("PROBE", "IRRELEVANCE", "TARGET"), tmin=-0.2, tmax=0.8, fill=None):
    n = int(round((tmax - tmin) * fs))
    t = tmin + np.arange(n) / fs
    data = np.zeros((len(labels), len(names), n)) if fill is None else fill(len(labels), len(names), n, t)
    ev = [NS(trial_id=f"T{i}", stimulus_id=f"S{i}", event_label=l, block_id=f"B{i%2}", sample_index=1000 + i * 500)
          for i, l in enumerate(labels)]
    return FakeEpochs(data, t, list(names), fs, ev)

CFG = P300Config("Pz", 0.300, 0.600)

def test_constant_5uv():
    e = make(); e.data[:, 2, :] = 5e-6
    r = measure_p300(e, CFG)
    assert all(m.p300_amplitude_v == pytest.approx(5e-6) for m in r.measurements)
    assert to_microvolts(r.measurements[0].p300_amplitude_v) == pytest.approx(5.0)

def test_ramp_exact_mean_and_endpoints():
    e = make(); t = e.times_s
    e.data[:] = 0; e.data[:, 2, :] = t * 1e-5          # linear in time
    sel = (t >= 0.3 - 1e-9) & (t < 0.6 - 1e-9)
    assert sel.sum() == 75                                 # 250 Hz * 0.3 s
    assert t[sel][0] == pytest.approx(0.3) and t[sel][-1] == pytest.approx(0.6 - 1/250)
    r = measure_p300(e, CFG)
    assert r.measurements[0].p300_amplitude_v == pytest.approx(e.data[0, 2, sel].mean())
    assert r.measurements[0].n_samples == 75

def test_boundary_samples_excluded():
    e = make(); t = e.times_s
    e.data[:, 2, :] = 0.0
    e.data[:, 2, np.argmin(abs(t - 0.296))] = 1.0      # just before start
    e.data[:, 2, np.argmin(abs(t - 0.600))] = 1.0      # exclusive endpoint
    assert measure_p300(e, CFG).measurements[0].p300_amplitude_v == 0.0
    e.data[:, 2, np.argmin(abs(t - 0.300))] = 75.0     # inclusive start counts
    assert measure_p300(e, CFG).measurements[0].p300_amplitude_v == pytest.approx(1.0)

@pytest.mark.parametrize("fs", [128.0, 250.0, 500.0, 1000.0])
def test_sampling_rates(fs):
    e = make(fs=fs); e.data[:, 2, :] = 3e-6
    r = measure_p300(e, CFG)
    expected = int(((e.times_s >= 0.3 - 1e-9) & (e.times_s < 0.6 - 1e-9)).sum())
    assert r.measurements[0].n_samples == expected and abs(expected - 0.3 * fs) < 1
    assert r.measurements[0].p300_amplitude_v == pytest.approx(3e-6)

def test_channel_order_irrelevant():
    a = make(); a.data[:, 2, :] = 7e-6
    b = make(names=("Pz", "Fz", "Cz")); b.data[:, 0, :] = 7e-6
    assert [m.p300_amplitude_v for m in measure_p300(a, CFG).measurements] == \
           [m.p300_amplitude_v for m in measure_p300(b, CFG).measurements]

def test_missing_and_duplicate_pz():
    with pytest.raises(P300MeasurementError, match="not found"):
        measure_p300(make(names=("Fz", "Cz", "Oz")), CFG)
    with pytest.raises(P300MeasurementError, match="unique"):
        measure_p300(make(names=("Pz", "Cz", "Pz")), CFG)

def test_channel_type_validated_when_available():
    e = make(); e.channel_types = ["eeg", "eeg", "eog"]
    with pytest.raises(P300MeasurementError, match="type"):
        measure_p300(e, CFG)

@pytest.mark.parametrize("s,en", [(0.6, 0.3), (0.3, 0.3), (-1.0, 0.6), (0.3, 5.0)])
def test_bad_windows(s, en):
    with pytest.raises(P300MeasurementError):
        measure_p300(make(), P300Config("Pz", p300_window_start_s=s, p300_window_end_s=en))

def test_empty_window_and_bad_time_vector():
    with pytest.raises(P300MeasurementError, match="No samples"):
        measure_p300(make(fs=100.0), P300Config("Pz", p300_window_start_s=0.301, p300_window_end_s=0.305))
    e = make(); e.times_s = e.times_s * 2
    with pytest.raises(P300MeasurementError, match="inconsistent"):
        measure_p300(e, CFG)
    e = make(); e.data = e.data[:, :, :-1]
    with pytest.raises(P300MeasurementError):
        measure_p300(e, CFG)
    e = make(); e.data = e.data[0]
    with pytest.raises(P300MeasurementError, match="n_epochs"):
        measure_p300(e, CFG)

def test_metadata_and_categories_preserved():
    e = make(); e.data[0, 2, :] = 1e-6; e.data[1, 2, :] = 2e-6; e.data[2, 2, :] = 3e-6
    r = measure_p300(e, CFG)
    assert [m.event_type for m in r.measurements] == ["PROBE", "IRRELEVANCE", "TARGET"]
    assert [m.p300_amplitude_v for m in r.measurements] == pytest.approx([1e-6, 2e-6, 3e-6])
    m = r.measurements[1]
    assert (m.trial_id, m.stimulus_id, m.block_id, m.event_sample_index, m.participant_id, m.recording_id) == \
           ("T1", "S1", "B1", 1500, "P01", "R1")
    assert len(r.measurements) == 3 and r.unit == "V"

def test_nan_reported_not_dropped_silently():
    e = make(); e.data[:, 2, :] = 1e-6; e.data[1, 2, 80] = np.nan  # t=0.12 outside window -> fine
    assert len(measure_p300(e, CFG).exclusions) == 0
    idx = int(np.argmin(abs(e.times_s - 0.4))); e.data[1, 2, idx] = np.nan
    r = measure_p300(e, CFG)
    assert len(r.measurements) == 2 and len(r.exclusions) == 1
    assert r.exclusions[0].trial_id == "T1" and "non-finite" in r.exclusions[0].reason
    assert all(np.isfinite(m.p300_amplitude_v) for m in r.measurements)

def test_missing_event_metadata_reported():
    e = make(); e.events[0] = NS(trial_id=None, stimulus_id="S0", event_label="PROBE")
    r = measure_p300(e, CFG)
    assert len(r.exclusions) == 1 and len(r.measurements) == 2
    with pytest.raises(P300MeasurementError):
        e.events.pop(); measure_p300(e, CFG)

def test_input_not_mutated():
    e = make(fill=lambda n, c, t, tt: np.random.default_rng(0).normal(size=(n, c, t)) * 1e-5)
    before = e.data.copy(); measure_p300(e, CFG)
    assert np.array_equal(before, e.data)


# --- fixes: centralized config, units ------------------------------------
P300_SEC = NS(channel="Pz", measurement_start_ms=300, measurement_end_ms=600, measurement="mean")
SIGNAL_SEC = NS(primary_p300_channel="Pz")

def test_from_settings_converts_ms_and_matches_approved_config():
    c = P300Config.from_settings(P300_SEC, SIGNAL_SEC)
    assert (c.p300_channel, c.p300_window_start_s, c.p300_window_end_s) == ("Pz", 0.3, 0.6)
    e = make(); e.data[:, 2, :] = 5e-6
    assert measure_p300(e, c).measurements[0].p300_amplitude_v == pytest.approx(5e-6)

def test_configured_boundaries_change_selected_samples():
    e = make()
    n_default = measure_p300(e, P300Config.from_settings(P300_SEC)).measurements[0].n_samples
    wide = NS(channel="Pz", measurement_start_ms=300, measurement_end_ms=700, measurement="mean")
    assert measure_p300(e, P300Config.from_settings(wide)).measurements[0].n_samples > n_default

def test_configured_channel_is_used():
    e = make(); e.data[:, 1, :] = 9e-6
    cz = NS(channel="Cz", measurement_start_ms=300, measurement_end_ms=600, measurement="mean")
    assert measure_p300(e, P300Config.from_settings(cz)).measurements[0].p300_amplitude_v == pytest.approx(9e-6)

def test_signal_channel_mismatch_rejected():
    with pytest.raises(P300MeasurementError, match="disagrees"):
        P300Config.from_settings(P300_SEC, NS(primary_p300_channel="Cz"))

def test_unit_validation():
    e = make(); e.unit = "V"
    assert len(measure_p300(e, CFG).measurements) == 3
    for bad in ("uV", "mV"):
        e.unit = bad
        with pytest.raises(P300MeasurementError, match="unit"):
            measure_p300(e, CFG)
