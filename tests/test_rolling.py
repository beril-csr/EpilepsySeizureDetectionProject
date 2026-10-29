"""Rolling baseline: absorbs slow plateau shifts, still catches a real seizure."""
import numpy as np

from omnineuro import Config, analyse, engines, evaluate
from omnineuro.data import make_synthetic


def test_rolling_z_absorbs_a_plateau_but_not_a_short_burst():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, 1200)
    v[400:] += 6.0                       # a lasting shift (like the drift seen in a real recording)
    v[200:220] += 8.0                    # a short burst (like a 40 s seizure at 2 s windows)
    bad = np.zeros(1200, bool); init = np.zeros(1200, bool); init[:100] = True
    z = engines.rolling_z(v, bad, init, ref_n=300, gap_n=15)
    assert z[205:215].mean() > 5, "a short burst must still stand out"
    assert abs(z[1000:1200].mean()) < 1.0, "a lasting shift must be absorbed after a while"
    fixed = engines.robust_z(v, init)
    assert fixed[1000:1200].mean() > 5, "(the fixed baseline never forgets the shift)"


def test_rolling_pipeline_still_detects_the_seizure():
    x, fs = make_synthetic(seed=0)
    m = evaluate(analyse(x, fs, Config(rolling_baseline=True), run_ca=False), [(500.0, 545.0)])
    assert m["sensitivity"] == 1.0 and m["latency_s"][0] < 10 and m["false_alarms"] == 0


def test_rolling_reduces_false_alarm_minutes_on_a_long_plateau():
    def shifted(seed):
        x, fs = make_synthetic(seed=seed, dur=2600)
        a, b = int(800 * fs), int(2400 * fs)
        env = np.minimum(np.arange(b - a) / (120 * fs), 1.0)
        x[:, a:b] += 15 * env * np.sin(2 * np.pi * 8 * np.arange(a, b) / fs)
        return x, fs
    x, fs = shifted(0)
    fixed = evaluate(analyse(x, fs, Config(), run_ca=False), [(500.0, 545.0)])
    roll = evaluate(analyse(x, fs, Config(rolling_baseline=True), run_ca=False), [(500.0, 545.0)])
    assert roll["false_alarm_minutes"] < 0.5 * fixed["false_alarm_minutes"], (fixed, roll)
    assert roll["sensitivity"] == 1.0


def test_evaluate_reports_minutes_and_episodes():
    x, fs = make_synthetic(seed=0)
    m = evaluate(analyse(x, fs, Config(), run_ca=False), [(500.0, 545.0)])
    assert m["false_alarm_minutes"] == 0 and m["false_alarm_episodes"] == 0


def test_energy_only_design_works():
    x, fs = make_synthetic(seed=0)
    m = evaluate(analyse(x, fs, Config(rolling_baseline=True, w_chaos=0.0), run_ca=False), [(500.0, 545.0)])
    assert m["sensitivity"] == 1.0 and m["false_alarms"] == 0
