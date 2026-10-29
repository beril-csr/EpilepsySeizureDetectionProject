"""Checkpoint 7: loading data and the whole pipeline."""
import os
import struct
import tempfile

import numpy as np

from omnineuro import Config, analyse, evaluate
from omnineuro import data as D
from omnineuro import safety


def _write_edf(path, signals, fs, labels, unit="uV"):
    """Write a tiny EDF file (digital range +-32767 mapped to +-3276.7 uV) to test read_edf()."""
    ns, n = len(signals), signals.shape[1]
    n_rec = n // fs
    pad = lambda s, w: str(s).ljust(w)[:w].encode()
    head = pad("0", 8) + pad("p", 80) + pad("r", 80) + pad("01.01.20", 8) + pad("00.00.00", 8)
    head += pad(256 + 256 * ns, 8) + pad("", 44) + pad(n_rec, 8) + pad(1, 8) + pad(ns, 4)
    for field, w in [(labels, 16), (["x"] * ns, 80), ([unit] * ns, 8), (["-3276.7"] * ns, 8), (["3276.7"] * ns, 8),
                     (["-32767"] * ns, 8), (["32767"] * ns, 8), (["HP"] * ns, 80), ([fs] * ns, 8), ([""] * ns, 32)]:
        head += b"".join(pad(x, w) for x in field)
    dig = np.round(signals * 10).astype("<i2")
    body = b"".join(dig[k, r * fs:(r + 1) * fs].tobytes() for r in range(n_rec) for k in range(ns))
    open(path, "wb").write(head + body)


def test_read_edf_round_trip():
    rng = np.random.default_rng(0)
    sig = np.round(rng.normal(0, 50, (3, 256 * 4)), 1)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.edf")
        _write_edf(p, sig, 256, ["F3-C3", "T8-P8", "T8-P8"])
        names, out, fs = D.read_edf(p)
        assert fs == 256 and names == ["F3-C3", "T8-P8-0", "T8-P8-1"]
        assert np.allclose(out[0], sig[0], atol=0.06)
        x, fs2, ch = D.load_chbmit(p, left="F3-C3", right="T8-P8")      # duplicate-name matching
        assert x.shape == (2, 1024) and ch[1] == "T8-P8-0"


def test_parse_summary():
    text = ("File Name: chb01_01.edf\nNumber of Seizures in File: 0\n\n"
            "File Name: chb01_03.edf\nNumber of Seizures in File: 2\n"
            "Seizure 1 Start Time: 2996 seconds\nSeizure 1 End Time: 3036 seconds\n"
            "Seizure 2 Start Time: 3100 seconds\nSeizure 2 End Time: 3150 seconds\n")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.txt"); open(p, "w").write(text)
        assert D.parse_summary(p, "chb01_01.edf") == []
        assert D.parse_summary(p, "chb01_03.edf") == [(2996.0, 3036.0), (3100.0, 3150.0)]


def test_synthetic_pipeline_end_to_end():
    x, fs = D.make_synthetic(seed=0)
    res = analyse(x, fs, Config())
    m = evaluate(res, [(500.0, 545.0)])
    assert m["sensitivity"] == 1.0 and m["false_alarms"] == 0
    assert 0 < m["latency_s"][0] < 10
    assert m["auc_p_move"] > 0.95
    # the movement artifact (200-203 s) is rejected, the muscle burst (350 s) never raises an alarm
    assert (res.state[100:102] == safety.ARTIFACT).any()
    assert not (res.state[174:178] == safety.SEIZURE).any()
    # the CA reacts: both grids busier during the seizure than before it
    p0, p1 = m["ca_patho_pre_during"]; h0, h1 = m["ca_healthy_pre_during"]
    assert p1 > 10 * p0 and h1 > 3 * h0, m


def test_baseline_needs_clean_data():
    x, fs = D.make_synthetic()
    try:
        analyse(x[:, :int(fs * 10)], fs, Config(baseline_s=300))
    except ValueError:
        return
    raise AssertionError("expected a clear error for a too-short recording")
