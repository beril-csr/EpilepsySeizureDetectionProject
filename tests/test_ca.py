"""Checkpoint 6: the memristor and the cellular automaton."""
import numpy as np

from omnineuro.ca import MemristiveCA
from omnineuro.memristor import R_OFF, R_ON, MemristorArray


def test_memristor_switches_and_remembers():
    m = MemristorArray(1)
    assert not m.read()[0] and abs(m.resistance[0] - R_OFF) < 1
    m.write([3.0]);  assert m.read()[0] and abs(m.resistance[0] - R_ON) < 1      # SET
    m.write([0.5]);  assert m.read()[0]                                          # small voltage: memory kept
    m.write([-3.0]); assert not m.read()[0]                                      # RESET


def test_cells_obey_refractory_period():
    ca = MemristiveCA(seed=3)
    hist = np.array([ca.step(1.0).copy() for _ in range(120)])
    on = hist[:-2]
    assert on.any()
    assert not (on & hist[1:-1]).any(), "a cell was ON two steps in a row"
    assert not (on & hist[2:]).any(), "a cell re-fired inside its 2-step refractory period"


def test_epilepsy_level_raises_activity_and_spreads():
    def run(s):
        ca = MemristiveCA(seed=1)
        acts = np.array([(ca.step(s), ca.activity())[1] for _ in range(250)])[50:]
        return acts.mean(axis=0)
    healthy, sick = run(0.0), run(1.0)
    assert sick[0] > 5 * healthy[0], (healthy, sick)      # pathogenic grid goes strongly active
    assert sick[1] > 3 * healthy[1], (healthy, sick)      # ...and drags the healthy grid with it
    assert healthy[0] < 0.03                              # healthy phase stays quiet


def test_same_seed_same_result():
    a, b = MemristiveCA(seed=7), MemristiveCA(seed=7)
    assert all((a.step(0.6) == b.step(0.6)).all() for _ in range(30))
