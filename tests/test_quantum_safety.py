"""Checkpoint 4-5: Bloch-sphere maths and the safety protocol."""
import numpy as np

from omnineuro import safety
from omnineuro.config import Config
from omnineuro.quantum import QuantumEngine


def test_bloch_formula_holds():
    q = QuantumEngine(memory=0.5)
    for p in [0.0, 0.2, 0.9, 1.0, 0.5]:
        pm, theta, (x, y, z) = q.step(p, 0.3)
        assert abs(pm - np.sin(theta / 2) ** 2) < 1e-9          # P = sin^2(theta/2)
        assert abs(x * x + y * y + z * z - 1) < 1e-9            # the arrow ends ON the sphere
        assert 0 <= pm <= 1


def test_steady_input_converges_to_itself():
    q = QuantumEngine(memory=0.5)
    for _ in range(50):
        pm = q.step(0.8, 1.0)[0]
    assert abs(pm - 0.8) < 1e-6          # no systematic bias when the evidence is constant


def test_inconsistent_phase_is_damped():
    steady, jumpy = QuantumEngine(0.5), QuantumEngine(0.5)
    rng = np.random.default_rng(0)
    for _ in range(60):
        a = steady.step(0.8, 0.5)[0]
        b = jumpy.step(0.8, rng.uniform(-np.pi, np.pi))[0]
    assert b < a - 0.1, (a, b)           # same evidence, random phases -> partial destructive interference


def test_ambiguity_zone_and_artifact():
    cfg = Config()
    assert safety.decide_state(0.5, False, cfg) == safety.UNCLEAR
    assert safety.decide_state(0.9, False, cfg) == safety.SEIZURE
    assert safety.decide_state(0.1, False, cfg) == safety.NORMAL
    assert safety.decide_state(0.9, True, cfg) == safety.ARTIFACT      # artifact beats everything


def test_artifact_rules():
    cfg = Config()
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 10, (2, 512))
    assert safety.artifact_reason(clean, 0.02, cfg) == ""
    assert safety.artifact_reason(clean * 100, 0.02, cfg) != ""        # huge amplitude
    assert safety.artifact_reason(np.zeros((2, 512)), 0.02, cfg) != "" # flat line
    assert safety.artifact_reason(clean, 0.6, cfg) != ""               # muscle noise


def test_emg_ratio_separates_muscle_from_brain():
    fs, t = 256, np.arange(512) / 256
    brain = np.vstack([np.sin(2 * np.pi * 8 * t)] * 2)
    muscle = np.vstack([np.sin(2 * np.pi * 42 * t) + 0.1 * np.sin(2 * np.pi * 8 * t)] * 2)
    assert safety.emg_ratio(brain, fs) < 0.05
    assert safety.emg_ratio(muscle, fs) > 0.5
