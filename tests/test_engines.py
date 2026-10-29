"""Checkpoint 1-3: do the engines measure what the document says they measure?"""
import numpy as np

from omnineuro import engines
from omnineuro.config import Config


def test_hfd_smooth_vs_noisy():
    t = np.arange(512) / 256
    smooth = engines.higuchi_fd(np.sin(2 * np.pi * 5 * t))
    noise = engines.higuchi_fd(np.random.default_rng(0).normal(size=512))
    assert smooth < 1.2, smooth          # a clean sine is a smooth curve: dimension close to 1
    assert noise > 1.8, noise            # white noise fills the plane: dimension close to 2


def test_lidx_sign_and_size():
    rng = np.random.default_rng(1)
    a = rng.normal(size=1000)
    assert abs(engines.log_energy_ratio(a, a)) < 1e-6                 # equal power -> 0
    assert abs(engines.log_energy_ratio(a, 2 * a) - np.log(4)) < 1e-3  # right 2x amplitude = 4x variance
    assert engines.log_energy_ratio(2 * a, a) < 0                      # left stronger -> negative


def test_lidx_survives_flat_electrode():
    z = np.zeros(500)
    assert np.isfinite(engines.log_energy_ratio(z, z))                 # that is what epsilon is for


def test_phase_sync_locked_vs_random():
    t = np.arange(1024) / 256
    x = np.sin(2 * np.pi * 8 * t)
    _, c_locked = engines.phase_sync(x, np.roll(x, 5))                 # same rhythm, small delay
    rng = np.random.default_rng(2)
    _, c_rand = engines.phase_sync(rng.normal(size=1024), rng.normal(size=1024))
    assert c_locked > 0.9 and c_rand < 0.3, (c_locked, c_rand)


def test_robust_z_ignores_outliers_in_baseline():
    v = np.concatenate([np.random.default_rng(3).normal(0, 1, 200), [1000.0]])
    z = engines.robust_z(v, np.ones(len(v), bool))
    assert abs(np.median(z)) < 0.2 and z[-1] > 20


def test_one_engine_alone_cannot_raise_an_alarm():
    cfg = Config()
    _, p = engines.fuse(np.array([500.0]), np.array([0.0]), cfg)       # huge energy, no complexity drop
    assert p[0] <= 0.5 + 1e-9, p
    _, p2 = engines.fuse(np.array([8.0]), np.array([8.0]), cfg)        # both agree
    assert p2[0] > 0.9, p2
