"""
Does each component of OmniNeuro actually help?  Run:  python scripts/ablation.py
Uses 8 random synthetic recordings per row.  SYNTHETIC data only: it tests the plumbing and the
claims' internal logic; it cannot prove anything about real brains.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from omnineuro import Config, analyse, evaluate
from omnineuro.data import make_synthetic

SEIZ = [(500, 545)]


def row(name, cfg, mod=None, seeds=range(8)):
    sens, lat, fa, flips, auc = [], [], [], [], []
    for sd in seeds:
        data, fs = make_synthetic(seed=sd)
        if mod:
            data = mod(data, fs, sd)
        e = evaluate(analyse(data, fs, cfg, run_ca=False), SEIZ)
        sens.append(e["sensitivity"]); lat += e["latency_s"]; fa.append(e["false_alarms"])
        flips.append(e["state_flips"]); auc.append(e["auc_p_move"])
    print(f"{name:36s} sensitivity={np.mean(sens):.2f}  latency={np.mean(lat):4.1f}s  "
          f"false alarms/recording={np.mean(fa):.2f}  state flips={np.mean(flips):5.1f}  AUC={np.nanmean(auc):.3f}")


def alpha_burst(amp):
    def f(d, fs, sd):
        d = d.copy(); a, b = int(620 * fs), int(660 * fs)
        d[:, a:b] += amp * np.sin(2 * np.pi * 10 * np.arange(a, b) / fs)
        return d
    return f


def noisy(d, fs, sd):
    return d + np.random.default_rng(100 + sd).normal(0, 25, d.shape)


def plateau_rows():
    """A lasting shift of the background (an 8 Hz rhythm appears at 800 s and stays) on a 43-minute recording."""
    print("--- slow drift: a lasting 8 Hz rhythm from 800 s on (no second seizure!)")
    for name, cfg in [("fixed 5-min baseline", Config()),
                      ("rolling baseline, 10-min reference", Config(rolling_baseline=True)),
                      ("rolling baseline, 5-min reference", Config(rolling_baseline=True, ref_s=300.0))]:
        mins, sens = [], []
        for sd in range(5):
            x, fs = make_synthetic(seed=sd, dur=2600)
            a, b = int(800 * fs), int(2400 * fs)
            env = np.minimum(np.arange(b - a) / (120 * fs), 1.0)
            x[:, a:b] += 15 * env * np.sin(2 * np.pi * 8 * np.arange(a, b) / fs)
            e = evaluate(analyse(x, fs, cfg, run_ca=False), SEIZ)
            mins.append(e["false_alarm_minutes"]); sens.append(e["sensitivity"])
        print(f"{name:36s} sensitivity={np.mean(sens):.2f}  false-alarm minutes={np.mean(mins):5.1f}")


if __name__ == "__main__":
    print("--- is the quantum engine useful?")
    row("full system", Config())
    row("without the phase term", Config(use_phase=False))
    row("without any smoothing (memory=0)", Config(memory=0.0))
    print("--- noisier recordings")
    row("full system, +25 uV noise", Config(), noisy)
    row("no phase term, +25 uV noise", Config(use_phase=False), noisy)
    row("no smoothing, +25 uV noise", Config(memory=0.0), noisy)
    print("--- known weakness: a strong 10 Hz (alpha) rhythm looks like 'energy up, chaos down'")
    row("alpha burst +10 uV", Config(), alpha_burst(10))
    row("alpha burst +20 uV", Config(), alpha_burst(20))
    plateau_rows()
