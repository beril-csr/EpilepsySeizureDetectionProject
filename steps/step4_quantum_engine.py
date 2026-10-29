"""STEP 4 - Quantum engine: raw probabilities in, smoothed P_move out.  Flicker vs lag."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omnineuro.quantum import QuantumEngine

rng = np.random.default_rng(0)
truth = np.r_[np.full(30, 0.3), np.full(40, 0.7), np.full(30, 0.3)]       # the "real" state
raw = np.clip(truth + rng.normal(0, 0.2, 100), 0, 1)                      # a noisy classifier
q = QuantumEngine(memory=0.5, use_phase=True);  out = [q.step(p, 0.3)[0] for p in raw]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(raw, color="#8a8984", label="raw (flickers)");  ax.plot(out, color="#2a78d6", lw=2, label="P_move")
ax.axhspan(.4, .6, color="#eda100", alpha=.2);  ax.legend()
plt.tight_layout();  plt.savefig("outputs/steps/step4.png", dpi=100)
crossings = lambda v: int(np.sum(np.diff(v > 0.5) != 0))
print("times the signal crosses 0.5:  raw =", crossings(raw), " P_move =", crossings(np.array(out)))
print("NOTE: P_move reacts later than raw (lag). Try memory=0.0 and memory=0.8 and watch both effects.")
