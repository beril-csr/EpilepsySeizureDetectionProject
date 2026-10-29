"""STEP 5 - The memristive cellular automaton on its own: healthy (s=0) to epileptiform (s=1)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omnineuro.ca import MemristiveCA

levels = [0.0, 0.5, 1.0]
fig, axes = plt.subplots(2, 3, figsize=(8, 5.5))
for j, s in enumerate(levels):
    ca = MemristiveCA(seed=1)
    acts = []
    for t in range(250):
        ca.step(s)
        if t >= 50:
            acts.append(ca.activity())          # average over time: ONE snapshot is too noisy to compare
    acts = np.mean(acts, axis=0)
    p, h = ca.grids()
    axes[0, j].imshow(p, cmap="Greys");  axes[0, j].set_title(f"s = {s}\npathogenic")
    axes[1, j].imshow(h, cmap="Greys");  axes[1, j].set_title("healthy")
    print(f"s={s}: average over time -> pathogenic {acts[0]*100:4.1f}% ON | healthy {acts[1]*100:4.1f}% ON")
for a in axes.ravel():
    a.set_xticks([]);  a.set_yticks([])
plt.tight_layout();  plt.savefig("outputs/steps/step5.png", dpi=100)
