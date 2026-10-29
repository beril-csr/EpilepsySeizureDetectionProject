"""STEP 2 - Physics engine: energy and Lidx per 2-second window."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omnineuro import engines
from omnineuro.data import make_synthetic

data, fs = make_synthetic()
filt = engines.bandpass(data, fs, 1, 40)          # keep 1-40 Hz
win = int(2 * fs)                                  # 2 seconds = 512 samples
n = filt.shape[1] // win
energy, lidx = [], []
for i in range(n):                                 # a "for loop": repeat for window 0, 1, 2 ...
    left, right = filt[0, i*win:(i+1)*win], filt[1, i*win:(i+1)*win]
    energy.append(engines.log_total_energy(left, right))
    lidx.append(engines.log_energy_ratio(left, right))
t = np.arange(n) * 2
fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(t, energy);  ax[0].set_ylabel("log energy")
ax[1].plot(t, lidx);    ax[1].set_ylabel("Lidx (right/left)");  ax[1].set_xlabel("seconds")
plt.tight_layout();  plt.savefig("outputs/steps/step2.png", dpi=100)
print("seizure windows: mean Lidx =", round(np.mean(lidx[250:270]), 2), "(negative = left side stronger: the 'focus')")
