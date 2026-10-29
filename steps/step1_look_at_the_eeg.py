"""STEP 1 - What does the data look like?   python steps/step1_look_at_the_eeg.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omnineuro.data import make_synthetic

data, fs = make_synthetic()           # data has 2 rows (left, right) and one column per sample
print("shape:", data.shape, " -> 2 channels x", data.shape[1], "samples;  fs =", fs, "Hz")
print("duration:", data.shape[1] / fs, "seconds")

fig, axes = plt.subplots(3, 1, figsize=(10, 6))
for ax, (a, b, title) in zip(axes, [(100, 110, "normal"), (495, 505, "seizure starts at 500 s"), (515, 525, "mid-seizure")]):
    s, e = int(a * fs), int(b * fs)
    ax.plot(data[0, s:e], label="left");  ax.plot(data[1, s:e], label="right", alpha=.7)
    ax.set_title(title, loc="left");  ax.set_ylabel("uV")
axes[0].legend()
plt.tight_layout();  plt.savefig("outputs/steps/step1.png", dpi=100)
print("saved outputs/steps/step1.png")
