"""STEP 3 - Chaos engine: Higuchi fractal dimension.  Look at the picture to see WHY the slope is the answer."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omnineuro import engines
from omnineuro.data import make_synthetic

t = np.arange(512) / 256
signals = {"smooth sine": np.sin(2*np.pi*5*t),
           "random noise": np.random.default_rng(0).normal(size=512)}
data, fs = make_synthetic();  filt = engines.bandpass(data, fs, 1, 40)
signals["EEG, normal"] = filt[0, 100*256:100*256+512]
signals["EEG, seizure"] = filt[0, 520*256:520*256+512]
for name, x in signals.items():
    print(f"{name:14s} HFD = {engines.higuchi_fd(x):.2f}")      # ~1 = smooth, ~2 = very irregular

# the log-log line behind the number, for noise vs sine
fig, ax = plt.subplots(figsize=(5, 4))
for name in ("smooth sine", "random noise"):
    x = signals[name];  ks = np.arange(1, 11);  L = []
    for k in ks:
        L.append(np.mean([np.abs(np.diff(x[m::k])).sum() * (len(x)-1) / ((len(x[m::k])-1) * k) / k for m in range(k)]))
    ax.plot(np.log(1/ks), np.log(L), "o-", label=name)
ax.set_xlabel("ln(1/k)"); ax.set_ylabel("ln L(k)"); ax.legend(); ax.set_title("slope = fractal dimension")
plt.tight_layout();  plt.savefig("outputs/steps/step3.png", dpi=100)
