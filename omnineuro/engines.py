"""
engines.py  -  the Physics engine and the Chaos engine, plus the "semantic fusion" vote.

Your document's three engines:
    Physics  : "is the brain generating enough power?"   -> energy, and the left/right ratio Lidx
    Chaos    : "is the network actively computing?"       -> Higuchi fractal dimension (HFD)
    Quantum  : "how confident is the system?"             -> see quantum.py

Everything here works on ONE window of EEG at a time (e.g. 2 s = 512 samples),
except `robust_z` and `fuse`, which work on the whole series of windows.
"""
import numpy as np
from scipy import signal as sps


# --------------------------------------------------------------- preprocessing
def bandpass(data, fs, lo, hi, causal=False):
    """
    Keep only frequencies between lo and hi Hz.  `data` has shape (channels, samples).
    sos = "second-order sections": the numerically stable way to store a filter.
    sosfiltfilt runs the filter forwards AND backwards, so the waveform is not shifted in time
    (fine for offline analysis).  sosfilt runs forward only: slightly shifted, but usable in real time.
    """
    sos = sps.butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    run = sps.sosfilt if causal else sps.sosfiltfilt
    return run(sos, data, axis=-1)


# -------------------------------------------------------------- Physics engine
def log_energy_ratio(left, right, eps=1e-6):
    """
    Equation (1) of your document:   Lidx = ln( (Var(C4) + eps) / (Var(C3) + eps) )

    Var = variance = average squared deviation = "signal power" for a zero-mean signal.
    Ratio > 1 -> right side stronger -> ln > 0.   Ratio < 1 -> left stronger -> ln < 0.
    ln() squeezes huge ratios to manageable numbers.   eps stops 0/0 if an electrode goes flat.
    """
    return float(np.log((np.var(right) + eps) / (np.var(left) + eps)))


def log_total_energy(left, right, eps=1e-6):
    """Total energy of both channels, on a log scale.  The 'energy goes UP' half of seizure detection."""
    return float(np.log(np.var(left) + np.var(right) + eps))


# --------------------------------------------------------------- Chaos engine
def higuchi_fd(x, kmax=10):
    """
    Higuchi Fractal Dimension: how rough/complex is a signal?   ~1 = smooth line, ~2 = pure noise.

    Idea (matches your note (3)):
      1. For each step size k = 1..kmax, look at the signal with every k-th sample
         (k different "offsets" m = 0..k-1 so no sample is ignored).
      2. Add up the up-and-down distance travelled along that thinned-out curve.
         Rescale so different k are comparable.  Average over the offsets -> L(k).
      3. A rough curve gets much SHORTER when you skip samples; a smooth one barely changes.
         Plot ln L(k) against ln(1/k): the slope of that line is the fractal dimension.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    ks = np.arange(1, kmax + 1)
    lengths = np.empty(kmax)
    for i, k in enumerate(ks):
        per_offset = []
        for m in range(k):
            seg = x[m::k]                                  # samples m, m+k, m+2k, ...
            n_intervals = len(seg) - 1
            if n_intervals < 1:
                continue
            travelled = np.abs(np.diff(seg)).sum()
            per_offset.append(travelled * (n - 1) / (n_intervals * k) / k)
        lengths[i] = np.mean(per_offset)
    slope = np.polyfit(np.log(1.0 / ks), np.log(lengths + 1e-12), 1)[0]
    return float(slope)


# ------------------------------------------------- phase synchrony (for quantum)
def phase_sync(left, right):
    """
    How do the two hemispheres line up in time?  Returns (phase_lag, coherence).

    hilbert() turns each signal into a rotating arrow (its "analytic signal"); angle() gives
    the arrow's direction at every sample.  We subtract the two directions and average the
    resulting unit arrows (this is your note (1): vector summation!):
        coherence ~ 1 : arrows all point the same way -> neurons "fire in the same direction"
        coherence ~ 0 : arrows point everywhere       -> they cancel (destructive interference)
    phase_lag is the direction of the summed arrow.
    """
    diff = np.angle(sps.hilbert(left)) - np.angle(sps.hilbert(right))
    c = np.mean(np.exp(1j * diff))
    return float(np.angle(c)), float(np.abs(c))


# --------------------------------------------------------- series-level helpers
def robust_z(values, ref_mask):
    """
    "How unusual is each value compared with this patient's own baseline?"
    z = (value - median) / MAD,  where MAD = median absolute deviation (x1.4826 so it matches a
    normal curve's standard deviation).  Median/MAD instead of mean/std so that a few bad
    windows inside the baseline cannot distort the reference.
    z = 0 normal, z = +3 clearly above baseline, z = -3 clearly below.
    """
    ref = values[ref_mask]
    med = np.median(ref)
    mad = 1.4826 * np.median(np.abs(ref - med))
    mad = max(mad, 1e-6)
    return np.clip((values - med) / mad, -50, 50)


def fuse(z_energy, z_chaos, cfg):
    """
    "Semantic fusion / adaptive vote" (Fig. 2 of your document).
    The two engines vote with their z-scores (weights in cfg), the votes are averaged,
    and a sigmoid turns the average into a raw probability between 0 and 1.
        sigmoid(v) = 1 / (1 + e^-v)
    `bias` is the evidence level where probability = 0.5.
    Returns (evidence, probability).
    """
    # CHANGE (added when the project was completed): cap each vote.  Without the cap a single engine
    # with a huge z (e.g. a big alpha burst or an unflagged movement) could outvote the other one.
    z_energy = np.clip(z_energy, -cfg.z_cap, cfg.z_cap)
    z_chaos = np.clip(z_chaos, -cfg.z_cap, cfg.z_cap)
    evidence = (cfg.w_energy * z_energy + cfg.w_chaos * z_chaos) / (cfg.w_energy + cfg.w_chaos)
    prob = 1.0 / (1.0 + np.exp(-(evidence - cfg.bias) / cfg.slope))
    return evidence, prob


def rolling_z(values, bad, init_mask, ref_n, gap_n, min_n=90):
    """
    Like robust_z, but "normal" moves with the recording.
    For window i the reference is the clean windows (not `bad`) in [i - gap_n - ref_n, i - gap_n):
    the last ref_n windows, stopping gap_n windows ago so a seizure that is just starting cannot
    pollute its own reference.  While fewer than min_n clean windows exist (the start of the
    recording) the fixed initial baseline `init_mask` is used instead.
    Causal: window i only looks at the past.
    """
    values = np.asarray(values, dtype=float)
    z = np.zeros(len(values))
    init_ref = values[init_mask]
    for i in range(len(values)):
        lo, hi = max(0, i - gap_n - ref_n), i - gap_n
        ref = values[lo:hi][~bad[lo:hi]] if hi > lo else np.array([])
        if ref.size < min_n:
            ref = init_ref
        med = np.median(ref)
        mad = max(1.4826 * np.median(np.abs(ref - med)), 1e-6)
        z[i] = (values[i] - med) / mad
    return np.clip(z, -50, 50)
