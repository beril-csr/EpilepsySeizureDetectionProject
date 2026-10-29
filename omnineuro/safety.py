"""
safety.py  -  the Safety-First Feedback Protocol (section 2.4 of your document).

Rule of the document: *only say something when you are sure.*
    * if a window looks like an ARTIFACT (movement, electrode problem, muscle) -> do not interpret it
    * if 0.4 < P_move < 0.6 (the ambiguity zone)                                -> "UNCLEAR", stay silent
    * otherwise                                                                  -> NORMAL or SEIZURE-LIKE

Why rules and not the autoencoder in the document?  An autoencoder trained on normal EEG flags
ANYTHING abnormal - and a seizure is abnormal.  It would veto exactly the windows we want to
detect.  Simple physical rules (too big / flat / full of muscle noise) only reject signals that
cannot be brain activity.  See docs/HONEST_LIMITS.md for how to add an autoencoder anyway.
"""
import numpy as np

NORMAL, SEIZURE, UNCLEAR, ARTIFACT = "NORMAL", "SEIZURE-LIKE", "UNCLEAR", "ARTIFACT"


def artifact_reason(window_uv, emg_ratio, cfg):
    """
    window_uv : (2, samples) filtered EEG of this window, in uV
    emg_ratio : fraction of raw power between 30-55 Hz in this window (muscle activity lives there)
    Returns "" if the window is clean, otherwise a short reason string.
    """
    if np.abs(window_uv).max() > cfg.max_abs_uv:
        return "amplitude too large (movement / electrode pop)"
    if window_uv.var(axis=1).min() < cfg.min_var_uv2:
        return "flat signal (electrode disconnected?)"
    if emg_ratio > cfg.max_emg_ratio:
        return "muscle activity (jaw / neck tension)"
    return ""


def decide_state(p_move, artifact, cfg):
    """Turn (probability, artifact flag) into one of the four feedback states."""
    if artifact:
        return ARTIFACT
    if cfg.amb_lo < p_move < cfg.amb_hi:
        return UNCLEAR
    return SEIZURE if p_move >= cfg.amb_hi else NORMAL


def emg_ratio(raw_window, fs, lo=30.0, hi=55.0):
    """
    Share of a window's power that lies between `lo` and `hi` Hz.  Muscle (EMG) noise is broadband
    and strong above 30 Hz; clean brain EEG has very little power there.
    raw_window : (channels, samples), UNFILTERED (the band-pass would remove what we want to measure).
    Returns the largest ratio over the channels (one noisy channel is enough to distrust a window).
    """
    x = raw_window - raw_window.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(x * np.hanning(x.shape[1]), axis=1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / fs)
    band = (freqs >= lo) & (freqs <= hi)
    return float(np.max(power[:, band].sum(axis=1) / np.maximum(power.sum(axis=1), 1e-12)))
