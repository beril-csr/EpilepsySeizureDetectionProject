"""
data.py  -  getting EEG into the program.

Everything downstream expects the same thing:

    data : numpy array of shape (2, n_samples), in MICROVOLTS
           row 0 = LEFT hemisphere channel, row 1 = RIGHT hemisphere channel
    fs   : sampling rate in Hz (CHB-MIT = 256)

Three ways to get it:
    make_synthetic()  - fake EEG with a known seizure. Needs nothing. Used for demos and tests.
    load_chbmit()     - read a real CHB-MIT .edf file (uses `mne` if installed, else the built-in read_edf()).
    parse_summary()   - read seizure start/end times from chbXX-summary.txt (for EVALUATION only).
"""
import re

import numpy as np


# ---------------------------------------------------------------- synthetic EEG
def make_synthetic(fs=256, dur=900, seizure=(500, 545), emg=(350, 354), seed=0):
    """
    Fake two-channel EEG, so you can test the whole pipeline without downloading anything.

    background : "pink" noise (more power at low frequencies, like real EEG) + a 10 Hz alpha rhythm
    seizure    : a rhythmic 7 Hz + 14 Hz oscillation that ramps up, stronger on the LEFT side (the "focus")
    artifacts  : a big slow movement (200-203 s) and a burst of muscle noise (`emg` window)
    """
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    t = np.arange(n) / fs

    def pink_noise():
        freqs = np.fft.rfftfreq(n, 1 / fs)
        freqs[0] = freqs[1]                                   # avoid dividing by zero at 0 Hz
        spectrum = (rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)) / freqs
        x = np.fft.irfft(spectrum, n)
        return 15 * x / x.std()                               # scale to ~15 uV rms

    left, right = pink_noise(), pink_noise()
    alpha = 5 * np.sin(2 * np.pi * 10 * t)
    left, right = left + alpha, right + 0.8 * alpha

    # --- seizure: a rhythm that fades in and out, replacing part of the background
    s0, s1 = int(seizure[0] * fs), int(seizure[1] * fs)
    k = np.arange(s1 - s0)
    ramp = np.clip(np.minimum(k, (s1 - s0) - k) / (5 * fs), 0, 1)       # 5 s fade in / out
    rhythm = np.sin(2 * np.pi * 7 * t[s0:s1]) + 0.4 * np.sin(2 * np.pi * 14 * t[s0:s1])
    left[s0:s1] = left[s0:s1] * (1 - 0.6 * ramp) + 90 * ramp * rhythm
    right[s0:s1] = right[s0:s1] * (1 - 0.5 * ramp) + 45 * ramp * rhythm

    # --- artifact 1: slow, huge movement
    a0, a1 = 200 * fs, 203 * fs
    swing = 700 * np.sign(np.sin(2 * np.pi * 1.3 * t[a0:a1]))
    left[a0:a1] += swing
    right[a0:a1] += swing

    # --- artifact 2: muscle burst (broadband noise, mostly high frequency)
    if emg is not None:
        m0, m1 = int(emg[0] * fs), int(emg[1] * fs)
        burst = rng.normal(size=(2, m1 - m0))
        burst = np.diff(burst, axis=1, prepend=0)                        # differencing = emphasise high freq
        left[m0:m1] += 60 * burst[0]
        right[m0:m1] += 60 * burst[1]

    return np.vstack([left, right]), float(fs)


# -------------------------------------------------------------------- real data
def _find_channel(names, wanted):
    """Match a channel name ignoring case/spaces; handle CHB-MIT's duplicated 'T8-P8-0' names."""
    clean = lambda s: s.upper().replace(" ", "")
    for n in names:
        if clean(n) == clean(wanted):
            return n
    for n in names:
        if clean(n).startswith(clean(wanted) + "-"):
            return n
    raise ValueError(f"Channel '{wanted}' not found. Channels in this file: {names}")


def load_chbmit(edf_path, left="F3-C3", right="F4-C4"):
    """
    Read one CHB-MIT recording.  Returns (data_uv, fs, (left_name, right_name)).

    Why F3-C3 / F4-C4?  Your document compares C3 (left motor cortex) with C4 (right).
    CHB-MIT uses a bipolar montage, so the closest left/right pair over that area is
    F3-C3 and F4-C4.  (The old notebook used data[0] and data[1] = FP1-F7 and F7-T7:
    both on the LEFT side, so it could never measure left-vs-right.)
    """
    try:
        import mne  # the reference EDF reader, used when installed
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        names, fs = raw.ch_names, float(raw.info["sfreq"])
        l_name, r_name = _find_channel(names, left), _find_channel(names, right)
        data = raw.get_data(picks=[l_name, r_name]) * 1e6          # MNE gives volts; we want microvolts
    except ImportError:                                            # no mne: use the built-in reader below
        names, signals, fs = read_edf(edf_path)
        l_name, r_name = _find_channel(names, left), _find_channel(names, right)
        data = np.vstack([signals[names.index(l_name)], signals[names.index(r_name)]])
    return data, fs, (l_name, r_name)


def read_edf(path):
    """
    Minimal EDF reader (pure numpy) so you do not need `mne`.  EDF = a 256-byte header, then
    one 256-byte block per signal, then the data as 16-bit integers stored in "records".
    Returns (channel_names, list_of_arrays_in_microvolts, sampling_rate).
    Assumes every channel has the same sampling rate (true for CHB-MIT).
    NOT tested on a real CHB-MIT file in the environment this was written in -- if it fails,
    `pip install mne` and load_chbmit() will use that instead.
    """
    with open(path, "rb") as f:
        head = f.read(256)
        header_bytes, n_rec = int(head[184:192]), int(head[236:244])
        rec_dur, ns = float(head[244:252]), int(head[252:256])

        def block(width):                                          # one field for ALL signals
            raw = f.read(ns * width)
            return [raw[i * width:(i + 1) * width].decode("ascii", "ignore").strip() for i in range(ns)]

        labels, _transducer, units = block(16), block(80), block(8)
        pmin, pmax = [float(x) for x in block(8)], [float(x) for x in block(8)]
        dmin, dmax = [float(x) for x in block(8)], [float(x) for x in block(8)]
        _prefilter, nsamp = block(80), [int(x) for x in block(8)]
        f.seek(header_bytes)
        ints = np.frombuffer(f.read(), dtype="<i2")

    per_rec = sum(nsamp)
    if n_rec < 0:                                                  # -1 = "unknown": infer from file size
        n_rec = len(ints) // per_rec
    ints = ints[:n_rec * per_rec].reshape(n_rec, per_rec)
    scale_to_uv = {"uV": 1.0, "µV": 1.0, "mV": 1e3, "V": 1e6}
    seen, names, out, off, fs = {}, [], [], 0, None
    for k in range(ns):
        sig = ints[:, off:off + nsamp[k]].reshape(-1).astype(float)
        off += nsamp[k]
        if labels[k].startswith("EDF Annotations"):
            continue
        gain = (pmax[k] - pmin[k]) / (dmax[k] - dmin[k])
        out.append(((sig - dmin[k]) * gain + pmin[k]) * scale_to_uv.get(units[k], 1.0))
        names.append(labels[k])
        fs = nsamp[k] / rec_dur
    for n in set(names):                                           # duplicates -> T8-P8-0, T8-P8-1 (as MNE does)
        if names.count(n) > 1:
            seen[n] = 0
    for i, n in enumerate(names):
        if n in seen:
            names[i], seen[n] = f"{n}-{seen[n]}", seen[n] + 1
    return names, out, fs


def parse_summary(summary_path, edf_name):
    """
    Read seizure times for one file from chbXX-summary.txt.

    The summary file looks like:
        File Name: chb01_03.edf
        ...
        Number of Seizures in File: 1
        Seizure Start Time: 2996 seconds
        Seizure End Time: 3036 seconds

    Returns a list of (start_s, end_s) tuples (empty list if the file has no seizure).
    Newer CHB-MIT files write "Seizure 1 Start Time:" so the pattern allows an optional number.
    IMPORTANT: these times are only used to SCORE the detector, never as an input to it.
    """
    text = open(summary_path, encoding="utf-8", errors="ignore").read()
    blocks = re.split(r"(?im)^\s*File Name:\s*", text)[1:]
    for block in blocks:
        if block.strip().startswith(edf_name):
            starts = re.findall(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)", block, flags=re.I)
            ends = re.findall(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)", block, flags=re.I)
            return [(float(a), float(b)) for a, b in zip(starts, ends)]
    raise ValueError(f"{edf_name} not found in {summary_path}")
