"""
pipeline.py  -  glues everything together.  This is the file that shows the WHOLE idea in order:

    EEG (left, right)
      1. band-pass filter, cut into 2-second windows
      2. per window:  Physics (energy, Lidx)   Chaos (HFD, dHFD)   Quantum input (phase lag, coherence)
                      + artifact check (safety.py)
      3. compare with this patient's own baseline  ->  z-scores
      4. fuse the engines into evidence -> raw probability p_raw        (engines.fuse)
      5. quantum engine smooths/combines windows -> P_move               (quantum.py)
      6. safety protocol -> NORMAL / SEIZURE-LIKE / UNCLEAR / ARTIFACT   (safety.py)
      7. P_move sets the epilepsy level s of the memristive CA           (ca.py)

`analyse()` returns a `Result` (a bag of arrays, one entry per window).  `evaluate()` scores it
against known seizure times.  Plotting / sound / text live in outputs.py.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata

from . import engines, safety
from .ca import MemristiveCA
from .config import Config
from .quantum import QuantumEngine


@dataclass
class Result:
    cfg: Config
    fs: float
    t0: np.ndarray                      # start time (s) of every window
    # engine outputs per window
    energy: np.ndarray = None           # log total energy
    lidx: np.ndarray = None             # Eq.(1)  left/right energy ratio
    hfd: np.ndarray = None              # mean Higuchi FD of both channels
    dhfd: np.ndarray = None             # Eq.(2)  HFD(left) - HFD(right)
    coherence: np.ndarray = None        # phase coherence between hemispheres (0..1)
    phase_lag: np.ndarray = None
    emg: np.ndarray = None              # muscle-noise ratio
    artifact: np.ndarray = None         # bool
    artifact_why: list = field(default_factory=list)
    # z-scores vs baseline
    z_energy: np.ndarray = None
    z_chaos: np.ndarray = None          # sign flipped: POSITIVE when complexity DROPS
    z_lidx: np.ndarray = None
    z_dhfd: np.ndarray = None
    # decision chain
    evidence: np.ndarray = None
    p_raw: np.ndarray = None
    p_move: np.ndarray = None
    theta: np.ndarray = None
    bloch: np.ndarray = None            # (n, 3)
    state: np.ndarray = None            # array of strings
    s: np.ndarray = None                # epilepsy level sent to the CA
    # CA outputs
    ca_patho: np.ndarray = None         # fraction of cells ON, every CA step
    ca_healthy: np.ndarray = None
    ca_snap_p: np.ndarray = None        # (n, g, g) snapshot at the end of each window
    ca_snap_h: np.ndarray = None
    ca_R_patho: np.ndarray = None       # mean memristance (ohm) at the end of each window
    ca_R_healthy: np.ndarray = None
    baseline_mask: np.ndarray = None

    @property
    def n(self):
        return len(self.t0)

    @property
    def t_mid(self):
        return self.t0 + self.cfg.win_s / 2


# ------------------------------------------------------------------ main routine
def analyse(data, fs, cfg=None, run_ca=True):
    """
    data : (2, n_samples) array in microvolts.  Row 0 = left hemisphere, row 1 = right.
    Returns a Result.
    """
    cfg = cfg or Config()
    data = np.asarray(data, dtype=float)
    assert data.ndim == 2 and data.shape[0] == 2, "data must have shape (2, n_samples): left, right"

    filt = engines.bandpass(data, fs, cfg.band_lo, cfg.band_hi)
    win = int(round(cfg.win_s * fs))
    n = data.shape[1] // win
    res = Result(cfg=cfg, fs=fs, t0=np.arange(n) * cfg.win_s)

    # ---- 2. per-window features -------------------------------------------------
    keys = ["energy", "lidx", "hfd", "dhfd", "coherence", "phase_lag", "emg"]
    feat = {k: np.zeros(n) for k in keys}
    artifact = np.zeros(n, bool)
    why = []
    for i in range(n):
        a, b = i * win, (i + 1) * win
        L, R = filt[0, a:b], filt[1, a:b]
        feat["energy"][i] = engines.log_total_energy(L, R, cfg.eps)
        feat["lidx"][i] = engines.log_energy_ratio(L, R, cfg.eps)
        hl, hr = engines.higuchi_fd(L, cfg.kmax), engines.higuchi_fd(R, cfg.kmax)
        feat["hfd"][i], feat["dhfd"][i] = (hl + hr) / 2, hl - hr
        feat["phase_lag"][i], feat["coherence"][i] = engines.phase_sync(L, R)
        feat["emg"][i] = safety.emg_ratio(data[:, a:b], fs)
        reason = safety.artifact_reason(filt[:, a:b], feat["emg"][i], cfg)
        artifact[i] = bool(reason)
        why.append(reason)
    for k, v in feat.items():
        setattr(res, k, v)
    res.artifact, res.artifact_why = artifact, why

    # ---- 3. baseline z-scores ---------------------------------------------------
    base = (res.t0 + cfg.win_s <= cfg.baseline_s) & ~artifact
    if base.sum() < 10:
        raise ValueError("Fewer than 10 clean baseline windows. Increase baseline_s or check the recording.")
    res.baseline_mask = base
    if cfg.rolling_baseline:                                 # "normal" = the last ref_s seconds
        ref_n, gap_n = int(cfg.ref_s / cfg.win_s), int(cfg.gap_s / cfg.win_s)
        zf = lambda v: engines.rolling_z(v, artifact, base, ref_n, gap_n)
    else:                                                    # "normal" = the first baseline_s seconds
        zf = lambda v: engines.robust_z(v, base)
    res.z_energy = zf(res.energy)
    res.z_chaos = -zf(res.hfd)                               # minus: a DROP in chaos is the seizure sign
    res.z_lidx = zf(res.lidx)
    res.z_dhfd = zf(res.dhfd)

    # ---- 4. fusion  -> 5. quantum -> 6. safety -----------------------------------
    res.evidence, res.p_raw = engines.fuse(res.z_energy, res.z_chaos, cfg)
    q = QuantumEngine(cfg.memory, cfg.use_phase)
    p_move, theta, bloch = np.zeros(n), np.zeros(n), np.zeros((n, 3))
    last = (0.0, 0.0, (0.0, 0.0, 1.0))
    for i in range(n):
        if not artifact[i]:                                   # artifacts add no evidence: state is held
            last = q.step(res.p_raw[i], res.phase_lag[i])
        p_move[i], theta[i], bloch[i] = last[0], last[1], last[2]
    res.p_move, res.theta, res.bloch = p_move, theta, bloch
    res.state = np.array([safety.decide_state(p_move[i], artifact[i], cfg) for i in range(n)])

    # ---- 7. memristive CA --------------------------------------------------------
    res.s = epilepsy_level(res.state, res.p_move)
    if run_ca:
        run_cellular_automaton(res)
    return res


def epilepsy_level(state, p_move, smooth=0.5, hold_decay=0.9):
    """
    Turn OmniNeuro's decision into the CA's epilepsy level s (0..1), one value per window.
      SEIZURE-LIKE : target rises with confidence (P_move 0.5 -> 0,  1.0 -> 1)
      NORMAL       : target 0
      UNCLEAR / ARTIFACT : no new evidence -> keep the old value but let it fade (x0.9)
    Then s moves only part-way to the target each window (`smooth`), so the model changes gradually.
    """
    s = np.zeros(len(state))
    cur = 0.0
    for i, st in enumerate(state):
        if st == safety.SEIZURE:
            cur += smooth * (np.clip((p_move[i] - 0.5) / 0.5, 0, 1) - cur)
        elif st == safety.NORMAL:
            cur += smooth * (0.0 - cur)
        else:
            cur *= hold_decay
        s[i] = cur
    return s


def run_cellular_automaton(res):
    """Run the memristive CA along the recording: each EEG window = `steps_per_window` CA steps."""
    cfg, n = res.cfg, res.n
    ca = MemristiveCA(cfg.grid, cfg.center, cfg.seed)
    k = cfg.steps_per_window
    res.ca_patho, res.ca_healthy = np.zeros(n * k), np.zeros(n * k)
    res.ca_snap_p = np.zeros((n, cfg.grid, cfg.grid), dtype=np.uint8)
    res.ca_snap_h = np.zeros_like(res.ca_snap_p)
    res.ca_R_patho, res.ca_R_healthy = np.zeros(n), np.zeros(n)
    for i in range(n):
        for j in range(k):
            ca.step(res.s[i])
            res.ca_patho[i * k + j], res.ca_healthy[i * k + j] = ca.activity()
        p, h = ca.grids()
        res.ca_snap_p[i], res.ca_snap_h[i] = p, h
        res.ca_R_patho[i], res.ca_R_healthy[i] = ca.mean_resistance()


# --------------------------------------------------------------------- scoring
def events(mask):
    """Runs of consecutive True values -> list of (first_index, last_index)."""
    out, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def auc(scores, labels):
    """Area under the ROC curve: the chance that a random seizure window scores higher than a random
    normal one.  0.5 = coin flip, 1.0 = perfect.  (Mann-Whitney formulation.)"""
    labels = np.asarray(labels, bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(res, seizures, tolerance_s=30.0):
    """
    Score the detector against known seizure times.  `seizures` = list of (start_s, end_s).
    These times are used ONLY here - never as an input to the detector.

      sensitivity       : fraction of seizures with at least one SEIZURE-LIKE alarm during them
      latency_s         : seconds from seizure start until the alarm window ENDS (the earliest the
                          system could know), per detected seizure
      false_alarms      : alarm events that are not within `tolerance_s` of any seizure
      false_alarms_per_h: the same, per hour of analysed (post-baseline) recording
      auc_p_move        : how well P_move separates seizure windows from the rest
    """
    cfg = res.cfg
    w = cfg.win_s
    lab = np.zeros(res.n, bool)
    for a, b in seizures:
        lab |= (res.t0 < b) & (res.t0 + w > a)
    alarm = res.state == safety.SEIZURE
    usable = ~res.baseline_mask & (res.t0 >= cfg.baseline_s) & ~res.artifact

    detected, latency = [], []
    for a, b in seizures:
        hit = np.flatnonzero(alarm & (res.t0 < b) & (res.t0 + w > a))
        detected.append(len(hit) > 0)
        if len(hit):
            latency.append(float(res.t0[hit[0]] + w - a))
    false_alarms = 0
    for i0, i1 in events(alarm):
        t_a, t_b = res.t0[i0], res.t0[i1] + w
        near = any(t_a < b + tolerance_s and t_b > a - tolerance_s for a, b in seizures)
        false_alarms += 0 if near else 1
    # alarms that are not near any seizure: how many MINUTES, and how many separate EPISODES
    # (alarm bursts closer than 60 s apart are merged, so flicker does not inflate the count)
    near = np.zeros(res.n, bool)
    for a, b in seizures:
        near |= (res.t0 < b + tolerance_s) & (res.t0 + w > a - tolerance_s)
    fa_win = alarm & ~near
    merged = []
    for i0, i1 in events(fa_win):
        if merged and (i0 - merged[-1][1]) * w <= 60.0:
            merged[-1] = (merged[-1][0], i1)
        else:
            merged.append((i0, i1))
    hours = max((res.n * w - cfg.baseline_s) / 3600.0, 1e-9)
    out = {
        "n_seizures": len(seizures),
        "sensitivity": float(np.mean(detected)) if detected else float("nan"),
        "latency_s": latency,
        "false_alarms": false_alarms,
        "false_alarms_per_h": false_alarms / hours,
        "false_alarm_episodes": len(merged),
        "false_alarm_minutes": float(fa_win.sum() * w / 60.0),
        "auc_p_move": auc(res.p_move[usable], lab[usable]),
        "auc_p_raw": auc(res.p_raw[usable], lab[usable]),
        "artifact_windows": int(res.artifact.sum()),
        "state_flips": int(np.sum(res.state[1:] != res.state[:-1])),
    }
    if res.ca_healthy is not None and seizures:
        k = cfg.steps_per_window
        step_t = np.repeat(res.t0, k)
        a, b = seizures[0]
        pre = (step_t >= a - 120) & (step_t < a - 20)
        dur = (step_t >= a + 10) & (step_t < b)
        out["ca_patho_pre_during"] = (float(res.ca_patho[pre].mean()), float(res.ca_patho[dur].mean()))
        out["ca_healthy_pre_during"] = (float(res.ca_healthy[pre].mean()), float(res.ca_healthy[dur].mean()))
    return out
