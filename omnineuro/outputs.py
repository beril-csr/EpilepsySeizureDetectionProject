"""
outputs.py  -  everything a human sees or hears (section 2.3 and section 5 of your document).

    dashboard()    one picture: engines, probability, safety states, CA activity, grid snapshots
    bloch_figure() the "arrow in a sphere" of Fig. 3
    sonify()       Table 1: P_move -> pitch, |dHFD| -> timbre, energy -> volume; silence when unsure
    make_report()  Fig. 7: a cautious text report (rule-based; optional LLM rewrite)
"""
import os

import matplotlib

matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from . import safety

# Validated categorical colours (blue, orange, aqua) + neutrals.
BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b"
STATE_COLOR = {safety.NORMAL: "#d9d8d3", safety.UNCLEAR: "#eda100",
               safety.SEIZURE: ORANGE, safety.ARTIFACT: "#6250d6"}


def _shade(ax, seizures):
    for k, (a, b) in enumerate(seizures or []):
        ax.axvspan(a, b, color=GRAY, alpha=0.18, lw=0, label="labelled seizure" if k == 0 else None)


def _style(ax, ylabel):
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis="y", color="#e5e4df", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# --------------------------------------------------------------------- dashboard
def _snapshot_windows(res, seizures):
    if seizures:
        a, b = seizures[0]
        i0, i1 = int(a // res.cfg.win_s), int(b // res.cfg.win_s)
        pick = [max(i0 - 10, 0), i0 + 2, (i0 + i1) // 2, min(i1 + 10, res.n - 1)]
    else:
        pick = [int(np.argmax(res.p_move))] * 1 + list(np.linspace(0, res.n - 1, 3).astype(int))
    return pick[:4]


def dashboard(res, path, seizures=None, title="OmniNeuro -> memristive CA"):
    """One picture: engines, probabilities, safety states, CA activity, and grid snapshots."""
    idx = _snapshot_windows(res, seizures)
    t = res.t_mid
    fig = plt.figure(figsize=(12, 12.5))
    gs = fig.add_gridspec(6, 8, height_ratios=[1.1, 1.1, 0.22, 1.1, 1.0, 1.0], hspace=0.6)

    ax0 = fig.add_subplot(gs[0, :]); _shade(ax0, seizures)
    ax0.plot(t, np.clip(res.z_energy, -10, 20), color=BLUE, lw=1.4, label="Physics: energy (z)")
    ax0.plot(t, np.clip(res.z_chaos, -10, 20), color=ORANGE, lw=1.4, label="Chaos: complexity drop (z)")
    ax0.axvline(res.cfg.baseline_s, color=GRAY, ls=":", lw=1)
    _style(ax0, "z-score (clipped at -10..20)")
    ax0.set_ylim(-13, 30)
    ax0.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    ax0.set_title(title, loc="left", fontsize=12, color=INK)

    ax1 = fig.add_subplot(gs[1, :], sharex=ax0); _shade(ax1, seizures)
    ax1.axhspan(res.cfg.amb_lo, res.cfg.amb_hi, color="#eda100", alpha=0.18, lw=0, label="ambiguity zone")
    ax1.plot(t, res.p_raw, color=GRAY, lw=1, label="raw probability")
    ax1.plot(t, res.p_move, color=BLUE, lw=1.8, label="P_move (quantum engine)")
    ax1.set_ylim(-0.02, 1.02); _style(ax1, "probability")
    ax1.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)

    ax2 = fig.add_subplot(gs[2, :], sharex=ax0)
    for st, col in STATE_COLOR.items():
        m = res.state == st
        if m.any():
            ax2.bar(res.t0[m], 1, width=res.cfg.win_s, align="edge", color=col, label=st.lower())
    ax2.set_yticks([]); ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", bbox_to_anchor=(0, -0.3), fontsize=8, frameon=False, ncol=4)
    for side in ("top", "right", "left"):
        ax2.spines[side].set_visible(False)

    ax3 = fig.add_subplot(gs[3, :], sharex=ax0); _shade(ax3, seizures)
    k = res.cfg.steps_per_window
    ts = np.repeat(res.t0, k) + np.tile(np.arange(k) / k, res.n) * res.cfg.win_s
    sm = lambda v: np.convolve(v, np.ones(k) / k, mode="same")
    ax3.plot(ts, sm(res.ca_patho) * 100, color=ORANGE, lw=1.4, label="pathogenic grid")
    ax3.plot(ts, sm(res.ca_healthy) * 100, color=AQUA, lw=1.4, label="healthy grid")
    _style(ax3, "cells ON (%)"); ax3.set_xlabel("time (s)")
    ax3.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)

    for a in (ax0, ax1, ax2):
        a.tick_params(labelbottom=False)
    for j, i in enumerate(idx):
        for r, (snap, name) in enumerate(((res.ca_snap_p, "pathogenic"), (res.ca_snap_h, "healthy"))):
            a = fig.add_subplot(gs[4 + r, j * 2:(j + 1) * 2])
            a.imshow(snap[i], cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{name}  t={res.t0[i]:.0f}s" if r == 0 else f"{name}", fontsize=8, color=INK)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ Bloch sphere
def bloch_figure(res, path, i=None):
    """Fig. 3: the 'mental vector' on the Bloch sphere. North = normal, south = seizure-like."""
    if i is None:
        i = int(np.argmax(res.p_move))
    fig = plt.figure(figsize=(4.6, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 36), np.linspace(0, np.pi, 18))
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="#d9d8d3", lw=0.4)
    x, y, z = res.bloch[i]
    ax.quiver(0, 0, 0, x, y, z, color=BLUE, lw=3, arrow_length_ratio=0.15)
    ax.text(0, 0, 1.15, "|0>  normal", ha="center", fontsize=8)
    ax.text(0, 0, -1.3, "|1>  seizure-like", ha="center", fontsize=8)
    ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"P_move = {res.p_move[i] * 100:.0f}%   (t = {res.t0[i]:.0f} s)", fontsize=10)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ sonification
PENTATONIC = np.array([0, 2, 4, 7, 9])        # C major pentatonic: any note sounds pleasant


def note_frequency(p_move):
    """P_move 0..1 -> a note in C major pentatonic across two octaves (higher confidence = higher pitch)."""
    n = len(PENTATONIC) * 2
    k = int(np.clip(p_move * n, 0, n - 1))
    semitones = PENTATONIC[k % 5] + 12 * (k // 5)
    return 261.63 * 2 ** (semitones / 12)      # C4 = 261.63 Hz


def sonify(res, path, clip=None, sr=22050, note_s=0.3):
    """
    Table 1 of your document:
        P_move   -> PITCH   (high = confident)
        |dHFD|   -> TIMBRE  (pure tone = focused;  extra harmonics + noise = distorted)
        energy   -> VOLUME
    Safety protocol: UNCLEAR and ARTIFACT windows are SILENT (the feedback "fades to silence").
    clip = (start_s, end_s) renders only part of the recording.
    """
    rng = np.random.default_rng(0)
    i0, i1 = 0, res.n
    if clip:
        i0, i1 = int(clip[0] // res.cfg.win_s), min(int(clip[1] // res.cfg.win_s), res.n)
    m = int(sr * note_s)
    tt = np.arange(m) / sr
    env = np.sin(np.pi * np.arange(m) / m) ** 0.5       # soft attack and release, no clicks
    out = []
    for i in range(i0, i1):
        if res.state[i] in (safety.UNCLEAR, safety.ARTIFACT):
            out.append(np.zeros(m)); continue
        f = note_frequency(res.p_move[i])
        volume = 0.15 + 0.65 / (1 + np.exp(-(res.z_energy[i] - 3) / 3))      # sigmoid of energy z
        dist = float(np.clip(abs(res.z_dhfd[i]) / 6.0, 0, 0.8))               # 0 pure .. 0.8 distorted
        tone = np.sin(2 * np.pi * f * tt) + dist * (0.6 * np.sin(2 * np.pi * 3 * f * tt)
                                                   + 0.4 * rng.normal(size=m))
        out.append(volume * env * tone / (1 + dist))
    audio = np.concatenate(out) if out else np.zeros(m)
    peak = max(float(np.abs(audio).max()), 1e-9)
    wavfile.write(path, sr, (audio / max(peak, 1.0) * 32767 * 0.9).astype(np.int16))
    return path


# ------------------------------------------------------------------------ report
def make_report(res, i=None, use_llm=False):
    """
    Section 2.3.1 / Fig. 7.  Always produces the rule-based report (works offline - section 6.4's
    "offline safety mechanism").  If use_llm=True and a key is set in the environment variable
    GEMINI_API_KEY, the text is passed to Gemini to be rephrased; any failure falls back silently
    to the local report.  The model is NEVER told the true label.
    """
    if i is None:
        i = int(np.argmax(np.where(res.artifact, -1, res.p_move)))
    st = res.state[i]
    side = "right" if res.lidx[i] > 0 else "left"
    lines = [
        "OMNINEURO FEEDBACK REPORT (decision support - requires clinician verification)",
        f"Window starting at t = {res.t0[i]:.0f} s     State: {st}",
        f"  Physics  : energy {res.z_energy[i]:+.1f} z vs baseline;  Lidx = {res.lidx[i]:+.2f} "
        f"({side} hemisphere has more energy)",
        f"  Chaos    : complexity (HFD) {res.hfd[i]:.2f}, {res.z_chaos[i]:+.1f} z "
        f"({'DROP' if res.z_chaos[i] > 0 else 'rise/no change'});  dHFD = {res.dhfd[i]:+.3f}",
        f"  Quantum  : P_move = {res.p_move[i]:.2f};  hemispheric phase coherence = {res.coherence[i]:.2f}",
    ]
    if st == safety.SEIZURE:
        lines.append("Interpretation: energy is above this patient's own baseline while signal complexity is "
                     "below it, and the pattern persisted across windows. This is compatible with rhythmic "
                     "(possibly ictal) activity. It is NOT a diagnosis.")
    elif st == safety.UNCLEAR:
        lines.append("Interpretation: evidence is ambiguous (0.4 < P < 0.6). No feedback issued. Relax and continue.")
    elif st == safety.ARTIFACT:
        lines.append(f"Interpretation: window rejected - {res.artifact_why[i]}. Check electrodes / muscle tension.")
    else:
        lines.append("Interpretation: within this patient's baseline range.")
    text = "\n".join(lines)
    if use_llm and os.environ.get("GEMINI_API_KEY"):
        try:
            from google import genai                                  # pip install google-genai
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            prompt = ("You are assisting a neurophysiologist. Rewrite the EEG feature summary below as a short, "
                      "cautious note. Use ONLY the numbers given, do not diagnose, and end with: "
                      "'Requires clinician verification.'\n\n" + text)
            reply = client.models.generate_content(
                model=os.environ.get("OMNINEURO_LLM_MODEL", "gemini-2.5-flash"), contents=prompt)
            text += "\n\nLLM-rephrased note:\n" + reply.text
        except Exception as e:                                         # offline, bad key, renamed model ...
            text += f"\n\n(LLM unavailable: {type(e).__name__}. Local report shown.)"
    return text
