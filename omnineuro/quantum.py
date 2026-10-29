"""
quantum.py  -  the Quantum-Inspired engine (Bloch sphere + interference).

IMPORTANT: this is *quantum probability formalism* used as a maths tool (exactly how your
document words it). No quantum physics happens in the brain or in this code; we just borrow
two ideas from the maths of quantum mechanics:

  1. A two-state system is a point on a sphere (the Bloch sphere).
     North pole = state |0> ("normal"), south pole = state |1> ("seizure-like").
     The polar angle theta says how far toward |1> we are, and
         P_move = |<1|psi>|^2 = sin^2(theta / 2)          <- the formula in your document
     theta = 0 -> P = 0,   theta = pi/2 -> P = 0.5 (maximum ambiguity),   theta = pi -> P = 1.

  2. Probabilities come from squaring a COMPLEX amplitude, and amplitudes ADD before squaring:
         |a + b|^2 = |a|^2 + |b|^2 + 2|a||b|cos(phase difference)
     The last term is "interference" (your note (1)).  Amplitudes pointing the same way
     reinforce; amplitudes pointing different ways partly cancel.

What the code does with those ideas
-----------------------------------
For each window we build a complex amplitude
        a_t = sqrt(p_raw) * e^(i * phase_lag_t)
(strength = how seizure-like the engines say the window is; direction = how the two hemispheres
line up in time).  The running state is a leaky sum of these amplitudes:
        b_t = memory * b_(t-1) + (1 - memory) * a_t
and the output is P_move = |b_t|^2.
A steady, synchronised signal keeps adding up (constructive).  An isolated burst whose phase is
unrelated to its neighbours is damped (partially destructive).
HONEST NOTE: with phases forced to 0 (cfg.use_phase=False) this is an ordinary smoothing filter.
Whether the phase term gives a real advantage on real EEG is something you can TEST with the
ablation in docs/HONEST_LIMITS.md - it is not proven here.
"""
import numpy as np


class QuantumEngine:
    def __init__(self, memory=0.5, use_phase=True):
        self.memory = memory
        self.use_phase = use_phase
        self.b = 0j                      # complex state; starts at "nothing happening"

    def reset(self):
        self.b = 0j

    def step(self, p_raw, phase_lag=0.0):
        """Feed one window.  Returns (p_move, theta, bloch_xyz)."""
        phase = phase_lag if self.use_phase else 0.0
        a = np.sqrt(np.clip(p_raw, 0.0, 1.0)) * np.exp(1j * phase)
        self.b = self.memory * self.b + (1.0 - self.memory) * a
        mag = min(abs(self.b), 1.0)              # |b| = sqrt(P_move); cannot exceed 1
        p_move = mag ** 2
        theta = 2.0 * np.arcsin(mag)             # inverse of P = sin^2(theta/2)
        phi = np.angle(self.b)                   # azimuth angle around the sphere
        bloch = (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
        return float(p_move), float(theta), bloch
