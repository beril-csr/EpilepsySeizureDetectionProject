"""
ca.py  -  the memristive Cellular Automaton (CA) of Karamani et al., driven by OmniNeuro.

WHAT IS A CELLULAR AUTOMATON?  A grid of simple cells.  Each cell is ON or OFF.  At every time-step
each cell looks at its inputs and applies one fixed rule to decide its next state.  Nothing else.
Complex behaviour (waves, bursts) comes purely from many cells interacting.

THE MODEL (follows section II-IV of the paper in your document)
  * TWO grids of 25 x 25 cells:  grid 0 = PATHOGENIC (the seizure focus), grid 1 = HEALTHY.
  * Wiring is RANDOM, not "nearest neighbours":  every cell listens to between 2 and 7 random
    cells of its own grid.  Cells in the central 12 x 12 block of each grid also listen to 6
    random cells in the central block of the OTHER grid (the "long-range synapses").
  * Rule:   a cell turns ON if  (a) at least half of its inputs are ON,
                                or (b) its external excitation signal `Exc` is above 0.5.
  * Refractory period: after being ON for one step, a cell is deaf for 2 steps (like a neuron).
  * The cell's state lives in a MEMRISTOR (memristor.py): READ phase tells neighbours the state,
    WRITE phase applies +V (set) or -V (reset).

WHAT OMNINEURO CHANGES  (this is the new part, from your "Summary" page)
  The paper switches between a healthy phase and an epileptiform phase by hand.  Here the
  switch is a number  s  between 0 and 1 that comes from OmniNeuro's confidence (P_move):
      s = 0 : healthy.  Pathogenic grid has all its synapses and gets only low random noise.
      s = 1 : epileptiform.  The pathogenic grid LOSES synapses (paper: Z_epilepsy < Z_healthy)
              and its central block receives a strong CHAOTIC excitation signal.
  Values in between give a smooth in-between state.
  The activity of the HEALTHY grid is the "spread of the seizure" that the paper shows in Fig. 5-6.
"""
import numpy as np

from .memristor import MemristorArray

V_WRITE = 3.0           # volts applied when the SET condition is met (> V_S, so it switches)
V_REFRESH = 3.0         # volts (negative) applied otherwise: resets the cell
THRESH_NEIGH = 0.5      # fraction of ON inputs needed to switch a cell ON
THRESH_EXC = 0.5        # external excitation above this switches a cell ON
INACTIVE_STEPS = 2      # refractory period after activation
MAX_INTRA, MIN_INTRA, N_INTER = 7, 2, 6
SYNAPSE_LOSS = 0.6      # at s=1 the pathogenic grid keeps (1-0.6)=40 % of its intra-grid synapses
NOISE_MEAN, NOISE_STD = 0.20, 0.12   # background excitation ("random amplitude noise")
CHAOS_R = 3.99          # logistic-map parameter (3.99 = fully chaotic)


class MemristiveCA:
    def __init__(self, grid=25, center=12, seed=0):
        self.g, self.n = grid, grid * grid
        self.rng = np.random.default_rng(seed)
        N = self.n
        self.mem = MemristorArray(2 * N)                     # cells 0..N-1 pathogenic, N..2N-1 healthy
        self.timer = np.zeros(2 * N, dtype=int)              # refractory countdown per cell
        self.chaos_x = 0.37                                  # state of the logistic map (see _excitation)

        # ---- which cells are in the central block (they get inter-grid links)
        r0 = (grid - center) // 2
        block = np.zeros((grid, grid), bool)
        block[r0:r0 + center, r0:r0 + center] = True
        self.center_local = block.ravel()                    # length N
        self.center = np.concatenate([self.center_local, self.center_local])   # length 2N
        center_ids = np.flatnonzero(self.center_local)

        # ---- wiring: columns 0..6 = intra-grid inputs, columns 7..12 = inter-grid inputs
        src = np.zeros((2 * N, MAX_INTRA + N_INTER), dtype=int)
        for gi in (0, 1):
            offset, other = gi * N, (1 - gi) * N
            for i in range(N):
                pool = np.delete(np.arange(N), i)            # not itself
                src[offset + i, :MAX_INTRA] = offset + self.rng.choice(pool, MAX_INTRA, replace=False)
                if self.center_local[i]:
                    src[offset + i, MAX_INTRA:] = other + self.rng.choice(center_ids, N_INTER, replace=False)
        self.src = src
        # how many intra-grid synapses each cell has when healthy: random integer 2..7
        self.k_healthy = self.rng.integers(MIN_INTRA, MAX_INTRA + 1, size=2 * N)
        self.state = self.mem.read()

    # ------------------------------------------------------------------ helpers
    def _intra_counts(self, s):
        """Number of intra-grid synapses per cell at epilepsy level s (pathogenic grid shrinks)."""
        k = self.k_healthy.copy()
        patho = np.arange(2 * self.n) < self.n
        shrunk = np.maximum(MIN_INTRA, np.round(k * (1.0 - SYNAPSE_LOSS * s))).astype(int)
        k[patho] = shrunk[patho]
        return k

    def _excitation(self, s):
        """External excitation Exc for every cell (values in [0, 1]).
        Healthy phase: weak random noise everywhere (rarely above the 0.5 threshold).
        Epileptiform phase: the pathogenic central block gets a CHAOTIC signal -- here a logistic
        map x -> r x (1-x), which is deterministic yet unpredictable -- shared by the block (so the
        block tends to fire together, as in a seizure) plus a little per-cell jitter."""
        exc = np.clip(self.rng.normal(NOISE_MEAN, NOISE_STD, 2 * self.n), 0, 1)
        self.chaos_x = CHAOS_R * self.chaos_x * (1 - self.chaos_x)
        jitter = self.rng.normal(0, 0.08, self.n)
        chaos = np.clip(self.chaos_x + jitter, 0, 1)
        pc = np.flatnonzero(self.center_local)                   # pathogenic central block
        exc[pc] = (1 - s) * exc[pc] + s * chaos[pc]
        return exc

    # --------------------------------------------------------------------- step
    def step(self, s):
        """Advance one time-step at epilepsy level s (0..1).  Returns boolean array of length 2N."""
        s = float(np.clip(s, 0.0, 1.0))
        C = self.mem.read()                                          # READ phase: current states

        k = self._intra_counts(s)                                    # which synapses exist now
        slot = np.arange(self.src.shape[1])[None, :]
        active_slot = np.where(slot < MAX_INTRA, slot < k[:, None], self.center[:, None])
        n_inputs = active_slot.sum(axis=1)
        n_on = (C[self.src] & active_slot).sum(axis=1)
        frac_on = n_on / np.maximum(n_inputs, 1)                     # "mean of inputs" (Millman)

        exc = self._excitation(s)
        deaf = self.timer > 0                                        # refractory cells ignore inputs
        set_cell = ((frac_on >= THRESH_NEIGH) | (exc > THRESH_EXC)) & ~deaf

        volts = np.where(set_cell, V_WRITE, -V_REFRESH)              # WRITE phase
        self.mem.write(volts)

        self.timer = np.maximum(self.timer - 1, 0)
        self.timer[set_cell] = INACTIVE_STEPS                        # newly active -> refractory
        self.state = self.mem.read()
        return self.state

    # ------------------------------------------------------------------ outputs
    def grids(self):
        """Current states as two g x g boolean images: (pathogenic, healthy)."""
        return self.state[:self.n].reshape(self.g, self.g), self.state[self.n:].reshape(self.g, self.g)

    def activity(self):
        """Fraction of cells ON in (pathogenic, healthy)."""
        return float(self.state[:self.n].mean()), float(self.state[self.n:].mean())

    def mean_resistance(self):
        r = self.mem.resistance
        return float(r[:self.n].mean()), float(r[self.n:].mean())
