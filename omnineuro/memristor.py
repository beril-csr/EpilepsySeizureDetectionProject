"""
memristor.py  -  a software memristor (the "memory resistor").

What is it?  A resistor whose resistance depends on the voltage that has flowed through it in the
past, and which REMEMBERS that value when the power is off.  Your note says it behaves like a
neuron/slime mould: it has a history.

In the Karamani et al. paper each CA cell stores its state (0 or 1) as the memristor's resistance:
        state 1 (active)   = LOW  resistance  R_ON  = 1.8 kOhm
        state 0 (inactive) = HIGH resistance  R_OFF = 211 kOhm
and the cell works in two phases per time-step:
        READ  : a small pulse that cannot change the memristor is sent through it; the voltage that
                comes out (a voltage divider) is high if R is low -> neighbours "hear" the state.
        WRITE : a bigger voltage is applied; if it is above a threshold the memristor switches.

The paper simulates this in SPICE with a physical model.  Here we use the same IDEA with a
simplified *behavioural* model (also described in the paper: "linear ON and OFF states",
threshold-switching, bipolar):

    w  in [0, 1]  is the internal state   (1 = fully ON, 0 = fully OFF)
    resistance    = R_OFF + (R_ON - R_OFF) * w
    V > +V_S      -> w rises, with a saturating rate  alpha * (V - V_S) / (c + |V - V_S|)
    V < -V_RST    -> w falls, same shape
    in between    -> w barely moves (tiny linear drift)  -> the memory

It is NOT the exact SPICE equations of the paper (those have physical units we do not have).
"""
import numpy as np

R_ON, R_OFF = 1.8e3, 211e3            # ohms (values from the paper)
R_LOAD = float(np.sqrt(R_ON * R_OFF)) # load resistor of the read divider (geometric middle)
V_READ = 1.0                          # read pulse, volts; below the switching thresholds
V_S, V_RST = 1.5, 1.5                 # switching thresholds (|V|), volts
ALPHA, C_SAT, BETA = 1.2, 0.1, 0.01   # switching rate, softness, tiny drift below threshold


class MemristorArray:
    """Many memristors at once (one per CA cell).  `w` is a numpy array, one number per cell."""

    def __init__(self, n):
        self.w = np.zeros(n)                      # everything starts OFF

    @property
    def resistance(self):
        return R_OFF + (R_ON - R_OFF) * self.w    # ohms

    def read(self):
        """READ phase.  Voltage divider: V_out = V_READ * R_LOAD / (R_LOAD + R_memristor).
        Low resistance -> V_out near 0.92 V (logic 1);  high resistance -> near 0.08 V (logic 0).
        Returns a boolean array: True = cell is active.  Does NOT change w."""
        v_out = V_READ * R_LOAD / (R_LOAD + self.resistance)
        return v_out > 0.5 * V_READ

    def write(self, volts):
        """WRITE phase.  `volts` = voltage applied across each memristor (array).
        Positive and large -> SET (towards ON).  Negative and large -> RESET (towards OFF)."""
        v = np.asarray(volts, dtype=float)
        up = np.where(v > V_S, ALPHA * (v - V_S) / (C_SAT + np.abs(v - V_S)), 0.0)
        down = np.where(v < -V_RST, ALPHA * (v + V_RST) / (C_SAT + np.abs(v + V_RST)), 0.0)
        drift = np.where(np.abs(v) <= V_S, BETA * v, 0.0)
        self.w = np.clip(self.w + up + down + drift, 0.0, 1.0)
