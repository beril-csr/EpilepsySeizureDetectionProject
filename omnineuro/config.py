"""
config.py  -  every tunable number in the project lives here.

WHY ONE FILE?  To answer "what if the window were 4 s instead of 2 s?" you change ONE number here
instead of hunting through the code.

A `dataclass` is a Python class that is mostly a bag of named values:
    Config()               -> all defaults
    Config(win_s=4.0)      -> defaults, except win_s

NOTE: this file was REBUILT.  The original config.py was accidentally overwritten while the project
was being completed.  The fields the other modules read (w_energy, w_chaos, bias, slope, use_phase,
amb_lo, amb_hi, max_abs_uv, min_var_uv2, max_emg_ratio) were kept, but the VALUES are new and were
re-tuned on the synthetic data in data.make_synthetic().
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ---------------------------------------------------------------- windows
    win_s: float = 2.0            # length of each analysis window, seconds (no overlap)

    # --------------------------------------------------------- preprocessing
    # Your document uses 8-30 Hz (motor imagery lives there).  Seizure rhythms live mostly
    # BELOW 8 Hz, so the band is widened to 1-40 Hz here.
    band_lo: float = 1.0
    band_hi: float = 40.0

    # --------------------------------------------------------------- engines
    kmax: int = 10                # Higuchi k_max: the largest step size k
    eps: float = 1e-6             # the epsilon of Eq.(1); data are in uV^2 so this is tiny

    # ----------------------------------------------- baseline ("what is normal?")
    baseline_s: float = 300.0     # the first N seconds define this patient's "normal"
                                  # (must be seizure-free; artifact windows are skipped)

    # Rolling baseline (option): "normal" = the median of the last ref_s seconds, ending gap_s
    # seconds ago.  Slow drift over an hour is absorbed; a 40 s seizure barely moves a 10 min median.
    rolling_baseline: bool = False
    ref_s: float = 600.0
    gap_s: float = 30.0

    # ----------------------------------------------------------- fusion (vote)
    w_energy: float = 1.0         # weight of the Physics vote
    w_chaos: float = 1.0          # weight of the Chaos vote
    z_cap: float = 6.0            # each engine's z-score is capped at +-z_cap BEFORE voting, so one
                                  # engine alone can never exceed bias (=3) -> at best "UNCLEAR".
    bias: float = 3.0             # evidence (in z-scores) at which raw probability = 0.5
    slope: float = 1.0            # how sharply probability rises around the bias

    # ------------------------------------------------------------ quantum engine
    memory: float = 0.5           # 0 = no smoothing; nearer 1 = smoother but laggier
    use_phase: bool = True        # False = plain smoothing (used for the ablation test)

    # ------------------------------------------------------------ safety protocol
    amb_lo: float = 0.4           # ambiguity zone from section 2.4 of the document
    amb_hi: float = 0.6
    max_abs_uv: float = 400.0     # a peak bigger than this is movement / electrode pop
    min_var_uv2: float = 0.01     # variance below this = flat line (electrode off)
    max_emg_ratio: float = 0.30   # share of power in 30-55 Hz above this = muscle noise

    # ---------------------------------------------------------------- CA model
    grid: int = 25                # each grid is grid x grid cells (paper: 25 x 25)
    center: int = 12              # central block that is linked between grids (paper: 12 x 12)
    steps_per_window: int = 10    # CA time-steps simulated for each EEG window
    seed: int = 0                 # random seed -> identical results on every run
