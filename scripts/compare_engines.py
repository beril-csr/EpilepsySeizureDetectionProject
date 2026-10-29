"""
Does the CHAOS drop signal seizures?  Energy alone vs chaos alone vs both.

Usage (from the project folder):
    python scripts/compare_engines.py ..\\data\\chb01_03.edf --summary ..\\data\\chb01-summary.txt
    python scripts/compare_engines.py FILE1.edf FILE2.edf --summary chb01-summary.txt

For every file it prints, for each score, the AUC = the chance that a randomly chosen SEIZURE window
scores higher than a randomly chosen NORMAL window:
    0.50 = no better than a coin flip      1.00 = perfect separation      below 0.50 = points the WRONG way
Only windows after the baseline period and not flagged as artifacts are used.
Seizure times come from the summary file and are used ONLY to score, never by the detector.

Read the numbers carefully: one seizure is about 20 windows, so a single file is weak evidence.
Use several files, and report the spread, not just the best one.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from omnineuro import Config, analyse
from omnineuro import data as D
from omnineuro.pipeline import auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edf", nargs="+")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--left", default="F3-C3")
    ap.add_argument("--right", default="F4-C4")
    ap.add_argument("--rolling", action="store_true", help="adaptive baseline (last 10 min)")
    a = ap.parse_args()
    cfg = Config(rolling_baseline=a.rolling)

    rows = []
    for path in a.edf:
        name = os.path.basename(path)
        seizures = D.parse_summary(a.summary, name)
        if not seizures:
            print(f"{name}: no seizure listed in the summary, skipped (use `evaluate` for false alarms)")
            continue
        x, fs, _ = D.load_chbmit(path, a.left, a.right)
        res = analyse(x, fs, cfg, run_ca=False)
        lab = np.zeros(res.n, bool)
        for s0, s1 in seizures:
            lab |= (res.t0 < s1) & (res.t0 + cfg.win_s > s0)
        use = (res.t0 >= cfg.baseline_s) & ~res.artifact
        if lab[use].sum() == 0:
            print(f"{name}: the seizure falls inside the baseline or artifact windows, skipped")
            continue
        scores = {
            "energy rise only": res.z_energy,
            "chaos drop only": res.z_chaos,
            "both (fused evidence)": res.evidence,
            "P_move (after quantum)": res.p_move,
        }
        r = {k: auc(v[use], lab[use]) for k, v in scores.items()}
        rows.append((name, int(lab[use].sum()), r))
        print(f"\n{name}: {int(lab[use].sum())} seizure windows vs {int((~lab[use]).sum())} normal windows")
        for k, v in r.items():
            print(f"  {k:24s} AUC = {v:.3f}")

    if len(rows) > 1:
        print("\nAverage over files:")
        for k in rows[0][2]:
            vals = [r[2][k] for r in rows]
            print(f"  {k:24s} mean AUC = {np.mean(vals):.3f}   (min {np.min(vals):.3f}, max {np.max(vals):.3f})")


if __name__ == "__main__":
    main()
