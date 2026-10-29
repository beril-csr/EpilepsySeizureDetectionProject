"""
Command line:
    python -m omnineuro demo                                   synthetic EEG, nothing to download
    python -m omnineuro run FILE.edf --summary chb01-summary.txt
    python -m omnineuro evaluate FOLDER                        every .edf in FOLDER that has a summary file
"""
import argparse
import os
import sys

import numpy as np

from . import data as D
from . import outputs
from .config import Config
from .pipeline import analyse, evaluate


def _print_metrics(m):
    for k, v in m.items():
        print(f"  {k:24s} {v:.3f}" if isinstance(v, float) else f"  {k:24s} {v}")


def _save_all(res, out, seizures, tag=""):
    os.makedirs(out, exist_ok=True)
    outputs.dashboard(res, f"{out}/dashboard{tag}.png", seizures)
    outputs.bloch_figure(res, f"{out}/bloch{tag}.png")
    clip = None
    if seizures:
        clip = (max(seizures[0][0] - 60, 0), seizures[0][1] + 60)
    outputs.sonify(res, f"{out}/sonification{tag}.wav", clip=clip)
    text = outputs.make_report(res, use_llm=True)
    open(f"{out}/report{tag}.txt", "w", encoding="utf-8").write(text)
    print("\n" + text + f"\n\nSaved dashboard, bloch sphere, sonification and report in ./{out}/")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omnineuro")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo");     d.add_argument("--out", default="outputs")
    r = sub.add_parser("run");      r.add_argument("edf"); r.add_argument("--summary"); r.add_argument("--out", default="outputs")
    r.add_argument("--left", default="F3-C3"); r.add_argument("--right", default="F4-C4")
    r.add_argument("--baseline", type=float, default=300.0)
    r.add_argument("--rolling", action="store_true", help="adaptive baseline (last 10 min) instead of the first 5 min")
    r.add_argument("--w-chaos", type=float, default=1.0, help="weight of the chaos vote (0 = energy only)")
    e = sub.add_parser("evaluate"); e.add_argument("folder"); e.add_argument("--left", default="F3-C3")
    e.add_argument("--right", default="F4-C4"); e.add_argument("--baseline", type=float, default=300.0)
    e.add_argument("--rolling", action="store_true", help="adaptive baseline (last 10 min) instead of the first 5 min")
    e.add_argument("--w-chaos", type=float, default=1.0, help="weight of the chaos vote (0 = energy only)")
    a = ap.parse_args(argv)

    if a.cmd == "demo":
        print("Running on SYNTHETIC EEG (a seizure 500-545 s, a movement artifact at 200 s, a muscle burst at 350 s).")
        x, fs = D.make_synthetic()
        res = analyse(x, fs, Config())
        seiz = [(500.0, 545.0)]
        _print_metrics(evaluate(res, seiz))
        _save_all(res, a.out, seiz)
    elif a.cmd == "run":
        cfg = Config(baseline_s=a.baseline, rolling_baseline=a.rolling, w_chaos=a.w_chaos)
        x, fs, names = D.load_chbmit(a.edf, a.left, a.right)
        print(f"Loaded {a.edf}: channels {names}, {fs:.0f} Hz, {x.shape[1] / fs:.0f} s")
        seiz = D.parse_summary(a.summary, os.path.basename(a.edf)) if a.summary else []
        if any(s0 < cfg.baseline_s for s0, _ in seiz):
            print(f"WARNING: a seizure starts inside the {cfg.baseline_s:.0f} s baseline. Use a file whose "
                  "first minutes are seizure-free, or lower --baseline.")
        res = analyse(x, fs, cfg)
        if seiz:
            _print_metrics(evaluate(res, seiz))
        _save_all(res, a.out, seiz or None, tag="_" + os.path.splitext(os.path.basename(a.edf))[0])
    else:
        cfg = Config(baseline_s=a.baseline, rolling_baseline=a.rolling, w_chaos=a.w_chaos)
        files = sorted(f for f in os.listdir(a.folder) if f.endswith(".edf"))
        summ = [f for f in os.listdir(a.folder) if f.endswith("-summary.txt")]
        if not files or not summ:
            sys.exit("Need .edf files AND the chbXX-summary.txt in the folder.")
        tot = dict(n=0, hit=0, fa=0, hours=0.0, lat=[], ep=0, mins=0.0)
        for f in files:
            try:
                seiz = D.parse_summary(os.path.join(a.folder, summ[0]), f)
            except ValueError:                       # e.g. a duplicate download called "chb01_03 (1).edf"
                print(f"{f}: not listed in the summary file, skipped")
                continue
            x, fs, _ = D.load_chbmit(os.path.join(a.folder, f), a.left, a.right)
            res = analyse(x, fs, cfg, run_ca=False)
            m = evaluate(res, seiz)
            tot["n"] += len(seiz); tot["hit"] += int(round(m["sensitivity"] * len(seiz))) if seiz else 0
            tot["fa"] += m["false_alarms"]; tot["ep"] += m["false_alarm_episodes"]; tot["mins"] += m["false_alarm_minutes"]; tot["hours"] += (res.n * cfg.win_s - cfg.baseline_s) / 3600
            tot["lat"] += m["latency_s"]
            print(f"{f}: seizures={len(seiz)} sens={m['sensitivity']:.2f} false-alarm events={m['false_alarms']} "
                  f"episodes={m['false_alarm_episodes']} minutes={m['false_alarm_minutes']:.1f} latency={m['latency_s']}")
        print(f"\nTOTAL: {tot['hit']}/{tot['n']} seizures detected; false alarms: {tot['fa']} events = {tot['ep']} episodes = "
              f"{tot['mins']:.1f} min, in {tot['hours']:.1f} h "
              f"({tot['fa'] / max(tot['hours'], 1e-9):.2f}/h), mean latency "
              f"{np.mean(tot['lat']) if tot['lat'] else float('nan'):.1f} s")


if __name__ == "__main__":
    main()
