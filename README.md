# OmniNeuro → memristive Cellular Automaton

EEG → three explainable "engines" (Physics, Chaos, Quantum-inspired) → safety protocol → a two-grid
**memristive cellular automaton** that shows a seizure-like state spreading from a pathogenic grid into a
healthy one. Sound, a Bloch-sphere picture and a cautious text report come out of the same decision.

![dashboard](outputs/dashboard.png)

*(Synthetic EEG demo. The seizure at 500-545 s is detected 4 s after onset; the movement artifact at 200 s is
rejected; the muscle burst at 350 s is ignored because only one engine reacts.)*

## Quick start
```
pip install numpy scipy matplotlib
python -m omnineuro demo                  # synthetic EEG, writes ./outputs/
python tests/run_tests.py                 # 20 checks
python scripts/ablation.py                # does each component help? (synthetic)
python -m omnineuro run data/chb01_03.edf --summary data/chb01-summary.txt     # real CHB-MIT file
```

## Where to start reading
1. **`docs/GUIDE.md`**: learn Python-for-this-project and every part, step by step, with checks.
2. **`steps/`**: six small scripts, one per component, each producing a picture.
3. **`docs/HONEST_LIMITS.md`**: what is and is not validated. Read it before trusting any number.

## Layout
```
omnineuro/
  config.py      every tunable number
  data.py        synthetic EEG, CHB-MIT loading (own EDF reader), seizure-time parsing
  engines.py     Physics (energy, Lidx), Chaos (Higuchi FD), phase sync, z-scores, fusion vote
  quantum.py     Bloch-sphere engine, P_move = sin²(θ/2)
  safety.py      artifact rules, ambiguity zone 0.4-0.6
  memristor.py   software memristor (read / write phases)
  ca.py          two-grid memristive CA, driven by an epilepsy level s in [0, 1]
  pipeline.py    analyse() and evaluate()
  outputs.py     dashboard, Bloch sphere, sonification (Table 1), report
steps/  tests/  scripts/  docs/  data/  outputs/
```

## Status
Research prototype. Works on synthetic EEG. **First real-data runs (CHB-MIT chb01) were poor**: the seizure was missed with default settings and one seizure-free hour produced false alarms from slow drift. Details, and what was changed in response, are in `docs/HONEST_LIMITS.md` section 7.
Not a medical device. Not included: the motor-imagery (PhysioNet) branch of the original OmniNeuro paper.
