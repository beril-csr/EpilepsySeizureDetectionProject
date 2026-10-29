# Honest limits: read this before you believe any number

## 1. What has and has not been checked

| Checked | How |
|---|---|
| The code runs end to end and the maths does what the comments say (HFD, Lidx, Bloch formula, refractory period, memristor switching, EDF parsing, summary parsing) | 20 automated tests (`python tests/run_tests.py`); the tests were also confirmed to fail when the code is deliberately broken |
| The pipeline finds a seizure in **synthetic** EEG | demo + `scripts/ablation.py` (8 random recordings) |

| **NOT** checked | Why it matters |
|---|---|
| **Real EEG beyond three files of one patient.** | See section 7: the first real runs (CHB-MIT chb01_01, chb01_02, chb01_03, run by the project owner) did **not** work well. Nothing is known about other patients. |
| `read_edf` on a real CHB-MIT file | The real runs used `mne`. The built-in reader has only been tested on a small EDF file written by the test itself. |
| The optional Gemini report | Needs a key and internet; the call was never executed. It falls back to the local report on any error. |
| Thresholds (`bias=3`, `z_cap=6`, `vote_z`, artifact limits) | Hand-chosen on synthetic data. Real thresholds must be tuned on training files and measured on **different** files. |

## 2. Claims in the document that this project does **not** support

1. **"Zero-lag filter."** The quantum filter is a leaky average: it **delays** the decision. Measured: latency 4.2 s with it vs 2.0 s without (`scripts/ablation.py`). It removes flicker (see step 4), at the cost of lag.
2. **"Interference gives an advantage."** Switching off the phase term (`use_phase=False`) changed nothing measurable on clean data, and made detection 1.5 s *faster* on noisy data. The smoothing helps; the quantum phase part is not shown to. Test this on real data before claiming it.
3. **"The quantum engine senses seizure synchrony."** What is actually computed is the phase coherence between the two hemispheres (`engines.phase_sync`). Rhythmic activity raises it, but a strong alpha rhythm does too.
4. **"Model the seizure in the CA and visualise the pathway non-invasively."** The CA is driven by **one number** (`s`) from the scalp EEG. It does not localise the focus or the route of spread; the "pathogenic" grid is abstract. The CA is an *illustrative tissue model that reacts to the detector*. The only anatomical information is `Lidx` (which hemisphere has more energy). Real propagation mapping would need many channels and source localisation, not this.
5. **"Energy up, chaos down" detects seizures in general.** Plausible and consistent with literature, but your own note says chaos can rise microscopically and the direction depends on channels and seizure type. It must be verified on labelled data.
6. **The 58.52 % motor-imagery accuracy** from the original OmniNeuro paper: this project does not touch it. Motor-imagery decoding on PhysioNet (the first idea in your summary: EEGNet + OmniNeuro + a CA behind it) is **not implemented**; only the seizure branch is. The Physics engine (`log_energy_ratio`) is the same Eq. (1), so adding it would be a natural next step.

## 3. Known weaknesses (measured on synthetic data)

| Situation | Result |
|---|---|
| A strong 10 Hz alpha burst (+10 or +20 µV) | **false alarms** (0.4-0.5 per recording): alpha looks like "energy up, complexity down". Adding frequency information (e.g. a theta/alpha ratio) would help. |
| Movement artifact (huge amplitude) | rejected by `safety.artifact_reason` |
| Muscle-like burst (energy ↑ but complexity ↑) | not flagged as artifact, but **never** raises an alarm, because only one engine votes yes |
| Noisier recording | still detected, with more state flips and 1.3 s more latency |

Other design limits:
- **Baseline drift.** "Normal" is fixed from the first 5 minutes. Over an hour (sleep, medication, posture) real EEG drifts and would cause false alarms. A rolling baseline that ignores flagged windows is the obvious upgrade.
- **Two channels.** Hemisphere comparison uses one channel per side. More channels would be more robust.
- **Offline filter.** `bandpass` is zero-phase (looks at the future as well as the past). Use `causal=True` for a real-time version.
- **Autoencoder replaced.** The document's artifact veto is an autoencoder. An autoencoder trained on normal EEG also flags seizures as abnormal, so it would veto the very thing we want to detect. Simple physical rules are used instead.
- **"Adaptive vote"** is implemented as a z-score cap plus a weighted average; it is not learned.
- **The memristor** is a behavioural threshold model, not the SPICE equations of the paper.
- **Epilepsy level** `s` is smoothed and held during unclear windows, so the CA lags the detector by a window or two.

## 4. How to make a real claim (the checklist)

1. Pick patients/files for **tuning** and different ones for **testing**. Never tune on the test files.
2. Report **per seizure**: sensitivity, latency, and **false alarms per hour** (not "accuracy": almost every window is normal, so a detector that never fires has 99 % accuracy).
3. Compare with simple baselines (energy alone, line length, a plain threshold) to see whether the Chaos / Quantum parts add anything.
4. Run the ablation (`scripts/ablation.py`) on real data, especially `use_phase=False`.
5. Treat the output as decision support for a clinician, as your document itself says. This is a research prototype, not a medical device.

## 5. What changed from the earlier notebook (bugs that are fixed)

- Channels: it compared `FP1-F7` with `F7-T7` (both left side). Now `F3-C3` vs `F4-C4` (left vs right).
- Units: data were in volts with ε = 1e-6, which dominated the variance (1e-10). Data are converted to µV, so ε is tiny.
- The first file used (`chb01_01`) has no seizure. Use `chb01_03` (seizure at 2996-3036 s).
- The detector now compares each window with a **baseline**, instead of `|Lidx| + |ΔHFD|` (which only measured left/right difference, not seizure-likeness).
- The CA is no longer a flat 0 or a latched 625: it has random wiring, a refractory period, a seed region and a real read/write memristor.
- The Gemini prompt no longer contains the true label; the API key is read from the environment, not typed into the code.

## 6. A note about this folder's history

This project continues partly written files (`data.py`, `engines.py`, `quantum.py`, `safety.py`) found in the folder. While completing it the original `config.py` was **accidentally overwritten**. It was rebuilt from the settings the other files read; the field names are preserved but the values are new. `engines.fuse` also received one change (the z-score cap), and `data.py` gained `read_edf`. If you have the original `config.py` somewhere else, compare it before relying on these defaults.

## 7. Real-data results so far (CHB-MIT, patient chb01, default settings)

Run by the project owner on their own computer, with the original fixed 5-minute baseline. Seizure times come from `chb01-summary.txt`.

| File | What was measured | Result |
|---|---|---|
| chb01_03 (1 seizure, 2996-3036 s) | AUC, seizure windows vs normal windows (19 vs 1630) | energy rise only **0.947**; chaos (complexity) drop only **0.484**; both fused **0.707**; final P_move **0.712** |
| chb01_03 | alarms | **0 of 1 seizures detected** (no alarm raised) |
| chb01_01 (no seizure) | false alarms | 0 |
| chb01_02 (no seizure) | false alarms | **51 alarm events**, mostly one fragmented stretch of roughly 15 minutes late in the hour, during which the energy and complexity scores both stayed elevated: a slow drift of the recording, not a seizure |

What this does and does not show:
- On this one seizure, the **energy rise** separated seizure windows well, but the **chaos drop did not** (AUC about 0.5). This is evidence against the document's hypothesis *for this recording and these two channels*. One seizure (19 windows) is weak evidence either way.
- Because chaos was not informative, fusing it with energy made the detector worse than energy alone.
- The false alarms in chb01_02 came from the fixed baseline: after a slow change the end of the hour no longer resembled the first five minutes.
- AUC measures ranking only. It says nothing about how often energy alone would give false alarms.

### What was changed after seeing these results
A **rolling baseline** (`--rolling`, `Config(rolling_baseline=True)`) was added: "normal" is the median of the last 10 minutes, ending 30 s ago. It had been listed as an upgrade before real data was available. On a synthetic lasting shift it cut false-alarm time from 26 to about 5 minutes without losing the seizure (`python scripts/ablation.py`), but it does not remove it: the rolling reference needs roughly half its length to absorb a shift. A shorter reference helps more but risks absorbing a long seizure. `evaluate` now also reports false-alarm **minutes** and merged **episodes**, because counting raw events lets flicker inflate the number.

### Result with the rolling baseline (same three files, run by the project owner)

| File | Result |
|---|---|
| chb01_01 (no seizure) | 0 false-alarm events, 0 minutes |
| chb01_02 (no seizure) | 0 false-alarm events, 0 minutes (was 51 events) |
| chb01_03 (1 seizure) | detected; alarm came **30 s after onset** (the seizure lasts 40 s) |
| Total | 1/1 seizures, 0 false alarms in 2.8 h |

How much to trust it:
- These are the **same files used to diagnose the problem**, so this is a development result, not a test. The fix was chosen after looking at chb01_02.
- One seizure from one patient. Zero false alarms in 2.8 h only bounds the true false-alarm rate at roughly 1 per hour (rule of three: 3 / 2.8 h), not at zero.
- A 30 s latency on a 40 s seizure means the alarm arrives when the seizure is almost over. That is a weak result for any warning purpose, even though it counts as a detection.
- Why chb01_03 flipped from missed to detected has not been separated from chance, the baseline, or both engines changing; run `scripts/compare_engines.py ... --rolling` to see how each engine behaves with the new baseline.

### Engine comparison on chb01_03 with the rolling baseline (run by the project owner)

| Score | AUC with fixed baseline | AUC with rolling baseline |
|---|---|---|
| energy rise only | 0.947 | 0.997 |
| chaos (complexity) drop only | 0.484 | 0.770 |
| both fused | 0.707 | 0.993 |
| final P_move | 0.712 | 0.998 |

With a baseline that follows the recording, the chaos drop carries *some* signal (0.77) where it appeared to carry none. It is still much weaker than energy, and fusing it did not beat energy alone (0.993 vs 0.997). One seizure only.

### First files not seen during development (rolling baseline, default settings)

| File | Seizure | Result |
|---|---|---|
| chb01_04 | 1467-1494 s (27 s) | **missed** |
| chb01_15 | 1732-1772 s (40 s) | **missed** |
| both | | 0 of 2 detected; 0 false alarms |

Together with chb01_03 that is **1 of 3 seizures detected** (the one detection came 30 s after onset, with a 40 s seizure). The success on chb01_03 did not generalise to the next two seizures. Zero false alarms is easy for a detector that rarely fires, so it is not evidence of good behaviour here. Not yet known: whether the ranking signal exists in chb01_04/15 and the alarm threshold is simply too strict, or whether the features do not separate these seizures at all (`scripts/compare_engines.py` on those files answers this; chb01_04 and chb01_15 are now development files too).

### Engine comparison on all three seizure files seen so far (rolling baseline, AUC)

| File | energy only | chaos only | both fused | P_move |
|---|---|---|---|---|
| chb01_03 | 0.997 | 0.770 | 0.993 | 0.998 |
| chb01_04 | 0.934 | 0.513 | 0.743 | 0.785 |
| chb01_15 | 0.922 | 0.764 | 0.882 | 0.905 |
| mean of the three | **0.951** | 0.682 | 0.873 | 0.896 |

Findings, for this patient and these two channels only:
- **Energy rise ranks seizure windows high in all three files (0.92-1.00).**
- **The complexity drop is weak and inconsistent (0.51-0.77).**
- **Fusing the two scored lower than energy alone in every file.** The document's hypothesis that a chaos drop adds a useful signal is not supported here.
- This also explains the misses: the fusion needs both engines to agree. When complexity does not move, energy alone is capped so that the combined evidence only reaches the ambiguity zone, and no alarm is issued, even though the energy score itself separates the seizure windows well. The design (not just the data) causes those misses.
- AUC compares windows; it does not measure alarms or false alarms. An energy-only detector gives up the one job the chaos vote had: on the synthetic muscle burst (energy up, complexity up) energy alone raises a false alarm, the two-engine vote does not. The final test measures that trade-off on real seizure-free hours.

### Planned final test (written down BEFORE it is run)
Files not yet opened: chb01_16, chb01_18, chb01_21, chb01_26. Two configurations, both with the rolling baseline and no further tuning:
- **A, the document's design:** `--rolling` (energy + chaos vote)
- **B, energy only:** `--rolling --w-chaos 0`

Both will be run once on the same four files and **both results reported, whatever they are**. Chb01_01 and chb01_02 (seizure-free) are re-run for false alarms with the same two configurations. Nothing is changed afterwards.

chb01_01, chb01_02 and chb01_03 are now **development files**: they have been looked at, so any result on them is no longer a clean test. A fair test needs seizure files that have not been looked at (other chb01 files, ideally other patients), run once with the settings frozen.

### Final test: results (run by the project owner, settings frozen, both reported)
Deviations from the plan, stated openly: `chb01_16` was not downloaded, and `chb01_01` was skipped because its file was saved as `chb01_01 (1).edf`. So the test covers **three** untouched seizure files (chb01_18, 21, 26) and **one** seizure-free file (chb01_02, which was already a development file), 3.3 h in total.

| Design | Seizures detected | Latency | False-alarm events / episodes (3.3 h) |
|---|---|---|---|
| A: document's design (energy + chaos), `--rolling` | **0 / 3** | n/a | 0 / 0 |
| B: energy only, `--rolling --w-chaos 0` | **2 / 3** (chb01_18, chb01_21) | 30 s, 29 s | 9 / 4 (0.4 min) |

Per file, B: chb01_02 (no seizure) 6 events / 3 episodes; chb01_18 detected, 3 events / 1 episode; chb01_21 detected, 0; chb01_26 missed under both designs.

What this does and does not show:
- On these files the document's two-engine design detected **no** seizures. Energy alone detected two of three, at the price of about four false-alarm episodes in 3.3 h (each very short).
- This is 3 seizures from one patient. "2 of 3" has a very wide uncertainty; it is not evidence that the detector works in general.
- Design B was chosen after looking at the development files, so these three files are a fair but tiny test of that choice, not proof.
- Only one seizure-free file was used for false alarms. The real false-alarm rate is not established.
- Not a medical device; nothing here supports clinical use.
