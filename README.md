# TopoPK — Real-Data Demo

This repository contains the demo code accompanying the paper on **Topological Pharmacokinetics (TopoPK)**. It runs the full TopoPK pipeline on a real clinical PK dataset (15 MPA plasma concentration–time profiles): starting from sparsely sampled curves, it extracts topological and nonlinear-dynamics features and reproduces the signature figures used in the paper.

---

## Pipeline

1. **Preprocessing** — reads the `ID / Time / Conc` columns, anchors the time endpoints, and densifies each sparse profile to 400 points with LULD interpolation (linear on ascending segments, log-linear on descending ones).
2. **TopoPK feature extraction** — computes H0 persistence features (main/secondary peak birth, death, persistence, Δ*t*), permutation entropy, Betti-1, and nonlinear coupling indices.
3. **Late-peak characterization** — detects the second peak after the primary one and reports peak/valley positions, persistence, inter-peak time, valley-to-peak ratios, and the EHC (enterohepatic circulation) proxy AUC fraction derived from a log-linear counterfactual extrapolation.
4. **Visualization** — produces one figure per three profiles, laid out as concentration curve / normalized H0 diagram / delay-embedding phase plot, labeled Weak / Intermediate / Strong by late-peak persistence tertile.

---

## Requirements

```bash
pip install numpy pandas scipy matplotlib openpyxl
```

---

## Usage

```bash
python analyze_realdata.py
```

---

## Citation

If you use this code in your research, please cite:

> *Topological Pharmacokinetics: A Perspective on Shape-Based and Network-Aware Drug Profiling*
