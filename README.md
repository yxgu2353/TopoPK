# TopoPK — Topological Pharmacokinetics Toolkit

A Python toolkit for analyzing pharmacokinetic (PK) time-concentration curves using topological data analysis (TDA).

## Overview

TopoPK extracts shape-based topological features from PK curves to characterize drug absorption, distribution, and elimination patterns. It combines persistent homology, phase space reconstruction, and curvature analysis to provide a quantitative signature of PK profiles.

## Features

| Feature | Description |
|---------|-------------|
| **H0 Persistence** | Peak detection and persistence (prominence) of concentration peaks |
| **Entropy (PE)** | Shannon entropy of peak prominence distribution |
| **Integral (PI)** | Normalized sum of peak prominences |
| **Pers Ratio** | Persistence ratio between primary and secondary peaks |
| **Delta_t** | Time interval between major peaks |
| **Decouple** | Valley depth between absorption and elimination phases |
| **Dev** | Phase space deviation from linear elimination trajectory |
| **N_PTP** | Number of peak turning points in phase space curvature |
| **Beta_1** | H1 persistent homology — number of loop structures in phase space |
| **Lambda_NL** | Normalized maximum persistence in H1, reflecting nonlinearity |

## Dependencies

- `numpy`, `pandas`, `matplotlib`
- `scipy`
- `ripser` (optional, for Beta_1 and Lambda_NL)

## Quick Start

```python
import numpy as np
from TopoPK_toolkit import TopoPKAnalyzer, luld_interpolate, plot_topopk_dashboard

# Sparse PK samples
t = np.array([0, 0.5, 1, 2, 4, 8, 12, 24, 36, 48])
C = 800 * np.exp(-1.5 * t) + 200 * np.exp(-0.05 * t)

# Interpolate with LULD method
t_dense, C_dense = luld_interpolate(t, C, num_points=400)

# Extract topological features
analyzer = TopoPKAnalyzer(t_dense, C_dense, tau=5)
features = analyzer.extract_all()

# Visualize
plot_topopk_dashboard(t_dense, C_dense, features=features, tau=5)
```

## Output

The dashboard visualization includes three panels:
1. **PK curve** with detected peaks and turning points
2. **Persistence diagram** (H0 birth-death plot)
3. **Phase space** (C(t) vs C(t+τ)) with deviation and curvature markers
