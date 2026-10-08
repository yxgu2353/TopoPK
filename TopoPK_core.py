
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import distance

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    warnings.warn("未安装 ripser 库，Beta_1 和 Lambda_NL 的计算将返回 -1")


def luld_interpolate(t_sparse, C_sparse, num_points=400):
    t_dense = np.linspace(np.min(t_sparse), np.max(t_sparse), num_points)
    C_dense = np.zeros_like(t_dense)

    for i, t in enumerate(t_dense):
        idx = np.searchsorted(t_sparse, t) - 1

        if idx < 0:
            C_dense[i] = C_sparse[0]
            continue
        if idx >= len(t_sparse) - 1:
            C_dense[i] = C_sparse[-1]
            continue

        t1, t2 = t_sparse[idx], t_sparse[idx + 1]
        C1, C2 = C_sparse[idx], C_sparse[idx + 1]

        if C2 >= C1 or C1 <= 0 or C2 <= 0:
            C_dense[i] = C1 + (C2 - C1) / (t2 - t1) * (t - t1)
        else:
            C_dense[i] = C1 * np.exp((np.log(C2) - np.log(C1)) / (t2 - t1) * (t - t1))

    return t_dense, C_dense


class TopoPKAnalyzer:
    def __init__(self, t, C, tau=5, noise_ratio=0.05):
        self.t = np.asarray(t)
        self.C = np.asarray(C)
        self.tau = tau
        self.max_C = np.max(self.C) if np.max(self.C) > 0 else 1e-9
        self.threshold = noise_ratio * self.max_C
        self.features = {}

    def extract_all(self):
        self._extract_h0_and_pers()
        self._extract_entropy_pi()
        self._extract_phase_space_beta1()
        self._extract_dt_features()
        self._extract_ptp()
        return self.features

    def _extract_h0_and_pers(self):
        f = self.features
        peak_idx, properties = find_peaks(self.C, prominence=self.threshold)
        proms = list(properties['prominences']) if len(peak_idx) > 0 else []

        if len(peak_idx) == 0 and self.C[0] >= np.max(self.C):
            peak_idx, proms = np.array([0]), [self.C[0]]

        self.proms, self.peak_idx = proms, peak_idx
        f['N_H0'] = len(peak_idx)

        f.update({
            'Birth_Time_main': 0.0, 'Birth_Level_main': 0.0, 'Pers_main': 0.0,
            'Birth_Time_sub': 0.0, 'Birth_Level_sub': 0.0, 'Delta_t': 0.0, 'Pers_Ratio': 0.0,
        })
        if len(peak_idx) == 0:
            return

        order = np.argsort(proms)[::-1]
        main = order[0]
        f['Birth_Time_main'] = float(self.t[peak_idx[main]])
        f['Birth_Level_main'] = float(self.C[peak_idx[main]]) 
        f['Pers_main'] = float(proms[main] / self.max_C)

        if len(peak_idx) >= 2:
            sub = order[1]
            t_sorted = np.sort(peak_idx)
            f['Birth_Time_sub'] = float(self.t[peak_idx[sub]])
            f['Birth_Level_sub'] = float(self.C[peak_idx[sub]])
            f['Delta_t'] = float(self.t[t_sorted[1]] - self.t[t_sorted[0]])
            f['Pers_Ratio'] = float(proms[sub] / proms[main])

    def _extract_entropy_pi(self):
        f = self.features
        total = np.sum(self.proms)
        f['Integral_PI'] = float(total / self.max_C)
        if len(self.proms) > 0 and total > 0:
            probs = np.array(self.proms) / total
            f['Entropy_PE'] = float(-np.sum(probs * np.log2(probs + 1e-12)))
        else:
            f['Entropy_PE'] = 0.0

    def _extract_phase_space_beta1(self):
        f = self.features
        if not HAS_RIPSER:
            f['Beta_1'], f['Lambda_NL'] = -1, -1.0
            return
        if len(self.C) <= self.tau:
            f['Beta_1'], f['Lambda_NL'] = 0, 0.0
            return

        X, Y = self.C[:-self.tau] / self.max_C, self.C[self.tau:] / self.max_C
        u = np.insert(np.cumsum(np.sqrt(np.diff(X) ** 2 + np.diff(Y) ** 2)), 0, 0)
        if u[-1] > 0:
            u_new = np.linspace(0, 1, 400)
            P = np.column_stack([np.interp(u_new, u / u[-1], X), np.interp(u_new, u / u[-1], Y)])
        else:
            P = np.column_stack([X, Y])

        dgms = ripser(P, maxdim=1)['dgms']

        h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
        f['Beta_1'] = int(np.sum((h1[:, 1] - h1[:, 0]) > 0.015)) if len(h1) else 0

        finite = h1[h1[:, 1] < np.inf] if len(h1) else h1
        p_max = np.max(finite[:, 1] - finite[:, 0]) if len(finite) else 0.0
        diam = np.max(distance.pdist(P)) if len(P) > 1 else 0.0
        f['Lambda_NL'] = float(p_max / diam) if diam > 0 else 0.0

    def _extract_dt_features(self):
        f = self.features
        if len(self.peak_idx) < 2:
            f['Decouple'] = 0.0
        else:
            p1, p2 = self.peak_idx[0], self.peak_idx[1]
            valley = np.argmin(self.C[p1:p2]) + p1
            f['Decouple'] = float(self.C[valley] / self.max_C)

        C_elim = self.C[self.peak_idx[-1] if len(self.peak_idx) > 0 else 0:]
        if len(C_elim) <= self.tau:
            f['Dev'] = 0.0
            return
        x, y = C_elim[:-self.tau], C_elim[self.tau:]
        A, B = y[-1] - y[0], x[0] - x[-1]
        norm = np.sqrt(A ** 2 + B ** 2)
        if norm < 1e-5:
            f['Dev'] = 0.0
        else:
            d = np.abs((A * x + B * y + (x[-1] * y[0] - x[0] * y[-1])) / norm)
            f['Dev'] = float(np.max(d) / (np.max(x) - np.min(x)))

    def _extract_ptp(self):
        if len(self.C) <= self.tau + 5:
            self.features['N_PTP'] = 0
            return

        x, y = self.C[:-self.tau], self.C[self.tau:]
        dx, dy = np.gradient(x), np.gradient(y)
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        denom = (dx ** 2 + dy ** 2) ** 1.5
        denom[denom == 0] = 1e-12
        kappa = np.abs(dx * ddy - dy * ddx) / denom

        kappa[:3] = 0
        kappa[-3:] = 0

        peaks, _ = find_peaks(kappa, prominence=max(1e-5, 0.05 * np.max(kappa)))
        self.features['N_PTP'] = len(peaks)
