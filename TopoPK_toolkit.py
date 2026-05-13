import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.spatial import distance
import warnings

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
            
        t1, t2 = t_sparse[idx], t_sparse[idx+1]
        C1, C2 = C_sparse[idx], C_sparse[idx+1]
        
        if C2 >= C1:
            slope = (C2 - C1) / (t2 - t1)
            C_dense[i] = C1 + slope * (t - t1)
        else:
            if C1 <= 0 or C2 <= 0:
                slope = (C2 - C1) / (t2 - t1)
                C_dense[i] = C1 + slope * (t - t1)
            else:
                slope = (np.log(C2) - np.log(C1)) / (t2 - t1)
                C_dense[i] = C1 * np.exp(slope * (t - t1))
                
    return t_dense, C_dense


class TopoPKAnalyzer:
    def __init__(self, t, C, tau=5, noise_ratio=0.05):
        self.t = np.asarray(t)
        self.C = np.asarray(C)
        self.tau = tau
        self.max_C = np.max(self.C) if np.max(self.C) > 0 else 1e-9
        self.threshold = noise_ratio * self.max_C
        self.features = {}
        self.ptp_indices = [] 
    def extract_all(self):
        self._extract_h0_and_pers()
        self._extract_entropy_pi()
        self._extract_phase_space_beta1()
        self._extract_dt_features() 
        self._extract_ptp() 
        return self.features

    def _extract_h0_and_pers(self):
        peak_idx, properties = find_peaks(self.C, prominence=self.threshold)
        proms = properties['prominences'] if len(peak_idx) > 0 else []
        

        if len(peak_idx) == 0 and self.C[0] >= np.max(self.C):
            peak_idx, proms = [0], [self.C[0]]
            
        self.proms = list(proms)
        self.peak_idx = peak_idx
        self.features['N_H0'] = len(peak_idx)
        
        if len(peak_idx) > 0:
            sorted_prom_idx = np.argsort(self.proms)[::-1]
            main_idx = sorted_prom_idx[0]
            
            self.features['Birth_Time_main'] = float(self.t[peak_idx[main_idx]])
            self.features['Birth_Level_main'] = float(self.C[peak_idx[main_idx]]) 
            self.features['Pers_main'] = float(self.proms[main_idx] / self.max_C)
    
            if len(peak_idx) >= 2:
                sub_idx = sorted_prom_idx[1]
                self.features['Birth_Time_sub'] = float(self.t[peak_idx[sub_idx]])
                self.features['Birth_Level_sub'] = float(self.C[peak_idx[sub_idx]]) 
                
                time_sorted_peaks = np.sort(peak_idx)
                self.features['Delta_t'] = float(self.t[time_sorted_peaks[1]] - self.t[time_sorted_peaks[0]])

                self.features['Pers_Ratio'] = float(self.proms[sub_idx] / self.proms[main_idx])
            else:
                self.features['Birth_Time_sub'] = 0.0
                self.features['Birth_Level_sub'] = 0.0
                self.features['Delta_t'] = 0.0
                self.features['Pers_Ratio'] = 0.0
        else:
            self.features['Birth_Time_main'] = 0.0
            self.features['Birth_Level_main'] = 0.0
            self.features['Pers_main'] = 0.0
            self.features['Birth_Time_sub'] = 0.0
            self.features['Birth_Level_sub'] = 0.0
            self.features['Delta_t'] = 0.0
            self.features['Pers_Ratio'] = 0.0

    def _extract_entropy_pi(self):
        self.features['Integral_PI'] = float(np.sum(self.proms) / self.max_C)
        if len(self.proms) > 0 and np.sum(self.proms) > 0:
            probs = np.array(self.proms) / np.sum(self.proms)
            self.features['Entropy_PE'] = float(-np.sum(probs * np.log2(probs + 1e-12)))
        else:
            self.features['Entropy_PE'] = 0.0

    def _extract_phase_space_beta1(self):
        if not HAS_RIPSER:
            self.features['Beta_1'] = -1
            self.features['Lambda_NL'] = -1.0
            return
        if len(self.C) <= self.tau:
            self.features['Beta_1'] = 0
            self.features['Lambda_NL'] = 0.0
            return
            
        X_raw = self.C[:-self.tau] / self.max_C
        Y_raw = self.C[self.tau:] / self.max_C
        dx, dy = np.diff(X_raw), np.diff(Y_raw)
        dists = np.sqrt(dx**2 + dy**2)
        u = np.insert(np.cumsum(dists), 0, 0)
        
        if u[-1] > 0:
            u_norm = u / u[-1]
            u_new = np.linspace(0, 1, 400)
            points_phase = np.column_stack([np.interp(u_new, u_norm, X_raw), np.interp(u_new, u_norm, Y_raw)])
        else:
            points_phase = np.column_stack([X_raw, Y_raw])
            
        dgms = ripser(points_phase, maxdim=1)['dgms']
        
        if len(dgms) > 1 and len(dgms[1]) > 0:
            h1_raw = dgms[1]
            valid_loops = h1_raw[(h1_raw[:, 1] - h1_raw[:, 0]) > 0.015]
            self.features['Beta_1'] = len(valid_loops)
        else:
            self.features['Beta_1'] = 0


        if points_phase.shape[0] > 2000:
            idx = np.random.choice(points_phase.shape[0], 2000, replace=False)
            P_sub = points_phase[idx]
        else:
            P_sub = points_phase
            
        pairwise_dists = distance.pdist(P_sub)
        diam = np.max(pairwise_dists) if len(pairwise_dists) > 0 else 0.0
 
        P_max_H1 = 0.0
        if len(dgms) > 1 and len(dgms[1]) > 0:
            h1_raw = dgms[1]
            finite_h1 = h1_raw[h1_raw[:, 1] < np.inf]
            if len(finite_h1) > 0:
                persistences = finite_h1[:, 1] - finite_h1[:, 0]
                P_max_H1 = np.max(persistences)
                
        if diam > 0:
            self.features['Lambda_NL'] = float(P_max_H1 / diam)
        else:
            self.features['Lambda_NL'] = 0.0

    def _extract_dt_features(self):
        if len(self.peak_idx) < 2:
            self.features['Decouple'] = 0.0
        else:
            p1, p2 = self.peak_idx[0], self.peak_idx[1]
            valley_idx = np.argmin(self.C[p1:p2]) + p1
            self.features['Decouple'] = float(self.C[valley_idx] / self.max_C)

        elim_start = self.peak_idx[-1] if len(self.peak_idx) > 0 else 0
        C_elim = self.C[elim_start:]
        if len(C_elim) <= self.tau:
            self.features['Dev'] = 0.0
            return
            
        x, y = C_elim[:-self.tau], C_elim[self.tau:]
        A, B = y[-1] - y[0], x[0] - x[-1]
        norm = np.sqrt(A**2 + B**2)
        if norm < 1e-5:
            self.features['Dev'] = 0.0
        else:
            distances = np.abs((A * x + B * y + (x[-1] * y[0] - x[0] * y[-1])) / norm)
            self.features['Dev'] = float(np.max(distances) / (np.max(x) - np.min(x)))

    def _extract_ptp(self):
        if len(self.C) <= self.tau + 5:
            self.features['N_PTP'] = 0
            return

        x = self.C[:-self.tau]
        y = self.C[self.tau:]
        
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        denominator = (dx**2 + dy**2)**1.5
        denominator[denominator == 0] = 1e-12 
        kappa = np.abs(dx * ddy - dy * ddx) / denominator
        
        margin = 3
        kappa[:margin] = 0
        kappa[-margin:] = 0
        
        threshold = max(1e-5, 0.05 * np.max(kappa))
        peaks, _ = find_peaks(kappa, prominence=threshold)
        
        self.features['N_PTP'] = len(peaks)
        self.ptp_indices = peaks


def plot_persistence_diagram(C, global_max, limit_max, threshold_ratio=0.05, ax=None):
    if ax is None: fig, ax = plt.subplots()

    prom_threshold = threshold_ratio * global_max
    peak_idx, properties = find_peaks(C, prominence=prom_threshold)
    h0_points = []
    global_min = np.min(C) 

    if len(peak_idx) == 0 and C[0] == np.max(C):
        h0_points.append((C[0], global_min)) 
    else:
        main_peak_idx_in_list = np.argmax(C[peak_idx])
        for i, (p, prom) in enumerate(zip(peak_idx, properties['prominences'])):
            if i == main_peak_idx_in_list:
                death = global_min
            else:
                death = C[p] - prom
            h0_points.append((C[p], death))

    ax.plot([0, limit_max], [0, limit_max], color='black', linestyle='-', alpha=0.3, lw=1)

    for birth, death in h0_points:
        ax.scatter(death, birth, c='#9b59b6', s=130, edgecolors='white', zorder=5)
        ax.plot([death, death], [birth, death], color='#9b59b6', linestyle=':', lw=1.5, alpha=0.6)

    ax.set_xlim([-limit_max * 0.05, limit_max])
    ax.set_ylim([-limit_max * 0.05, limit_max])
    ax.set_xlabel("Death", fontsize=11, fontweight='bold')
    ax.set_ylabel("Birth Level", fontsize=11, fontweight='bold') 
    ax.grid(True, linestyle=':', alpha=0.4)


def plot_phase_space(C, limit_max, features=None, tau=5, ax=None):
    if ax is None: fig, ax = plt.subplots()

    x_p, y_p = C[:-tau], C[tau:]
    peak_idx, _ = find_peaks(C, prominence=0.05 * np.max(C))
    elim_start_idx = peak_idx[-1] if len(peak_idx) > 0 else 0

    ax.plot([0, limit_max], [0, limit_max], color='gray', linestyle='--', alpha=0.4)
    ax.plot(x_p, y_p, color='#e74c3c',lw=2, alpha=0.5, label='Global')

    if len(C[elim_start_idx:]) > tau:
        x_elim, y_elim = C[elim_start_idx:-tau], C[elim_start_idx+tau:]
        ax.plot(x_elim, y_elim, color="#e74c3c", lw=2.5, label='Elimination')
    
    if features is not None and 'N_PTP' in features:
        ptp_indices = features.get('_ptp_indices', []) 
        if len(ptp_indices) > 0:
            for idx in ptp_indices:
                if idx < len(x_p):
                    ax.scatter(x_p[idx], y_p[idx], color='#f1c40f', s=200, marker='*', edgecolors='black', zorder=10)

    ax.set_xlim([-limit_max * 0.05, limit_max])
    ax.set_ylim([-limit_max * 0.05, limit_max])
    ax.set_xlabel("C(t)", fontsize=11, fontweight='bold')
    ax.set_ylabel("C(t+tau)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    if features is not None:
        dev_value = features.get('Dev', 0.0)
        n_ptp = features.get('N_PTP', 0)
        lambda_nl = features.get('Lambda_NL', 0.0)
        info_text = f"Dev: {dev_value:.3f}\nN_PTP: {n_ptp}\nλ_NL: {lambda_nl:.3f}"
        ax.text(limit_max * 0.05, limit_max * 0.85, info_text, 
                fontsize=9, fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=4))


def plot_topopk_dashboard(t, C, features=None, tau=5, title=None):
    global_max = np.max(C)
    limit_max = global_max * 1.15 
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    ax1 = axes[0]
    ax1.plot(t, C, color='#3498db', lw=2.5)
    ax1.fill_between(t, 0, C, color='#3498db', alpha=0.1)
    
    prom_threshold = 0.05 * global_max
    peak_idx, _ = find_peaks(C, prominence=prom_threshold)
    if len(peak_idx) == 0 and C[0] == np.max(C):
        peak_idx = [0]
    
    ax1.scatter(t[peak_idx], C[peak_idx], c='#e74c3c', s=70, edgecolors='white', zorder=5)

    for p_idx in peak_idx:
        ax1.axhline(y=C[p_idx], color='gray', linestyle='--', alpha=0.5, zorder=1)

  
    if features is not None:
        ptp_indices = features.get('_ptp_indices', [])
        for idx in ptp_indices:
            if idx < len(t):
                ax1.scatter(t[idx], C[idx], color='#f1c40f', s=150, marker='*', edgecolors='black', zorder=6)

    ax1.set_ylim([-limit_max * 0.05, limit_max]) 
    ax1.set_xlabel("Time", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Conc.", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)

    plot_persistence_diagram(C, global_max, limit_max, ax=axes[1])
    
    plot_phase_space(C, limit_max, features=features, tau=tau, ax=axes[2])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__": 
    t_patient = np.array([0, 0.5, 1, 2, 4, 8, 12, 24, 36, 48])
    C_patient = 800 * np.exp(-1.5 * t_patient) + 200 * np.exp(-0.05 * t_patient) 
    
    t_dense, C_dense = luld_interpolate(t_patient, C_patient, num_points=400)
    
    tau_value = 5
    analyzer = TopoPKAnalyzer(t_dense, C_dense, tau=tau_value)
    features = analyzer.extract_all()
    
    features['_ptp_indices'] = analyzer.ptp_indices 

    df_results = pd.DataFrame([features]).drop(columns=['_ptp_indices']).round(4)
    print("\nParamter：")
    print(df_results.T)
    
    plot_topopk_dashboard(t_dense, C_dense, features=features, tau=tau_value, title="IV 2-Comp Data TopoPK Signature")