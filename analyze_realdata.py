
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks


BASE = Path(__file__).resolve().parents[1]
CASE_DIR = Path(__file__).resolve().parent
OUT = CASE_DIR / "outputs_simplified"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE))
from TopoPK_toolkit_update import TopoPKAnalyzer, luld_interpolate  # noqa: E402


matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
    }
)

TAU = 5
DEFAULT_PROM = 0.05
N_INTERP = 400
LATE_WINDOW_START_H = 4.0


def anchor_digitized_times(t: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Snap near-axis endpoints to 0 and 12 h, preserving interior values."""
    t = np.asarray(t, dtype=float).copy()
    notes: list[str] = []
    if 0 <= t[0] <= 0.06:
        notes.append(f"t_first {t[0]:.4f}->0")
        t[0] = 0.0
    if abs(t[-1] - 12.0) <= 0.15:
        notes.append(f"t_last {t[-1]:.4f}->12")
        t[-1] = 12.0
    if not np.all(np.diff(t) > 0):
        raise ValueError("Time values must be strictly increasing after anchoring")
    return t, notes


def auc(t: np.ndarray, c: np.ndarray) -> float:
    return float(np.trapezoid(c, t))


def detect_late_peak(
    t: np.ndarray,
    c: np.ndarray,
    prominence_ratio: float = DEFAULT_PROM,
    late_start: float = LATE_WINDOW_START_H,
) -> dict[str, float | int | bool]:
    """Track the primary peak, pre-late-peak valley, and strongest late peak.

    The late peak is a local maximum at or after ``late_start`` whose raw-data
    prominence is at least ``prominence_ratio * Cmax``.
    """
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    cmax = float(np.max(c))
    dyn = max(float(np.ptp(c)), 1e-12)

    early_candidates = np.where(t < late_start)[0]
    if len(early_candidates) == 0:
        early_candidates = np.arange(len(t))
    primary_idx = int(early_candidates[np.argmax(c[early_candidates])])

    peaks, props = find_peaks(c, prominence=0.0)
    prominences = props.get("prominences", np.array([], dtype=float))
    late_locs = [j for j, idx in enumerate(peaks) if t[idx] >= late_start and idx > primary_idx]

    result: dict[str, float | int | bool] = {
        "late_detected": False,
        "primary_idx": primary_idx,
        "primary_time": float(t[primary_idx]),
        "primary_conc": float(c[primary_idx]),
        "late_idx": -1,
        "late_time": np.nan,
        "late_conc": np.nan,
        "late_prominence": 0.0,
        "late_persistence_cmax": 0.0,
        "late_persistence_range": 0.0,
        "valley_idx": -1,
        "valley_time": np.nan,
        "valley_conc": np.nan,
        "interpeak_time": np.nan,
        "secondary_primary_ratio": np.nan,
        "valley_cmax_ratio": np.nan,
        "valley_lower_peak_ratio": np.nan,
        "ehc_auc_fraction": np.nan,
        "ehc_excess_auc_fraction": np.nan,
        "counterfactual_r2": np.nan,
    }
    if not late_locs:
        return result

    best_j = max(late_locs, key=lambda j: prominences[j])
    late_idx = int(peaks[best_j])
    late_prom = float(prominences[best_j])
    if late_idx - primary_idx < 2:
        return result

    valley_idx = int(primary_idx + np.argmin(c[primary_idx : late_idx + 1]))
    if valley_idx in (primary_idx, late_idx):
        return result

    result.update(
        {
            "late_detected": bool(late_prom >= prominence_ratio * cmax),
            "late_idx": late_idx,
            "late_time": float(t[late_idx]),
            "late_conc": float(c[late_idx]),
            "late_prominence": late_prom,
            "late_persistence_cmax": late_prom / max(cmax, 1e-12),
            "late_persistence_range": late_prom / dyn,
            "valley_idx": valley_idx,
            "valley_time": float(t[valley_idx]),
            "valley_conc": float(c[valley_idx]),
            "interpeak_time": float(t[late_idx] - t[primary_idx]),
            "secondary_primary_ratio": float(c[late_idx] / max(c[primary_idx], 1e-12)),
            "valley_cmax_ratio": float(c[valley_idx] / max(cmax, 1e-12)),
            "valley_lower_peak_ratio": float(
                c[valley_idx] / max(min(c[primary_idx], c[late_idx]), 1e-12)
            ),
        }
    )

    total_auc = auc(t, c)
    result["ehc_auc_fraction"] = auc(t[valley_idx:], c[valley_idx:]) / max(total_auc, 1e-12)

    # Exploratory counterfactual: continue the pre-valley log-linear decline.
    fit_idx = np.arange(primary_idx + 1, valley_idx + 1)
    fit_idx = fit_idx[c[fit_idx] > 0]
    if len(fit_idx) >= 3:
        slope, intercept, r, _, _ = stats.linregress(t[fit_idx], np.log(c[fit_idx]))
        result["counterfactual_r2"] = float(r * r)
        if slope < 0:
            td, cd = luld_interpolate(t, c, num_points=N_INTERP)
            mask = td >= t[valley_idx]
            baseline = np.exp(intercept + slope * td[mask])
            excess = np.maximum(cd[mask] - baseline, 0.0)
            result["ehc_excess_auc_fraction"] = auc(td[mask], excess) / max(total_auc, 1e-12)
    return result


def h0_prominence_diagram(c_dense: np.ndarray, threshold_ratio: float = DEFAULT_PROM) -> np.ndarray:
    """Normalized superlevel H0 diagram represented as (death, birth)."""
    c = np.asarray(c_dense, dtype=float)
    dyn = float(np.ptp(c))
    if dyn <= 1e-12:
        return np.empty((0, 2))
    z = (c - np.min(c)) / dyn
    peaks, props = find_peaks(z, prominence=threshold_ratio)
    proms = props.get("prominences", np.array([], dtype=float))

    if len(peaks) == 0:
        peak_idx = int(np.argmax(z))
        return np.array([[0.0, float(z[peak_idx])]])

    main_j = int(np.argmax(z[peaks]))
    points = []
    for j, (idx, prom) in enumerate(zip(peaks, proms)):
        birth = float(z[idx])
        death = 0.0 if j == main_j else float(max(0.0, birth - prom))
        points.append([death, birth])
    return np.asarray(points, dtype=float)


def analyze_profile(subject: int, sdf: pd.DataFrame) -> tuple[dict, dict]:
    """Return the flat parameter record and the plotting payload for one profile."""
    sdf = sdf.sort_values("Time")
    t_raw = sdf["Time"].to_numpy(float)
    c_raw = sdf["Conc"].to_numpy(float)
    t, anchor_notes = anchor_digitized_times(t_raw)
    td, cd = luld_interpolate(t, c_raw, num_points=N_INTERP)

    core = TopoPKAnalyzer(td, cd, tau=TAU, noise_ratio=DEFAULT_PROM).extract_all()
    late = detect_late_peak(t, c_raw)

    record = {
        "Profile_ID": int(subject),
        "N_observations": int(len(t)),
        "Time_anchor_notes": "; ".join(anchor_notes),
        **core,
        "LatePeak_detected": int(bool(late["late_detected"])),
        "Primary_peak_time_h": late["primary_time"],
        "Primary_peak_mg_L": late["primary_conc"],
        "Late_peak_time_h": late["late_time"],
        "Late_peak_mg_L": late["late_conc"],
        "Late_peak_prominence_mg_L": late["late_prominence"],
        "Late_persistence_Cmax": late["late_persistence_cmax"],
        "Late_persistence_range": late["late_persistence_range"],
        "Valley_time_h": late["valley_time"],
        "Valley_mg_L": late["valley_conc"],
        "Interpeak_time_h": late["interpeak_time"],
        "Secondary_primary_ratio": late["secondary_primary_ratio"],
        "Valley_Cmax_ratio": late["valley_cmax_ratio"],
        "Valley_lower_peak_ratio": late["valley_lower_peak_ratio"],
        "EHC_proxy_AUC_fraction": late["ehc_auc_fraction"],
        "EHC_excess_proxy_AUC_fraction": late["ehc_excess_auc_fraction"],
        "Counterfactual_loglinear_R2": late["counterfactual_r2"],
    }

    details = {
        "t_raw": t,
        "c_raw": c_raw,
        "t_dense": td,
        "c_dense": cd,
        "core": core,
        "late": late,
        "diagram": h0_prominence_diagram(cd),
    }
    return record, details


def plot_signature_group(
    profile_ids: list[int],
    row_labels: list[str],
    details: dict[int, dict],
    suptitle: str,
    output_stem: str,
) -> None:
    """Draw three profiles using the curve / H0 / delay-embedding layout."""
    if len(profile_ids) != 3 or len(row_labels) != 3:
        raise ValueError("Each signature figure must contain exactly three profiles")
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for row_idx, (sid, label) in enumerate(zip(profile_ids, row_labels)):
        d = details[sid]
        c, t = d["c_dense"], d["t_dense"]
        late = d["late"]

        ax = axes[row_idx, 0]
        ax.plot(t, c, color="#2C3E50", lw=2)
        ax.plot(d["t_raw"], d["c_raw"], "o", color="#2C3E50", ms=4)
        if int(late["late_idx"]) >= 0:
            ax.scatter(late["primary_time"], late["primary_conc"], color="#D62728", s=55, zorder=4)
            ax.scatter(late["valley_time"], late["valley_conc"], color="#1F77B4", s=55, zorder=4)
            ax.scatter(
                late["late_time"], late["late_conc"], s=55, zorder=4,
                facecolor="#E69F00" if late["late_detected"] else "none",
                edgecolor="#E69F00", linewidth=1.5,
            )
        ax.set_title(f"{label}: profile {sid}")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("MPA (mg/L)")
        ax.grid(alpha=0.2)

        ax = axes[row_idx, 1]
        diag = d["diagram"]
        ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
        if len(diag):
            ax.scatter(diag[:, 0], diag[:, 1], s=70, color="#7B2CBF", edgecolor="white")
            for death, birth in diag:
                ax.plot([death, death], [death, birth], ":", color="#7B2CBF", lw=1)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Death level")
        ax.set_ylabel("Birth level")
        ax.set_title("Normalized H0 diagram")
        ax.grid(alpha=0.2)

        ax = axes[row_idx, 2]
        z = (c - np.min(c)) / max(np.ptp(c), 1e-12)
        ax.plot(z[:-TAU], z[TAU:], color="#009E73", lw=1.6)
        ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
        ax.set_xlabel("C(t), normalized")
        ax.set_ylabel(f"C(t+{TAU}), normalized")
        ax.set_title(
            f"Delay embedding: beta1={d['core']['Beta_1']}, lambda_NL={d['core']['Lambda_NL']:.3f}"
        )
        ax.grid(alpha=0.2)

    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / f"{output_stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_all_signature_groups(features: pd.DataFrame, details: dict[int, dict]) -> None:
    """Export all 15 profiles as five sequential three-profile figures."""
    if len(details) != 15:
        raise ValueError(f"Expected 15 profiles, found {len(details)}")

    # Tertiles of late-peak persistence replace the original cluster phenotype
    # labels, so no clustering or tau-sensitivity pass is needed here.
    rank = features.set_index("Profile_ID")["Late_persistence_range"].rank(method="first")
    n = len(rank)

    def label(sid: int) -> str:
        r = rank.loc[sid]
        if r <= n / 3:
            return "Weak"
        if r <= 2 * n / 3:
            return "Intermediate"
        return "Strong"

    all_ids = sorted(int(x) for x in details)
    for group_number, start in enumerate(range(0, len(all_ids), 3), start=1):
        profile_ids = all_ids[start : start + 3]
        id_range = f"{profile_ids[0]}-{profile_ids[-1]}"
        plot_signature_group(
            profile_ids,
            [label(sid) for sid in profile_ids],
            details,
            f"MPA TopoPK signatures | profiles {id_range}",
            f"Figure_signatures_group_{group_number}_profiles_{id_range}",
        )


def main() -> None:
    source = BASE / "260902_raw_data.xlsx"
    raw = pd.read_excel(source, sheet_name="Sheet1")
    raw.columns = [str(c).strip() for c in raw.columns]
    required = {"ID", "Time", "Conc"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Expected columns {sorted(required)}, got {raw.columns.tolist()}")
    raw = raw[["ID", "Time", "Conc"]].rename(columns={"ID": "Profile_ID"})
    raw = raw.sort_values(["Profile_ID", "Time"]).reset_index(drop=True)

    records: list[dict] = []
    details: dict[int, dict] = {}
    for sid, sdf in raw.groupby("Profile_ID"):
        record, detail = analyze_profile(int(sid), sdf)
        records.append(record)
        details[int(sid)] = detail

    features = pd.DataFrame(records).sort_values("Profile_ID").reset_index(drop=True)
    features.to_csv(
        OUT / "MPA_TopoPK_features.csv", index=False, encoding="utf-8-sig", float_format="%.8g"
    )

    plot_all_signature_groups(features, details)

    shown = [
        "Profile_ID", "N_observations", "N_H0", "Birth_Time_main", "Birth_Level_main",
        "Pers_main", "Birth_Time_sub", "Delta_t", "Pers_Ratio", "Entropy_PE",
        "Beta_1", "Lambda_NL", "Decouple", "Dev", "N_PTP",
        "LatePeak_detected", "Late_persistence_range", "Interpeak_time_h",
        "Valley_lower_peak_ratio", "EHC_proxy_AUC_fraction",
    ]
    print(features[shown].round(4).to_string(index=False))
    print(f"\n{len(features)} profiles -> {len(features) // 3} SVG figures in {OUT}")


if __name__ == "__main__":
    main()
