#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


# =========================
# Config
# =========================
EXCLUDE = 0
ALPHA_FDR = 0.05

IN_DIR = Path("test_retest_data")
OUT_DIR = Path("final_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_IDS = [1, 2, 4, 5, 6, 7, 8, 9]

ROI_NAMES = [
    "Primary Visual", "Early Visual", "Dorsal Stream Visual", "Ventral Stream Visual", "MT+ Complex",
    "Somatosensory / Motor", "Paracentral / Mid Cingulate", "Premotor", "Posterior Opercular",
    "Early Auditory", "Auditory Association", "Insular / Frontal Opercular", "Medial Temporal",
    "Lateral Temporal", "Temporo-Parieto-Occipital Junction", "Superior Parietal", "Inferior Parietal",
    "Posterior Cingulate", "Anterior Cingulate / Medial Prefrontal", "Orbital / Polar Frontal",
    "Inferior Frontal", "DorsoLateral Prefrontal",
]

MARKERS_BY_SUBJECT = {
    1: "o", 2: "s", 4: "^", 5: "D", 6: "v", 7: "*", 8: "<", 9: "P"
}


# =========================
# IO
# =========================
def load_pkl(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


# =========================
# Stats
# =========================
def fisher_z(x: np.ndarray) -> np.ndarray:
    return np.arctanh(x)


def one_sided_ttest_greater_than_zero(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.nan
    t_stat, p_two = ttest_1samp(x, popmean=0.0, nan_policy="omit")
    if np.isnan(t_stat) or np.isnan(p_two):
        return np.nan
    return (p_two / 2.0) if (t_stat > 0) else 1.0


def fdr_bh(pvals: np.ndarray, alpha: float):
    q = np.full_like(pvals, np.nan, dtype=float)
    reject = np.zeros_like(pvals, dtype=bool)
    mask = np.isfinite(pvals)
    if np.any(mask):
        rej, qvals, _, _ = multipletests(pvals[mask], alpha=alpha, method="fdr_bh")
        reject[mask] = rej
        q[mask] = qvals
    return reject, q


def star_from_q(q: float) -> str:
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


# =========================
# Main
# =========================
def main():
    n_roi = len(ROI_NAMES)
    n_sub = len(SUBJECT_IDS)

    dataP = np.full((n_roi, n_sub), np.nan)
    dataWM = np.full((n_roi, n_sub), np.nan)

    for s_idx, subid in enumerate(SUBJECT_IDS):
        if (EXCLUDE == 1) and (subid in (2, 5)):
            path_p = IN_DIR / f"CorrP_{subid}_exclude.pkl"
            path_wm = IN_DIR / f"CorrWM_{subid}_exclude.pkl"
        else:
            path_p = IN_DIR / f"CorrP_{subid}.pkl"
            path_wm = IN_DIR / f"CorrWM_{subid}.pkl"

        corrP = np.asarray(load_pkl(path_p))    # (n_roi, n_perm)
        corrWM = np.asarray(load_pkl(path_wm))  # (n_roi, n_perm)

        if corrP.shape != corrWM.shape:
            raise ValueError(f"Shape mismatch: {path_p.name} vs {path_wm.name}: {corrP.shape} vs {corrWM.shape}")
        if corrP.shape[0] != n_roi:
            raise ValueError(f"ROI mismatch: expected n_roi={n_roi}, got {corrP.shape[0]} in {path_p.name}")

        for i in range(n_roi):
            vP = corrP[i, :]
            vWM = corrWM[i, :]
            vP = vP[np.isfinite(vP)]
            vWM = vWM[np.isfinite(vWM)]

            if vP.size:
                dataP[i, s_idx] = float(np.mean(vP))
            if vWM.size:
                dataWM[i, s_idx] = float(np.mean(vWM))

    meanP = np.nanmean(dataP, axis=1)
    meanWM = np.nanmean(dataWM, axis=1)

    dataP_z = fisher_z(dataP)
    dataWM_z = fisher_z(dataWM)

    pP = np.array([one_sided_ttest_greater_than_zero(dataP_z[i]) for i in range(n_roi)], dtype=float)
    pWM = np.array([one_sided_ttest_greater_than_zero(dataWM_z[i]) for i in range(n_roi)], dtype=float)

    _, qP = fdr_bh(pP, alpha=ALPHA_FDR)
    _, qWM = fdr_bh(pWM, alpha=ALPHA_FDR)

    fig_w = max(8, n_roi * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, 8))

    x = np.arange(n_roi)
    offset = 0.18
    width = 0.30

    ax.bar(x - offset, meanP, width, color="#F58518", alpha=0.8, label="Face vs. Scene (Perception)")
    ax.bar(x + offset, meanWM, width, color="#4C78A8", alpha=0.8, label="Face vs. Scene (Working Memory)")

    jitter = 0.03
    for i in range(n_roi):
        for s_idx, subid in enumerate(SUBJECT_IDS):
            mk = MARKERS_BY_SUBJECT.get(subid, "o")

            vP = dataP[i, s_idx]
            if np.isfinite(vP):
                xj = (x[i] - offset) + (s_idx - (n_sub - 1) / 2) * jitter
                ax.plot(
                    xj, vP, marker=mk, linestyle="None",
                    markerfacecolor="none", markeredgecolor="black",
                    markersize=6, alpha=0.75
                )

            vWM = dataWM[i, s_idx]
            if np.isfinite(vWM):
                xj = (x[i] + offset) + (s_idx - (n_sub - 1) / 2) * jitter
                ax.plot(
                    xj, vWM, marker=mk, linestyle="None",
                    markerfacecolor="none", markeredgecolor="black",
                    markersize=6, alpha=0.75
                )

    ax.axhline(0, ls="--", lw=1, color="black", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(ROI_NAMES, rotation=90)
    ax.set_xlim(-0.7, n_roi - 0.3)
    ax.set_ylabel("Mean correlation across permutations", fontsize=15)

    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}".rstrip("0").rstrip(".")))

    yP = 1.02
    yWM = 1.06
    for i in range(n_roi):
        sp = star_from_q(qP[i])
        if sp:
            ax.text(
                x[i], yP, sp, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=18, color="#7a3a00"
            )
        swm = star_from_q(qWM[i])
        if swm:
            ax.text(
                x[i], yWM, swm, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=18, color="#1f3a5f"
            )

    legend_elems = [
        Patch(facecolor="#F58518", alpha=0.8, label="Face vs. Scene (Perception)"),
        Patch(facecolor="#4C78A8", alpha=0.8, label="Face vs. Scene (Working Memory)"),
    ]
    ax.legend(handles=legend_elems, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.2))

    fig.tight_layout()
    outpath = OUT_DIR / "figure_test-retest.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", outpath.resolve())


if __name__ == "__main__":
    main()
