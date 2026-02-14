#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D toy dataset (2×2) with angle sweep (CCGP vs XOR).

- Four conditions: FP, SP, FM, SM (Feature ± × Task ±)
- Base geometry: feature axis along x; task axis rotated in the x–y plane
- Optional out-of-plane tilt applied to a condition pair (default: FM & SM)
- Isotropic Gaussian noise
- Outputs a 2×2 figure:
  A: raw CCGP vs raw XOR
  B: probit-normalized CCGP vs XOR
  C: angle vs log(raw XOR / raw CCGP)
  D: angle vs (normalized XOR − normalized CCGP)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score


# =========================
# Configuration
# =========================
@dataclass(frozen=True)
class SimConfig:
    theta_deg: float = 90.0
    b_task: float = 2.0

    n_per_class: int = 2000
    noise_var: float = 1.0
    noise_scale: float = 1.0
    seed: int = 41

    tilt_pair: Tuple[str, str] = ("FM", "SM")
    tilt_out_deg: float = 0.0

    cv_splits: int = 5
    cv_seed: int = 0


# =========================
# Geometry utilities
# =========================
def get_axes(theta_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    th = np.deg2rad(theta_deg)
    v_feat = np.array([1.0, 0.0, 0.0], float)
    v_task = np.array([np.cos(th), np.sin(th), 0.0], float)
    return v_feat, v_task


def build_means(theta_deg: float, a_feat: float, b_task: float) -> Dict[str, np.ndarray]:
    v_feat, v_task = get_axes(theta_deg)

    def m(sf: int, st: int) -> np.ndarray:
        return sf * a_feat * v_feat + st * b_task * v_task

    return {"FP": m(+1, +1), "SP": m(-1, +1), "FM": m(+1, -1), "SM": m(-1, -1)}


def tilt_pair_out_of_plane(
    means: Dict[str, np.ndarray],
    pair: Tuple[str, str],
    angle_deg: float,
    v_feat: np.ndarray,
    v_task: np.ndarray,
) -> Dict[str, np.ndarray]:
    if angle_deg == 0.0:
        return {k: v.copy() for k, v in means.items()}

    a, b = pair
    if a not in means or b not in means:
        raise ValueError(f"pair must be keys of means, got {pair}")

    mu = {k: v.copy() for k, v in means.items()}

    mid = 0.5 * (mu[a] + mu[b])
    u = mu[a] - mid
    norm_u = np.linalg.norm(u)
    if norm_u == 0:
        return mu

    u_dir = u / norm_u
    n = np.cross(v_feat, v_task)
    norm_n = np.linalg.norm(n)
    if norm_n == 0:
        raise ValueError("v_feat and v_task are collinear; plane normal is undefined.")
    n_dir = n / norm_n

    th = np.deg2rad(angle_deg)
    r_dir = np.cos(th) * u_dir + np.sin(th) * n_dir
    r = norm_u * r_dir

    mu[a] = mid + r
    mu[b] = mid - r
    return mu


# =========================
# Sampling
# =========================
def sample_gaussian_isotropic(
    means: Dict[str, np.ndarray],
    n_per_class: int,
    noise_var: float,
    noise_scale: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cov = np.eye(3) * (noise_var * noise_scale)

    rows = []
    for label, mu in means.items():
        samples = rng.multivariate_normal(mean=mu, cov=cov, size=n_per_class)
        df = pd.DataFrame(samples, columns=["x", "y", "z"])
        df["condition"] = label
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# =========================
# Metrics: CCGP and XOR
# =========================
def compute_ccgp_xor(df: pd.DataFrame, cv_splits: int = 5, cv_seed: int = 0) -> Dict[str, float]:
    conds = df["condition"].astype(str).str.upper()

    face_set = {"FP", "FM"}
    task_map = {"FP": 0, "SP": 0, "FM": 1, "SM": 1}

    y_feat = conds.map(lambda c: 1 if c in face_set else 0).to_numpy(int)
    y_task = conds.map(task_map).to_numpy(int)
    y_xor = (y_feat ^ y_task).astype(int)

    X = df[["x", "y", "z"]].to_numpy(float)

    mask_P = conds.isin(["FP", "SP"]).to_numpy()
    mask_WM = conds.isin(["FM", "SM"]).to_numpy()

    X_P, y_feat_P = X[mask_P], y_feat[mask_P]
    X_WM, y_feat_WM = X[mask_WM], y_feat[mask_WM]

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="liblinear", max_iter=2000, random_state=0),
    )

    clf.fit(X_P, y_feat_P)
    acc_P2WM = accuracy_score(y_feat_WM, clf.predict(X_WM))
    clf.fit(X_WM, y_feat_WM)
    acc_WM2P = accuracy_score(y_feat_P, clf.predict(X_P))
    ccgp = 0.5 * (acc_P2WM + acc_WM2P)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed)
    xor_mean = float(cross_val_score(clf, X, y_xor, cv=cv, scoring="accuracy").mean())
    feat_P = float(cross_val_score(clf, X_P, y_feat_P, cv=cv, scoring="accuracy").mean())
    feat_WM = float(cross_val_score(clf, X_WM, y_feat_WM, cv=cv, scoring="accuracy").mean())

    return {"CCGP": float(ccgp), "XOR": xor_mean, "Feat_P": feat_P, "Feat_WM": feat_WM}


def sweep_angles(angles_deg: Iterable[float], a_feat: float, cfg: SimConfig) -> pd.DataFrame:
    rows = []
    v_feat, v_task = get_axes(cfg.theta_deg)

    for ang in angles_deg:
        base_means = build_means(cfg.theta_deg, a_feat, cfg.b_task)
        means = tilt_pair_out_of_plane(base_means, cfg.tilt_pair, float(ang), v_feat, v_task)

        df = sample_gaussian_isotropic(
            means=means,
            n_per_class=cfg.n_per_class,
            noise_var=cfg.noise_var,
            noise_scale=cfg.noise_scale,
            seed=cfg.seed,
        )

        res = compute_ccgp_xor(df, cv_splits=cfg.cv_splits, cv_seed=cfg.cv_seed)
        rows.append({"angle_deg": float(ang), **res})

    out = pd.DataFrame(rows)

    eps = 1e-6
    feat_mean = 0.5 * (out["Feat_P"] + out["Feat_WM"])
    z_feat = norm.ppf(feat_mean.clip(eps, 1 - eps))
    out["CCGP_ratio"] = norm.ppf(out["CCGP"].clip(eps, 1 - eps)) / z_feat
    out["XOR_ratio"] = norm.ppf(out["XOR"].clip(eps, 1 - eps)) / z_feat

    return out.replace([np.inf, -np.inf], np.nan)


# =========================
# Plotting helpers
# =========================
def plot_tradeoff_curves(
    results: pd.DataFrame,
    ax_raw: plt.Axes,
    ax_norm: plt.Axes,
    label: str,
    color=None,
) -> None:
    ax_raw.plot(results["CCGP"], results["XOR"], "-o", lw=1.8, alpha=0.9, label=label, color=color)
    ax_norm.plot(results["CCGP_ratio"], results["XOR_ratio"], "-o", lw=1.8, alpha=0.9, label=label, color=color)


# =========================
# Main
# =========================
def main() -> None:
    cfg = SimConfig(n_per_class=10000)

    angles = np.arange(0, 91, 5)
    a_feat_list = [0.5, 0.75, 1.0, 1.25, 1.5]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axA, axB, axC, axD = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0, 1, len(a_feat_list)))

    for a_val, col in zip(a_feat_list, colors):
        df = sweep_angles(angles, a_val, cfg)

        plot_tradeoff_curves(
            df,
            ax_raw=axA,
            ax_norm=axB,
            label=rf"$S_{{(F-S)}}={a_val:.2f}$",
            color=col,
        )

        eps = 1e-6
        log_ratio = np.log(
            np.clip(df["XOR"], eps, 1 - eps) /
            np.clip(df["CCGP"], eps, 1 - eps)
        )
        axC.plot(df["angle_deg"], log_ratio, "-o", lw=1.8, alpha=0.9, color=col)

        diff_ratio = df["XOR_ratio"] - df["CCGP_ratio"]
        axD.plot(df["angle_deg"], diff_ratio, "-o", lw=1.8, alpha=0.9, color=col)

    axA.set(xlabel="CCGP accuracy", ylabel="XOR accuracy")
    axB.set(xlabel="normalized CCGP", ylabel="normalized XOR")
    axC.set(xlabel="angle (deg)", ylabel="log (XOR accuracy / CCGP accuracy)", xlim=(-5, 95))
    axD.set(xlabel="angle (deg)", ylabel="normalized XOR − normalized CCGP", xlim=(-5, 95))

    for ax in (axA, axB, axC, axD):
        ax.grid(True, alpha=0.4)

    axC.margins(x=0.03)
    axD.margins(x=0.03)

    axB.legend(title="SNR", fontsize=9)

    for ax, lab in zip((axA, axB, axC, axD), "ABCD"):
        ax.text(-0.15, 0.99, lab, transform=ax.transAxes, fontsize=20, va="top", ha="left")

    fig.tight_layout()
    fig.savefig("final_output/figure_simulation_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
