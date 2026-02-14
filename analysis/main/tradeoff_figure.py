#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproducible end-to-end pipeline:

1) BigROI tradeoff summary + scatter (bootstrap CI with fixed RNG)
   + (extra) BigROI CCGP vs XOR scatter (raw + normalized), with draggable ROI labels
2) Fine-grained (HCP-MMP1 annot-based) tradeoff map on fsaverage (LH)
   - q/FDR mask affects ONLY visualization (LH stat map), not the tradeoff values themselves
3) Parcel-wise BrainSMASH test of Tradeoff × PG gradients (Margulies 2016; fsLR32k)
   - BrainSMASH input vector is NaN-free by construction
   - Correlation mask applied ONLY at statistics stage
   - Distance matrix is cached as .npy for strict reproducibility
"""

import os
import string
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import norm, ttest_1samp, pearsonr, spearmanr
from statsmodels.stats.multitest import fdrcorrection

from nilearn import plotting
from nilearn.datasets import fetch_surf_fsaverage

from neuromaps.datasets import fetch_annotation, fetch_fslr
from netneurotools import datasets as nntdata

from brainsmash.mapgen.base import Base
from brainsmash.workbench.geo import cortex


EPS = 1e-6


# =============================================================================
# Utilities
# =============================================================================
def ensure_dir(path_like) -> None:
    Path(path_like).mkdir(parents=True, exist_ok=True)


def get_gifti_data(img_or_path):
    gii = nib.load(img_or_path) if isinstance(img_or_path, str) else img_or_path
    return np.asarray(gii.agg_data()).squeeze()


def safe_ppf(p, eps=EPS):
    p = np.asarray(p, float)
    p = np.clip(p, eps, 1.0 - eps)
    return norm.ppf(p)


def bootstrap_ci_mean(x, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, (np.nan, np.nan)
    boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(np.mean(x)), (lo, hi)


def one_tailed_p_from_vec(x, mu0, alternative="greater"):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    _, p = ttest_1samp(x, popmean=mu0, nan_policy="omit", alternative=alternative)
    return float(p)


def fdr_qvals_from_pvec(pvec, alpha=0.05):
    pvec = np.asarray(pvec, float)
    finite = np.isfinite(pvec)
    qvec = np.full_like(pvec, np.nan, dtype=float)
    rej = np.zeros_like(pvec, dtype=bool)
    if finite.any():
        rej_sub, q_sub = fdrcorrection(pvec[finite], alpha=alpha)
        rej[finite] = rej_sub
        qvec[finite] = q_sub
    return qvec, rej


def subject_csv_path(base_dir, out_tag, sid, exclude):
    base_dir = str(base_dir)
    if (exclude == 1) and (sid in (2, 5)):
        return f"{base_dir}/summary_{out_tag}_{sid}_exclude.csv"
    else:
        return f"{base_dir}/summary_{out_tag}_{sid}.csv"


# =============================================================================
# Tradeoff definitions
# =============================================================================
def tradeoff_simple(acc_xor, acc_ccgp, eps=EPS):
    acc_xor = np.asarray(acc_xor, float)
    acc_ccgp = np.asarray(acc_ccgp, float)
    acc_ccgp = np.clip(acc_ccgp, eps, None)
    return np.log(acc_xor / acc_ccgp)


def tradeoff_probit(acc_feat, acc_ccgp, acc_xor, eps=EPS):
    z_feat = safe_ppf(acc_feat)
    z_ccgp = safe_ppf(acc_ccgp)
    z_xor = safe_ppf(acc_xor)

    denom = np.where(np.abs(z_feat) < eps, np.nan, z_feat)
    return (z_xor / denom) - (z_ccgp / denom)


def compute_tradeoff(acc_feat, acc_ccgp, acc_xor, index: int):
    if index == 1:
        return tradeoff_simple(acc_xor, acc_ccgp), "simple"
    if index == 2:
        return tradeoff_probit(acc_feat, acc_ccgp, acc_xor), "probit"
    raise ValueError("index must be 1 (simple) or 2 (probit)")


# =============================================================================
# BigROI pipeline
# =============================================================================
def load_subject_dfs(csv_files):
    dfs = []
    for f in csv_files:
        if not Path(f).exists():
            raise FileNotFoundError(f"Not found: {f}")
        dfs.append(pd.read_csv(f))
    return dfs


def group_qvals_over_rois(dfs, value_col, use_zval, chance=0.5, alpha=0.05):
    mat = []
    for df in dfs:
        v = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
        mat.append(v if use_zval else safe_ppf(v))
    mat = np.vstack(mat)

    popmean = 0.0 if use_zval else float(norm.ppf(chance))
    pvals = np.full(mat.shape[1], np.nan, dtype=float)
    tstats = np.full(mat.shape[1], np.nan, dtype=float)

    for r in range(mat.shape[1]):
        vec = mat[:, r]
        vec = vec[np.isfinite(vec)]
        if vec.size < 2:
            continue
        t, p = ttest_1samp(vec, popmean=popmean, nan_policy="omit", alternative="greater")
        tstats[r] = float(t)
        pvals[r] = float(p)

    qvals, rej = fdr_qvals_from_pvec(pvals, alpha=alpha)
    return pvals, qvals, tstats, rej


def build_roi_mask(q_feat, q_ccgp, q_xor, q_alpha):
    q_feat = np.asarray(q_feat, float)
    q_ccgp = np.asarray(q_ccgp, float)
    q_xor = np.asarray(q_xor, float)
    return ((q_ccgp < q_alpha) | (q_xor < q_alpha)) & (q_feat < q_alpha)


def compute_tradeoff_long_table_bigroi(
    dfs, subject_ids, roi_names, mask,
    feature_acc_col, ccgp_acc_col, xor_acc_col,
    index
):
    mask = np.asarray(mask, dtype=bool)
    roi_vec = np.asarray(roi_names)[mask]

    rows = []
    dropped = 0

    for sid, df in zip(subject_ids, dfs):
        feat = pd.to_numeric(df[feature_acc_col], errors="coerce").to_numpy(dtype=float)[mask]
        ccgp = pd.to_numeric(df[ccgp_acc_col], errors="coerce").to_numpy(dtype=float)[mask]
        xor = pd.to_numeric(df[xor_acc_col], errors="coerce").to_numpy(dtype=float)[mask]

        tr, t_index_name = compute_tradeoff(feat, ccgp, xor, index=index)
        dropped += int(np.sum(~np.isfinite(tr)))

        rows.append(pd.DataFrame({
            "SubjectID": sid,
            "ROI": roi_vec,
            "tradeoff": tr
        }))

    dat = pd.concat(rows, ignore_index=True)
    dat = dat[np.isfinite(dat["tradeoff"].to_numpy(dtype=float))].copy()

    if dropped > 0:
        print(f"[BigROI] Dropped non-finite tradeoff values: {dropped}")

    return dat, t_index_name


def aggregate_subject_roi(dat):
    return (dat.groupby(["ROI", "SubjectID"], as_index=False)
            .agg(tradeoff=("tradeoff", "mean")))


def summarize_by_roi(dat_agg, roi_order, n_boot=10000, alpha=0.05, seed=0, sort_by="mean"):
    roi_stats = []
    for roi, sub in dat_agg.groupby("ROI"):
        m, (lo, hi) = bootstrap_ci_mean(sub["tradeoff"].to_numpy(float), n_boot=n_boot, alpha=alpha, seed=seed)
        roi_stats.append({
            "ROI": roi,
            "mean": m,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_subj": int(sub["SubjectID"].nunique())
        })

    stats = pd.DataFrame(roi_stats).dropna(subset=["mean"])
    stats["ROI"] = pd.Categorical(stats["ROI"], categories=list(roi_order), ordered=True)

    if sort_by == "mean":
        stats = stats.sort_values("mean").reset_index(drop=True)
    elif sort_by == "roi":
        stats = stats.sort_values("ROI").reset_index(drop=True)
    else:
        raise ValueError("sort_by must be 'mean' or 'roi'")
    return stats


def plot_tradeoff_scatter(stats, dat_agg, subject_ids, markers, out_png,
                          jitter=0.30, panel_label="A",
                          ylabel="Abstract  <-  Tradeoff index  ->  Flexible",
                          seed=0):
    rng = np.random.default_rng(seed)

    out_png = Path(out_png)
    ensure_dir(out_png.parent)

    rois = stats["ROI"].astype(str).tolist()
    x_pos = np.arange(len(rois))
    id_to_marker = {sid: mk for sid, mk in zip(subject_ids, markers)}

    fig_w = max(8, 0.6 * len(rois))
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    for sid in subject_ids:
        mk = id_to_marker.get(sid, "o")
        sub = dat_agg[dat_agg["SubjectID"] == sid]
        xs, ys = [], []
        for i, roi in enumerate(rois):
            y = sub.loc[sub["ROI"].astype(str) == roi, "tradeoff"].to_numpy(dtype=float)
            y = y[np.isfinite(y)]
            if y.size:
                xs.append(x_pos[i] + (rng.random() - 0.5) * jitter)
                ys.append(float(np.mean(y)))
        if xs:
            ax.scatter(xs, ys, marker=mk, facecolors="none",
                       edgecolors="black", s=40, alpha=0.7)

    yerr_lo = stats["mean"].to_numpy() - stats["ci_lo"].to_numpy()
    yerr_hi = stats["ci_hi"].to_numpy() - stats["mean"].to_numpy()
    ax.errorbar(
        x_pos, stats["mean"].to_numpy(),
        yerr=[yerr_lo, yerr_hi],
        fmt="o", ms=6, lw=2, capsize=4, color="k", zorder=5
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(rois, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.7, len(rois) - 0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(-0.05, 1.05, str(panel_label),
            transform=ax.transAxes, fontsize=18, va="top", ha="left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_bigroi_tradeoff(
    index,
    ids,
    markers,
    roi_names,
    use_zval_for_group=True,
    q_alpha_mask=0.01,
    exclude=0,
    base_dir="decoding_summary",
    out_dir="final_output",
    chance=0.5,
    seed=0,
    out_tag_big="bigROI",   # <<< NEW: naming rule tag
):
    ensure_dir(out_dir)

    # >>> NEW naming rule
    csv_files = [subject_csv_path(base_dir, out_tag_big, sid, exclude) for sid in ids]
    dfs = load_subject_dfs(csv_files)

    FEATURE_ACC_COL = "Feature_Decoding_Accuracy"
    CCGP_ACC_COL = "CCGP_Feature_Decoding_Accuracy"
    XOR_ACC_COL = "XOR_Decoding_Accuracy"

    FEATURE_Z_COL = "Feature_Decoding_zval"
    CCGP_Z_COL = "CCGP_Feature_Decoding_zval"
    XOR_Z_COL = "XOR_Decoding_zval"

    if use_zval_for_group:
        _, q_feat, _, _ = group_qvals_over_rois(dfs, FEATURE_Z_COL, use_zval=True, alpha=0.05)
        _, q_ccgp, _, _ = group_qvals_over_rois(dfs, CCGP_Z_COL, use_zval=True, alpha=0.05)
        _, q_xor, _, _ = group_qvals_over_rois(dfs, XOR_Z_COL, use_zval=True, alpha=0.05)
    else:
        _, q_feat, _, _ = group_qvals_over_rois(dfs, FEATURE_ACC_COL, use_zval=False, chance=chance, alpha=0.05)
        _, q_ccgp, _, _ = group_qvals_over_rois(dfs, CCGP_ACC_COL, use_zval=False, chance=chance, alpha=0.05)
        _, q_xor, _, _ = group_qvals_over_rois(dfs, XOR_ACC_COL, use_zval=False, chance=chance, alpha=0.05)

    mask = build_roi_mask(q_feat, q_ccgp, q_xor, q_alpha=q_alpha_mask)

    excluded = np.asarray(roi_names)[~mask]
    print(f"\n[BigROI] Excluded ROIs ({excluded.size}):")
    for r in excluded:
        print(f"  {r}")

    dat, t_index_name = compute_tradeoff_long_table_bigroi(
        dfs=dfs,
        subject_ids=ids,
        roi_names=roi_names,
        mask=mask,
        feature_acc_col=FEATURE_ACC_COL,
        ccgp_acc_col=CCGP_ACC_COL,
        xor_acc_col=XOR_ACC_COL,
        index=index
    )

    expected = set(np.asarray(roi_names)[mask])
    actual = set(dat["ROI"].unique())
    if expected != actual:
        raise RuntimeError("ROI mismatch after masking; check ROI order and mask alignment")

    dat_agg = aggregate_subject_roi(dat)

    stats = summarize_by_roi(
        dat_agg=dat_agg,
        roi_order=list(np.asarray(roi_names)[mask]),
        n_boot=10000,
        alpha=0.05,
        seed=seed,
        sort_by="mean"
    )

    stats_out = Path(out_dir) / f"tradeoff_{t_index_name}_bigroi_stats.csv"
    stats.to_csv(stats_out, index=False)

    fig_out = Path(out_dir) / f"figure_tradeoff_bigroi_{t_index_name}.png"
    plot_tradeoff_scatter(
        stats=stats,
        dat_agg=dat_agg,
        subject_ids=ids,
        markers=markers,
        out_png=str(fig_out),
        jitter=0.30,
        panel_label="A",
        seed=seed
    )

    return t_index_name


# =============================================================================
# (MERGED) BigROI CCGP-vs-XOR scatter (raw + normalized) with draggable labels
# =============================================================================
def build_bigroi_scatter_from_csv(
    ids,
    roi_names,
    base_dir="decoding_summary",
    exclude=0,
    q_alpha_mask=0.01,
    use_zval_for_group=True,
    chance=0.5,
    out_png="final_output/figure_scatter_ccgpvsxor.png",
    out_tag_big="bigROI",   # <<< NEW: naming rule tag
):
    # >>> NEW naming rule
    csv_files = [subject_csv_path(base_dir, out_tag_big, sid, exclude) for sid in ids]
    dfs = [pd.read_csv(f) for f in csv_files]

    FEATURE_ACC_COL = "Feature_Decoding_Accuracy"
    CCGP_ACC_COL    = "CCGP_Feature_Decoding_Accuracy"
    XOR_ACC_COL     = "XOR_Decoding_Accuracy"

    FEATURE_Z_COL = "Feature_Decoding_zval"
    CCGP_Z_COL    = "CCGP_Feature_Decoding_zval"
    XOR_Z_COL     = "XOR_Decoding_zval"

    feature_overall = np.vstack([pd.to_numeric(df[FEATURE_ACC_COL], errors="coerce").to_numpy(float) for df in dfs])
    ccgp_feature    = np.vstack([pd.to_numeric(df[CCGP_ACC_COL],    errors="coerce").to_numpy(float) for df in dfs])
    xor_overall     = np.vstack([pd.to_numeric(df[XOR_ACC_COL],     errors="coerce").to_numpy(float) for df in dfs])

    if use_zval_for_group:
        _, q_feat, _, _ = group_qvals_over_rois(dfs, FEATURE_Z_COL, use_zval=True, alpha=0.05)
        _, q_ccgp, _, _ = group_qvals_over_rois(dfs, CCGP_Z_COL, use_zval=True, alpha=0.05)
        _, q_xor, _, _  = group_qvals_over_rois(dfs, XOR_Z_COL, use_zval=True, alpha=0.05)
    else:
        _, q_feat, _, _ = group_qvals_over_rois(dfs, FEATURE_ACC_COL, use_zval=False, chance=chance, alpha=0.05)
        _, q_ccgp, _, _ = group_qvals_over_rois(dfs, CCGP_ACC_COL, use_zval=False, chance=chance, alpha=0.05)
        _, q_xor, _, _  = group_qvals_over_rois(dfs, XOR_ACC_COL, use_zval=False, chance=chance, alpha=0.05)

    mask = build_roi_mask(q_feat, q_ccgp, q_xor, q_alpha=q_alpha_mask)
    mask = np.asarray(mask, dtype=bool)

    mean_ccgp = np.nanmean(ccgp_feature[:, mask], axis=0)
    mean_xor  = np.nanmean(xor_overall[:,  mask], axis=0)

    denom = safe_ppf(feature_overall[:, mask])
    mean_norm_ccgp = np.nanmean(safe_ppf(ccgp_feature[:, mask]) / denom, axis=0)
    mean_norm_xor  = np.nanmean(safe_ppf(xor_overall[:,  mask]) / denom, axis=0)

    roi_subset = np.asarray(roi_names)[mask]

    pairs = [
        ("CCGP accuracy", (mean_ccgp, mean_xor), "XOR accuracy"),
        ("normalized CCGP", (mean_norm_ccgp, mean_norm_xor), "normalized XOR"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    all_texts = []

    for i, (title, (x, y), ylab) in enumerate(pairs):
        ax = axes[i]

        for xi, yi, label in zip(x, y, roi_subset):
            face = "red" if (label in roi_names[:5]) else "white"
            ax.scatter(xi, yi, s=45, facecolor=face, edgecolor="black", alpha=0.9, zorder=3)
            t = ax.text(xi, yi, str(label), fontsize=8, alpha=0.8)
            all_texts.append(t)

        ax.text(-0.15, 0.99, string.ascii_uppercase[i],
                transform=ax.transAxes, fontsize=20, va="top", ha="left")

        ax.set_xlabel(title)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)

    plt.tight_layout()

    drag_state = {"artist": None, "offset": (0.0, 0.0)}

    def on_press(event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        for text in all_texts:
            contains, _ = text.contains(event)
            if contains:
                drag_state["artist"] = text
                x0, y0 = text.get_position()
                drag_state["offset"] = (x0 - event.xdata, y0 - event.ydata)
                break

    def on_release(event):
        drag_state["artist"] = None

    def on_motion(event):
        if drag_state["artist"] is None or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx, dy = drag_state["offset"]
        drag_state["artist"].set_position((event.xdata + dx, event.ydata + dy))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    plt.show()

    out_png = Path(out_png)
    ensure_dir(out_png.parent)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", os.path.abspath(out_png))
    print("Mask kept ROIs:", roi_subset.size)
    return roi_subset, mean_norm_ccgp, mean_norm_xor


# =============================================================================
# Fine-grained (annot-based) tradeoff + q/FDR keep mask
# =============================================================================
def one_tailed_p_by_label(df_vals, mu0=0.5):
    p_by_label = {}
    for lid, g in df_vals.groupby("LabelID"):
        x = g["val"].to_numpy(dtype=float)
        p_by_label[int(lid)] = one_tailed_p_from_vec(x, mu0=mu0, alternative="greater")
    return p_by_label


def fdr_qvals_from_pdict(p_dict, all_labels, alpha=0.05):
    pvec = np.array([p_dict.get(int(lid), np.nan) for lid in all_labels], float)
    qvec, _ = fdr_qvals_from_pvec(pvec, alpha=alpha)
    return qvec


def build_finegrained_tradeoff_and_mask(ids, csv_files, index, q_alpha=0.01, mu0=0.5):
    rows_tradeoff = []
    rows_ccgp, rows_xor, rows_feat = [], [], []

    CCGP_COL = "CCGP_Feature_Decoding_Accuracy"
    XOR_COL = "XOR_Decoding_Accuracy"
    FEATURE_COL = "Feature_Decoding_Accuracy"

    dropped = 0

    for sid, f in zip(ids, csv_files):
        df = pd.read_csv(f)
        label_ids = np.arange(1, len(df) + 1, dtype=int)

        ccgp = pd.to_numeric(df[CCGP_COL], errors="coerce").to_numpy(dtype=float)
        xor = pd.to_numeric(df[XOR_COL], errors="coerce").to_numpy(dtype=float)
        feat = pd.to_numeric(df[FEATURE_COL], errors="coerce").to_numpy(dtype=float)

        tr, t_index_name = compute_tradeoff(feat, ccgp, xor, index=index)
        dropped += int(np.sum(~np.isfinite(tr)))

        rows_tradeoff.append(pd.DataFrame({"SubjectID": sid, "LabelID": label_ids, "tradeoff": tr}))
        rows_ccgp.append(pd.DataFrame({"SubjectID": sid, "LabelID": label_ids, "val": ccgp}))
        rows_xor.append(pd.DataFrame({"SubjectID": sid, "LabelID": label_ids, "val": xor}))
        rows_feat.append(pd.DataFrame({"SubjectID": sid, "LabelID": label_ids, "val": feat}))

    dat_tradeoff = pd.concat(rows_tradeoff, ignore_index=True)
    dat_ccgp = pd.concat(rows_ccgp, ignore_index=True).dropna(subset=["val"])
    dat_xor = pd.concat(rows_xor, ignore_index=True).dropna(subset=["val"])
    dat_feat = pd.concat(rows_feat, ignore_index=True).dropna(subset=["val"])

    dat_tradeoff = dat_tradeoff[np.isfinite(dat_tradeoff["tradeoff"].to_numpy(dtype=float))].copy()
    dat_tradeoff["LabelID"] = dat_tradeoff["LabelID"].astype(int)

    if dropped > 0:
        print(f"[FineROI] Dropped non-finite tradeoff values: {dropped}")

    tradeoff_by_label = dat_tradeoff.groupby("LabelID")["tradeoff"].mean()

    p_ccgp = one_tailed_p_by_label(dat_ccgp, mu0=mu0)
    p_xor = one_tailed_p_by_label(dat_xor, mu0=mu0)
    p_feat = one_tailed_p_by_label(dat_feat, mu0=mu0)

    all_labels = sorted(set(p_ccgp) | set(p_xor) | set(p_feat))

    q_ccgp = fdr_qvals_from_pdict(p_ccgp, all_labels, alpha=0.05)
    q_xor = fdr_qvals_from_pdict(p_xor, all_labels, alpha=0.05)
    q_feat = fdr_qvals_from_pdict(p_feat, all_labels, alpha=0.05)

    keep_mask = (q_feat < q_alpha) & ((q_ccgp < q_alpha) | (q_xor < q_alpha))
    keep_dict = {int(lid): bool(k) for lid, k in zip(all_labels, keep_mask)}

    return tradeoff_by_label, keep_dict, t_index_name


def make_1idx_arrays(tradeoff_by_label, keep_dict, left_labels_array, right_labels_array):
    max_label = int(max(left_labels_array.max(), right_labels_array.max()))
    vals_1idx = np.full(max_label + 1, np.nan, dtype=float)
    mask_1idx = np.zeros(max_label + 1, dtype=bool)

    for lid, v in tradeoff_by_label.items():
        lid = int(lid)
        if 1 <= lid <= max_label:
            vals_1idx[lid] = float(v)

    for lid, ok in keep_dict.items():
        lid = int(lid)
        if 1 <= lid <= max_label:
            mask_1idx[lid] = bool(ok)

    return vals_1idx, mask_1idx


def vertex_stat_map(labels_array, vals_1idx, mask_1idx):
    tex = vals_1idx[labels_array]
    return np.where(mask_1idx[labels_array], tex, np.nan)


def symmetric_vmin_vmax(stat_map, fallback=1.0):
    v = stat_map[np.isfinite(stat_map)]
    if v.size == 0:
        return -fallback, fallback
    vabs = float(np.nanmax(np.abs(v)))
    if not np.isfinite(vabs) or vabs <= 0:
        vabs = fallback
    return -vabs, vabs


def plot_left_hemisphere_grid(fsaverage, stat_map, out_png, views, axes_coords,
                             cmap="coolwarm", label="Tradeoff index", panel="B"):
    vmin, vmax = symmetric_vmin_vmax(stat_map)

    fig = plt.figure(figsize=(18, 12))
    for view in views:
        ax = fig.add_axes(axes_coords[view], projection="3d")
        plotting.plot_surf_stat_map(
            surf_mesh=fsaverage.infl_left,
            stat_map=stat_map,
            bg_map=fsaverage.sulc_left,
            hemi="left",
            view=view,
            cmap=cmap,
            colorbar=False,
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=False,
            axes=ax,
        )
        ax.set_facecolor("none")

    norm_c = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar_ax = fig.add_axes([0.07, 0.45, 0.02, 0.45])
    cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm_c)
    cb.set_label(label, fontsize=25)
    cb.ax.tick_params(labelsize=20)

    fig.text(0.1, 0.95, str(panel), fontsize=30, va="top", ha="left")

    out_png = Path(out_png)
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_finegrained_surface(
    index,
    sub_ids,
    csv_files,
    lh_annot_path,
    rh_annot_path,
    out_dir="final_output",
    q_alpha=0.01
):
    ensure_dir(out_dir)

    left_labels_array, _, names = nib.freesurfer.read_annot(os.path.expanduser(lh_annot_path))
    right_labels_array, _, _ = nib.freesurfer.read_annot(os.path.expanduser(rh_annot_path))

    tradeoff_by_label, keep_dict, t_index_name = build_finegrained_tradeoff_and_mask(
        ids=sub_ids,
        csv_files=csv_files,
        index=index,
        q_alpha=q_alpha,
        mu0=0.5
    )

    vals_1idx, mask_1idx = make_1idx_arrays(
        tradeoff_by_label=tradeoff_by_label,
        keep_dict=keep_dict,
        left_labels_array=left_labels_array,
        right_labels_array=right_labels_array
    )
    lh_map = vertex_stat_map(left_labels_array, vals_1idx, mask_1idx)

    fsaverage = fetch_surf_fsaverage(mesh="fsaverage")

    views = ["lateral", "dorsal", "anterior", "medial", "ventral", "posterior"]
    axes_coords = {
        "anterior":  [0.01, 0.33, 0.44, 0.60],
        "posterior": [0.69, 0.33, 0.44, 0.60],
        "lateral":   [0.21, 0.51, 0.44, 0.50],
        "ventral":   [0.21, 0.25, 0.44, 0.50],
        "medial":    [0.49, 0.51, 0.44, 0.50],
        "dorsal":    [0.49, 0.25, 0.44, 0.50],
    }

    out_png = Path(out_dir) / f"figure_tradeoff_finegrained_LH_{t_index_name}.png"
    plot_left_hemisphere_grid(
        fsaverage=fsaverage,
        stat_map=lh_map,
        out_png=str(out_png),
        views=views,
        axes_coords=axes_coords,
        cmap="coolwarm",
        label="Tradeoff index",
        panel="B"
    )

    return t_index_name, tradeoff_by_label, keep_dict, names


# =============================================================================
# PG means (fsLR32k; Margulies gradients)
# =============================================================================
def load_mmp_labels_fslr32k():
    mmp = nntdata.fetch_mmpall()
    lh = nib.load(mmp[0]).darrays[0].data.astype(int)  # LH labels: 181..360
    rh = nib.load(mmp[1]).darrays[0].data.astype(int)  # RH labels: 1..180
    return lh, rh


def compute_pg_mean_fslr32k(grad_index, lh_labels, rh_labels):
    desc = f"fcgradient{grad_index:02d}"
    lh_grad_img, rh_grad_img = fetch_annotation(
        source="margulies2016",
        desc=desc,
        space="fsLR",
        den="32k"
    )
    lh_grad = get_gifti_data(lh_grad_img)
    rh_grad = get_gifti_data(rh_grad_img)

    pg_left = np.full(180, np.nan, dtype=float)
    for lab in range(181, 361):
        idx = (lh_labels == lab)
        if idx.any():
            pg_left[lab - 181] = float(np.mean(lh_grad[idx]))

    pg_right = np.full(180, np.nan, dtype=float)
    for lab in range(1, 181):
        idx = (rh_labels == lab)
        if idx.any():
            pg_right[lab - 1] = float(np.mean(rh_grad[idx]))

    return (pg_left + pg_right) / 2.0


# =============================================================================
# Surface geodesic distance (fsLR32k midthickness) -> dense -> parcellate -> 180x180
# =============================================================================
def get_fslr32k_midthickness_paths():
    surfs = fetch_fslr(density="32k")
    mid = surfs["midthickness"]

    try:
        return str(mid["L"]), str(mid["R"])
    except Exception:
        pass

    if isinstance(mid, (tuple, list)) and len(mid) == 2:
        return str(mid[0]), str(mid[1])

    if hasattr(mid, "L") and hasattr(mid, "R"):
        return str(mid.L), str(mid.R)

    raise TypeError(f"Unexpected type for surfs['midthickness']: {type(mid)} ; value={mid}")


def parcellate_dense_distmat_by_labels(dense_txt, labels, n_parcels, label_offset=0):
    Dv = np.loadtxt(dense_txt, dtype=float)
    if Dv.ndim != 2 or Dv.shape[0] != Dv.shape[1]:
        raise RuntimeError(f"Dense distmat must be square; got {Dv.shape} from {dense_txt}")

    labels = np.asarray(labels, int)
    if labels.shape[0] != Dv.shape[0]:
        raise RuntimeError(f"Label length {labels.shape[0]} != distmat size {Dv.shape[0]}")

    idxs = []
    for i in range(1, n_parcels + 1):
        lab = i + label_offset
        idxs.append(np.where(labels == lab)[0])

    Dp = np.full((n_parcels, n_parcels), np.nan, dtype=float)

    for i in range(n_parcels):
        Ii = idxs[i]
        if Ii.size == 0:
            continue
        for j in range(i, n_parcels):
            Ij = idxs[j]
            if Ij.size == 0:
                continue
            block = Dv[np.ix_(Ii, Ij)]
            val = float(np.mean(block)) if block.size else np.nan
            Dp[i, j] = val
            Dp[j, i] = val

    np.fill_diagonal(Dp, 0.0)
    return Dp


def compute_parcel_distance_matrix_fslr32k(
    out_dir,
    lh_labels, rh_labels,
    n_parcels=180,
    euclid=False,
    force_recompute=False
):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    suffix = "euclid" if euclid else "geodesic"
    D_cache = out_dir / f"D_parcel_{suffix}.npy"
    if (not force_recompute) and D_cache.exists():
        print(f"[Geo] Using cached parcel distances: {D_cache}")
        return np.load(D_cache)

    surf_L, surf_R = get_fslr32k_midthickness_paths()
    dense_L = out_dir / f"fsLR32k_L_dense_{suffix}.txt"
    dense_R = out_dir / f"fsLR32k_R_dense_{suffix}.txt"

    if force_recompute or (not dense_L.exists()):
        print(f"[Geo] Computing dense {suffix} distances (L)...")
        cortex(surface=surf_L, outfile=str(dense_L), euclid=bool(euclid))
    else:
        print(f"[Geo] Using cached dense distances (L): {dense_L}")

    if force_recompute or (not dense_R.exists()):
        print(f"[Geo] Computing dense {suffix} distances (R)...")
        cortex(surface=surf_R, outfile=str(dense_R), euclid=bool(euclid))
    else:
        print(f"[Geo] Using cached dense distances (R): {dense_R}")

    DL = parcellate_dense_distmat_by_labels(str(dense_L), lh_labels, n_parcels=n_parcels, label_offset=180)
    DR = parcellate_dense_distmat_by_labels(str(dense_R), rh_labels, n_parcels=n_parcels, label_offset=0)

    D = 0.5 * (DL + DR)
    np.fill_diagonal(D, 0.0)

    if np.any(~np.isfinite(D)):
        raise RuntimeError("Parcel distance matrix has NaN/inf. (Missing parcels?)")

    np.save(D_cache, D)
    print(f"[Geo] Saved parcel distances: {D_cache}")
    return D


# =============================================================================
# BrainSMASH (parcel-wise) + PG
# =============================================================================
def build_tradeoff_vector_180(tradeoff_by_label, keep_dict=None, n_parcels=180):
    trade_all = np.zeros(n_parcels, dtype=float)

    if keep_dict is None:
        alive_mask = np.ones(n_parcels, dtype=bool)
    else:
        alive_mask = np.zeros(n_parcels, dtype=bool)

    series = tradeoff_by_label.sort_index() if hasattr(tradeoff_by_label, "sort_index") else tradeoff_by_label

    for lid, val in series.items():
        lid = int(lid)
        if not (1 <= lid <= n_parcels):
            continue
        trade_all[lid - 1] = float(val)
        if keep_dict is not None:
            alive_mask[lid - 1] = bool(keep_dict.get(lid, False))

    return trade_all, alive_mask


def plot_scatter_and_null(x, y, null, obs, out_png, xlabel, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax1 = axes[0]
    ax1.scatter(x, y, s=40, alpha=0.9, color="black")
    if x.size >= 2:
        coef = np.polyfit(x, y, 1)
        xfit = np.linspace(x.min(), x.max(), 100)
        yfit = np.polyval(coef, xfit)
        ax1.plot(xfit, yfit, linestyle="--", linewidth=2, color="black")
    ax1.set_xlabel(xlabel, fontsize=15)
    ax1.set_ylabel(ylabel, fontsize=15)
    ax1.tick_params(labelsize=12)
    ax1.grid(alpha=0.3)
    ax1.text(-0.15, 0.99, "A", transform=ax1.transAxes, fontsize=20, va="top", ha="left")

    ax2 = axes[1]
    nn = null[np.isfinite(null)]
    ax2.hist(nn, bins=30, density=False, color="lightgray", edgecolor="black")
    ax2.axvline(obs, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Spearman ρ (null)", fontsize=15)
    ax2.set_ylabel("Count", fontsize=15)
    ax2.tick_params(labelsize=12)
    ax2.text(-0.15, 0.99, "B", transform=ax2.transAxes, fontsize=20, va="top", ha="left")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def run_pg_brainsmash_surface_geodesic_parcel(
    tradeoff_by_label,
    keep_dict,
    t_index_name,
    out_dir="final_output",
    n_sur=10000,
    seed=0,
    euclid=False,
    force_recompute_dist=False
):
    ensure_dir(out_dir)

    lh_labels, rh_labels = load_mmp_labels_fslr32k()

    dist_dir = Path(out_dir) / "distmats"
    D = compute_parcel_distance_matrix_fslr32k(
        out_dir=dist_dir,
        lh_labels=lh_labels,
        rh_labels=rh_labels,
        n_parcels=180,
        euclid=euclid,
        force_recompute=force_recompute_dist
    )

    trade_all, alive_mask = build_tradeoff_vector_180(tradeoff_by_label, keep_dict, n_parcels=180)

    gen = Base(x=trade_all, D=D, seed=seed)
    surrogates = gen(n=n_sur)

    for grad_index in [1, 2]:
        pg_mean = compute_pg_mean_fslr32k(grad_index, lh_labels, rh_labels)

        idx = alive_mask & np.isfinite(pg_mean)
        x = pg_mean[idx]
        y = trade_all[idx]

        if x.size < 3:
            print(f"\n[PG{grad_index}] Too few parcels after masking: n={x.size}")
            continue

        pear_r, pear_p = pearsonr(x, y)
        spear_r, spear_p = spearmanr(x, y)

        rho_obs = float(spear_r)
        rho_null = np.array([spearmanr(pg_mean[idx], surrogates[i, idx])[0] for i in range(n_sur)], float)
        p_rho = float(np.mean(np.abs(rho_null) >= np.abs(rho_obs)))

        mode = "euclid" if euclid else "geodesic"
        print(f"\n===== PG{grad_index} × Tradeoff ({t_index_name}) [surface-{mode}-parcellated] =====")
        print(f"Pearson r = {pear_r:.3f}  p = {pear_p:.3e}")
        print(f"Spearman ρ = {spear_r:.3f} p = {spear_p:.3e}")
        print(f"BrainSMASH p (two-sided, Spearman) = {p_rho:.3f}")

        out_png = Path(out_dir) / f"figure_PG{grad_index}_brainsmash_surface_{mode}_parcellated_{t_index_name}.png"
        plot_scatter_and_null(
            x=x,
            y=y,
            null=rho_null,
            obs=rho_obs,
            out_png=str(out_png),
            xlabel=f"PG{grad_index} score",
            ylabel="Tradeoff index",
        )


# =============================================================================
# Main
# =============================================================================
def main():
    SEED = 0

    USE_ZVAL_FOR_GROUP = True
    Q_ALPHA_MASK = 0.01
    EXCLUDE = 0

    IDS = [1, 2, 4, 5, 6, 7, 8, 9]
    MARKERS_SUBJECTS = ["o", "s", "^", "D", "v", "*", "<", "P"]

    ROI_NAMES = [
        "Primary Visual", "Early Visual", "Dorsal Stream Visual", "Ventral Stream Visual", "MT+ Complex",
        "Somatosensory / Motor", "Paracentral / Mid Cingulate", "Premotor", "Posterior Opercular",
        "Early Auditory", "Auditory Association", "Insular / Frontal Opercular", "Medial Temporal",
        "Lateral Temporal", "Temporo-Parieto-Occipital Junction", "Superior Parietal", "Inferior Parietal",
        "Posterior Cingulate", "Anterior Cingulate / Medial Prefrontal", "Orbital / Polar Frontal",
        "Inferior Frontal", "DorsoLateral Prefrontal",
    ]

    ABB_ROI_NAMES = ['V1', 'EVC', 'Dorsal', 'Ventral', 'MT+',
                 'Somatomotor', 'Paracent/MidCing', 'Premotor', 'PostOpercul',
                 'EarlyAudit', 'AuditAssoc', 'Ins/FrontOpercul', 'MedTemp',
                 'LatTemp', 'TPOJ', 'SupPariet', 'InfPariet',
                 'PostCing', 'ACC/MPFC', 'OFC/Polar',
                 'InfFront', 'DLPFC']

    index = int(input("tradeoff index? simple: 1; probit: 2\n> ").strip())

    out_dir = "final_output"
    ensure_dir(out_dir)

    # 1) BigROI tradeoff summary figure
    t_big = run_bigroi_tradeoff(
        index=index,
        ids=IDS,
        markers=MARKERS_SUBJECTS,
        roi_names=ROI_NAMES,
        use_zval_for_group=USE_ZVAL_FOR_GROUP,
        q_alpha_mask=Q_ALPHA_MASK,
        exclude=EXCLUDE,
        base_dir="decoding_summary",
        out_dir=out_dir,
        chance=0.5,
        seed=SEED,
        out_tag_big="bigROI",  # <<< MUST match saving out_tag
    )

    # 1b) BigROI CCGP-vs-XOR scatter
    build_bigroi_scatter_from_csv(
        ids=IDS,
        roi_names=ABB_ROI_NAMES,
        base_dir="decoding_summary",
        exclude=EXCLUDE,
        q_alpha_mask=Q_ALPHA_MASK,
        use_zval_for_group=USE_ZVAL_FOR_GROUP,
        out_png=str(Path(out_dir) / "figure_scatter_ccgpvsxor.png"),
        out_tag_big="bigROI",  # <<< MUST match saving out_tag
    )

    # 2) Fine-grained (annot-based)
    # >>> NEW naming rule (must match saving out_tag for fine-grained)
    out_tag_small = "smallROI"
    csv_files_small = [
        subject_csv_path("decoding_summary", out_tag_small, sid, EXCLUDE)
        for sid in IDS
    ]

    t_small, tradeoff_by_label, keep_dict, _names = run_finegrained_surface(
        index=index,
        sub_ids=IDS,
        csv_files=csv_files_small,
        lh_annot_path="atlas_label/lh.HCP-MMP1.annot",
        rh_annot_path="atlas_label/rh.HCP-MMP1.annot",
        out_dir=out_dir,
        q_alpha=Q_ALPHA_MASK
    )

    # 3) Parcel-wise BrainSMASH + PG
    run_pg_brainsmash_surface_geodesic_parcel(
        tradeoff_by_label=tradeoff_by_label,
        keep_dict=keep_dict,
        t_index_name=t_small,
        out_dir=out_dir,
        n_sur=10000,
        seed=SEED,
        euclid=False,
        force_recompute_dist=False
    )

    print(f"\nDone. BigROI={t_big}, FineROI={t_small}")


if __name__ == "__main__":
    main()
