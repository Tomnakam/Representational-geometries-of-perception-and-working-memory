#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from statsmodels.stats.multitest import fdrcorrection, multipletests
from scipy.special import logit
from scipy.stats import ttest_1samp

# =========================
# Global constants
# =========================
EPS = 1e-6

USE_ZVAL = True  # True: group tests use z-values vs 0; False: group tests use accuracy (logit) vs chance

exclude = 0

ids = [1, 2, 4, 5, 6, 7, 8, 9]
markers_subjects = ['o', 's', '^', 'D', 'v', '*', '<', 'P']
n_subjects = len(ids)

roi_names = [
    'Primary Visual', 'Early Visual', 'Dorsal Stream Visual', 'Ventral Stream Visual', 'MT+ Complex',
    'Somatosensory / Motor', 'Paracentral / Mid Cingulate', 'Premotor', 'Posterior Opercular',
    'Early Auditory', 'Auditory Association', 'Insular / Frontal Opercular', 'Medial Temporal',
    'Lateral Temporal', 'Temporo-Parieto-Occipital Junction', 'Superior Parietal', 'Inferior Parietal',
    'Posterior Cingulate', 'Anterior Cingulate / Medial Prefrontal', 'Orbital / Polar Frontal',
    'Inferior Frontal', 'DorsoLateral Prefrontal'
]
n_roi = len(roi_names)
x = np.arange(n_roi)

CHANCE = {
    "Feature": 0.5,
    "CCGP": 0.5,
    "XOR": 0.5,
    "Task": 0.5,
}

ALPHA = 0.05
N_BOOT = 10000
CI = 95
RNG = np.random.default_rng(42)

# =========================
# Column definitions
# =========================
acc_columns = [
    'Feature_Decoding_Accuracy', 'CCGP_Feature_Decoding_Accuracy', 'XOR_Decoding_Accuracy',
    'Task_Decoding_Accuracy', 'CCGP_Task_Decoding_Accuracy',
    'Feature_Decoding_Accuracy(P)', 'Feature_Decoding_Accuracy(WM)'
]
pval_columns = [
    'Feature_Decoding_pval', 'CCGP_Feature_Decoding_pval', 'XOR_Decoding_pval',
    'Task_Decoding_pval', 'CCGP_Task_Decoding_pval',
    'Feature_Decoding_pval(P)', 'Feature_Decoding_pval(WM)'
]
zval_columns = [
    'Feature_Decoding_zval', 'CCGP_Feature_Decoding_zval', 'XOR_Decoding_zval',
    'Task_Decoding_zval', 'CCGP_Task_Decoding_zval',
    'Feature_Decoding_zval(P)', 'Feature_Decoding_zval(WM)'
]

# =========================
# Helpers
# =========================
def q_to_stars(q: float) -> str:
    if (q is None) or (not np.isfinite(q)) or (q >= 0.05):
        return ""
    if q < 1e-4:
        return "****"
    if q < 1e-3:
        return "***"
    if q < 1e-2:
        return "**"
    return "*"


def safe_logit(p):
    p = np.asarray(p, float)
    p = np.clip(p, EPS, 1.0 - EPS)
    return logit(p)


def subject_csv_path(out_tag: str, sid: int, exclude_value: float, base_dir="decoding_summary") -> Path:
    """
    Use *_exclude.csv only when exclude==1 and sid in (2, 5).
    Otherwise use the standard file.
    """
    base = Path(base_dir)
    if (exclude_value == 1) and (sid in (2, 5)):
        return base / f"summary_{out_tag}_{sid}_exclude.csv"
    return base / f"summary_{out_tag}_{sid}.csv"


def load_subject_dfs(subject_ids, exclude_value, out_tag, base_dir="decoding_summary"):
    csv_files = [subject_csv_path(out_tag, sid, exclude_value, base_dir=base_dir) for sid in subject_ids]
    dfs = []
    for p in csv_files:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")
        dfs.append(pd.read_csv(p))
    return dfs


def stack_arrays(dfs, value_cols, p_cols):
    """
    dfs: list of per-subject DataFrames
    value_cols: accuracy columns
    p_cols: p-value columns
    """
    all_vals = np.array([df[value_cols].values for df in dfs])   # (subj, roi, K)
    all_pvals = np.array([df[p_cols].values for df in dfs])      # (subj, roi, K)
    mean_vals = np.nanmean(all_vals, axis=0)                     # (roi, K)
    return all_vals, all_pvals, mean_vals


def subject_level_fdr_mask(all_pvals_array, col_indices, alpha=0.05):
    """
    For each subject, apply FDR correction across ROIs and
    return a mask indicating significance per ROI × condition × subject.
    """
    n_subj, n_roi_local, _ = all_pvals_array.shape
    n_cond = len(col_indices)
    mask = np.zeros((n_roi_local, n_cond, n_subj), dtype=bool)
    for j, col in enumerate(col_indices):
        for s in range(n_subj):
            p = all_pvals_array[s, :, col]
            _, q = fdrcorrection(p, alpha=alpha)
            mask[:, j, s] = (q < alpha)
    return mask


def group_one_sample_test(dfs, roi_names_local, value_col, chance=0.5,
                          use_zval=False, alpha=0.05):
    """
    Group-level one-sample t-test (one-tailed, >0 or >chance) per ROI.
    If use_zval=True, test z-values vs 0.
    If use_zval=False, test logit-accuracy vs logit(chance).
    """
    n_roi_local = len(roi_names_local)
    data = []
    for df in dfs:
        v = df[value_col].values.astype(float)
        if use_zval:
            data.append(v)
        else:
            data.append(safe_logit(v))
    mat = np.vstack(data)  # (subj, roi)

    popmean = 0.0 if use_zval else float(logit(chance))
    pvals = np.full(n_roi_local, np.nan)
    tstats = np.full(n_roi_local, np.nan)

    for r in range(n_roi_local):
        vec = mat[:, r]
        vec = vec[np.isfinite(vec)]
        if vec.size < 2:
            continue
        # one-tailed, alternative="greater"
        t, p = ttest_1samp(vec, popmean=popmean, nan_policy='omit', alternative='greater')
        tstats[r] = t
        pvals[r] = p

    finite = np.isfinite(pvals)
    reject = np.zeros(n_roi_local, dtype=bool)
    qvals = np.full(n_roi_local, np.nan)
    if finite.any():
        rej_sub, q_sub = fdrcorrection(pvals[finite], alpha=alpha)
        reject[finite] = rej_sub
        qvals[finite] = q_sub

    return pvals, qvals, tstats, reject


def print_sig_rois(title, roi_names_local, pvals, qvals, tstats, reject):
    sig = np.where(reject)[0]
    print(title)
    print(f"ROIs with FDR q < .05 ({len(sig)}):")
    for i in sig:
        print(f"  {roi_names_local[i]:40s}  t={tstats[i]:6.3f}, p={pvals[i]:.5g}, q={qvals[i]:.5g}")


def roi_stats_combined(z_matrix, roi_names_local, alpha=0.05,
                       n_boot=10000, ci=95, rng=None):
    """
    Compute ROI-wise stats (mean_z, bootstrap CI, t-value, one-tailed p, FDR),
    using the same logic as angle_stats.csv.

    Returns a DataFrame with columns:
      ROI, N_subj_used, mean_z, CI_low, CI_high,
      t_value, p_one_tailed, p_one_tailed_FDR, significant_FDR
    """
    if rng is None:
        rng = np.random.default_rng(0)

    ci_alpha = (100 - ci) / 2.0
    lo_q, hi_q = ci_alpha, 100 - ci_alpha

    rows = []
    for r, roi in enumerate(roi_names_local):
        x = z_matrix[:, r]
        x = x[np.isfinite(x)]
        n = x.size

        if n == 0:
            rows.append(dict(
                ROI=roi, N_subj_used=0,
                mean_z=np.nan, CI_low=np.nan, CI_high=np.nan,
                t_value=np.nan, p_one_tailed=np.nan
            ))
            continue

        mean_z = float(x.mean())

        # bootstrap CI
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = x[idx].mean(axis=1)
        ci_low = float(np.percentile(boot_means, lo_q))
        ci_high = float(np.percentile(boot_means, hi_q))

        # t-test vs 0 (two-tailed → convert to one-tailed)
        tval, p_two = ttest_1samp(x, popmean=0.0, nan_policy="omit")
        tval = float(tval)
        if mean_z > 0:
            p_one = float(p_two / 2.0)
        else:
            p_one = float(1 - (p_two / 2.0))

        rows.append(dict(
            ROI=roi, N_subj_used=int(n),
            mean_z=mean_z, CI_low=ci_low, CI_high=ci_high,
            t_value=tval, p_one_tailed=p_one
        ))

    df = pd.DataFrame(rows)

    # FDR correction across ROIs
    pvals = df["p_one_tailed"].to_numpy(float)
    valid = np.isfinite(pvals)

    p_corr = np.full_like(pvals, np.nan, dtype=float)
    reject = np.zeros_like(pvals, dtype=bool)

    if valid.any():
        rej, p_adj, _, _ = multipletests(pvals[valid], alpha=alpha, method="fdr_bh")
        p_corr[valid] = p_adj
        reject[valid] = rej

    df["p_one_tailed_FDR"] = p_corr
    df["significant_FDR"] = reject
    return df


def compute_decoding_roi_stats_z(dfs, roi_names_local, outdir,
                                 alpha=ALPHA, n_boot=N_BOOT, ci=CI, rng=RNG):
    """
    For Feature / CCGP / XOR z-values, compute ROI-wise stats
    (same scheme as angle_stats) and save them as CSV.

    Outputs:
      decoding_stats_feature.csv
      decoding_stats_ccgp.csv
      decoding_stats_xor.csv
    """
    def _build_z_matrix(colname):
        return np.vstack([df[colname].to_numpy(dtype=float) for df in dfs])

    configs = [
        ("Feature_Decoding_zval",        "decoding_stats_feature.csv"),
        ("CCGP_Feature_Decoding_zval",   "decoding_stats_ccgp.csv"),
        ("XOR_Decoding_zval",            "decoding_stats_xor.csv"),
    ]

    for colname, fname in configs:
        z_mat = _build_z_matrix(colname)
        df_stats = roi_stats_combined(
            z_matrix=z_mat,
            roi_names_local=roi_names_local,
            alpha=alpha,
            n_boot=n_boot,
            ci=ci,
            rng=rng
        )
        df_stats.to_csv(outdir / fname, index=False)

# =========================
# I/O setup
# =========================
outdir = Path("final_output")
outdir.mkdir(parents=True, exist_ok=True)

# =========================
# Load data (plot always uses accuracy)
# =========================
# IMPORTANT:
#   This assumes your saving rule:
#     summary_{out_tag}_{subID}_exclude.csv   (only when exclude==1 and subID in (2,5))
#     summary_{out_tag}_{subID}.csv           (otherwise)
#
# Set out_tag here to whatever you used when saving.
out_tag = "bigROI"  # <-- change as needed (e.g., "smallROI", "atlasBIG")

dfs = load_subject_dfs(ids, exclude, out_tag=out_tag, base_dir="decoding_summary")

all_array, all_pvals_array, mean_values = stack_arrays(dfs, acc_columns, pval_columns)

# =========================
# Angle-style ROI stats for Feature / CCGP / XOR (z-values)
# =========================
compute_decoding_roi_stats_z(dfs, roi_names, outdir)

print("\n[Saved decoding_stats_feature_z.csv, decoding_stats_ccgp_z.csv, decoding_stats_xor_z.csv]")

# =========================
# Figure 1: Feature / CCGP / XOR + ratio
# =========================
bar_cols = [0, 1, 2]
bar_labels = ['Face vs Scene', 'CCGP (across P and WM)', 'XOR']
bar_colors = ['skyblue', 'deepskyblue', 'mediumpurple']
width = 0.25
N = len(bar_cols)

subj_mask = subject_level_fdr_mask(all_pvals_array, bar_cols, alpha=0.05)

if USE_ZVAL:
    p_feat, q_feat, t_feat, rej_feat = group_one_sample_test(
        dfs, roi_names, value_col="Feature_Decoding_zval",
        chance=CHANCE["Feature"], use_zval=True, alpha=0.05
    )
    p_ccgp, q_ccgp, t_ccgp, rej_ccgp = group_one_sample_test(
        dfs, roi_names, value_col="CCGP_Feature_Decoding_zval",
        chance=CHANCE["CCGP"], use_zval=True, alpha=0.05
    )
    p_xor, q_xor, t_xor, rej_xor = group_one_sample_test(
        dfs, roi_names, value_col="XOR_Decoding_zval",
        chance=CHANCE["XOR"], use_zval=True, alpha=0.05
    )
else:
    p_feat, q_feat, t_feat, rej_feat = group_one_sample_test(
        dfs, roi_names, value_col="Feature_Decoding_Accuracy",
        chance=CHANCE["Feature"], use_zval=False, alpha=0.05
    )
    p_ccgp, q_ccgp, t_ccgp, rej_ccgp = group_one_sample_test(
        dfs, roi_names, value_col="CCGP_Feature_Decoding_Accuracy",
        chance=CHANCE["CCGP"], use_zval=False, alpha=0.05
    )
    p_xor, q_xor, t_xor, rej_xor = group_one_sample_test(
        dfs, roi_names, value_col="XOR_Decoding_Accuracy",
        chance=CHANCE["XOR"], use_zval=False, alpha=0.05
    )

print_sig_rois(f"Group-level one-sample test: Feature ({'z' if USE_ZVAL else 'acc(logit)'}; one-tailed)", roi_names,
               p_feat, q_feat, t_feat, rej_feat)
print_sig_rois(f"Group-level one-sample test: CCGP ({'z' if USE_ZVAL else 'acc(logit)'}; one-tailed)", roi_names,
               p_ccgp, q_ccgp, t_ccgp, rej_ccgp)
print_sig_rois(f"Group-level one-sample test: XOR ({'z' if USE_ZVAL else 'acc(logit)'}; one-tailed)", roi_names,
               p_xor, q_xor, t_xor, rej_xor)

qvals_group = np.vstack([q_feat, q_ccgp, q_xor]).T

acc_cols_needed = [
    'CCGP_Feature_Decoding_Accuracy',
    'Feature_Decoding_Accuracy(P)',
    'Feature_Decoding_Accuracy(WM)',
]
acc_arrays = np.array([df[acc_cols_needed].values for df in dfs])
ccgp_feature = acc_arrays[:, :, 0]
feature_P = acc_arrays[:, :, 1]
feature_WM = acc_arrays[:, :, 2]
feature_mean = (feature_P + feature_WM) / 2.0
ccgp_feature_ratio = ccgp_feature / feature_mean

ccgp_feature_ratio[:, ~rej_feat] = np.nan
mean_ratio = np.nanmean(ccgp_feature_ratio, axis=0)
sem_ratio = np.nanstd(ccgp_feature_ratio, axis=0, ddof=1) / np.sqrt(n_subjects)

n_pilot = 4
subj_colors = ['black' if s < n_pilot else 'dimgray' for s in range(n_subjects)]

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(18, 8.5), sharex=True,
    gridspec_kw={'height_ratios': [3.1, 1.4], 'hspace': 0.10}
)

for j, col in enumerate(bar_cols):
    offset = (j - (N - 1) / 2) * width

    ax1.bar(
        x + offset, mean_values[:, col], width,
        label=bar_labels[j], color=bar_colors[j],
        edgecolor='black', linewidth=0.5
    )

    for s, marker in enumerate(markers_subjects[:n_subjects]):
        c = subj_colors[s]
        for r in range(n_roi):
            px = x[r] + offset
            py = all_array[s, r, col]
            if subj_mask[r, j, s]:
                ax1.scatter(px, py, facecolors=c, edgecolors=c,
                            marker=marker, s=35, linewidth=0.8, alpha=0.85, zorder=5)
            else:
                ax1.scatter(px, py, facecolors='none', edgecolors=c,
                            marker=marker, s=35, linewidth=1.5, alpha=0.85, zorder=5)

ax1.set_ylabel('Decoding accuracy', fontsize=20)
ax1.axhline(0.5, linestyle='--', color='gray', linewidth=1.5)
ax1.set_ylim(0.4, 0.9)
yt = np.round(np.arange(0.4, 0.91, 0.1), 1)
ax1.set_yticks(yt)
ax1.set_yticklabels([f"{t:.1f}" for t in yt], fontsize=16)
ax1.grid(True, linestyle='--', alpha=0.5)

ymin, ymax = ax1.get_ylim()
star_margin = (ymax - ymin) * 0.025
star_base = ymax + star_margin * 0.8
star_step = star_margin * 0.9
for r in range(n_roi):
    for j in range(3):
        stars = q_to_stars(qvals_group[r, j])
        if stars:
            ax1.text(x[r], star_base + j * star_step, stars,
                     ha='center', va='bottom', fontsize=20,
                     color=bar_colors[j], clip_on=False)

filled = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                       markersize=8, label='Significant')
outline = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                        markersize=8, markerfacecolor='none', label='Not significant')
handles, labels_ = ax1.get_legend_handles_labels()
handles += [filled, outline]
ax1.legend(handles, labels_, loc='upper right',
           frameon=True, fontsize=15, facecolor='white',
           framealpha=0.8, edgecolor='gray', borderaxespad=0.8)

bars = ax2.bar(x, mean_ratio, color='gray', width=0.50, edgecolor='black', linewidth=0.5)

for s in range(n_subjects):
    marker = markers_subjects[s % len(markers_subjects)]
    ax2.scatter(x, ccgp_feature_ratio[s], color='black', marker=marker, s=18, alpha=0.8)

for r, bar in enumerate(bars):
    if not rej_feat[r]:
        ax2.text(bar.get_x() + bar.get_width() / 2, 0.6, 'N/A',
                 ha='center', va='bottom', fontsize=18, color='black')

ax2.set_ylabel('Cross-decoding ratio', fontsize=20)
ax2.axhline(1, linestyle='--', color='gray', alpha=0.8)
ax2.set_ylim(0.6, 1.1)
yt2 = np.round(np.arange(0.6, 1.11, 0.1), 1)
ax2.set_yticks(yt2)
ax2.set_yticklabels([f"{t:.1f}" for t in yt2], fontsize=16)
ax2.grid(True, linestyle='--', alpha=0.5)

ax2.set_xticks(x)
ax2.set_xticklabels(roi_names, rotation=45, ha='right', fontsize=16)

ax1.margins(x=0.01)
ax2.margins(x=0.0)
fig.subplots_adjust(top=0.90, bottom=0.12, hspace=0.1)

plt.savefig(outdir / "figure_decoding_main.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# =========================
# Figure 2/3: Task summary & Feature(P/WM) summary
# =========================
plot_configs = [
    {
        "name": "Task",
        "cols": [3, 4],
        "labels": ["Perception vs Working Memory", "CCGP (Perception vs Working Memory)"],
        "colors": ["lightgray", "dimgray"],
        "outfile": outdir / "figure_decoding_supple_task.png",
        "chance": CHANCE["Task"]
    },
    {
        "name": "Feature(P/WM)",
        "cols": [5, 6],
        "labels": ["Feature decoding (Perception only)", "Feature decoding (Working Memory only)"],
        "colors": ["lightgray", "dimgray"],
        "outfile": outdir / "figure_decoding_supple_feature_separate.png",
        "chance": CHANCE["Feature"]
    }
]

for cfg in plot_configs:
    cols = cfg["cols"]
    labels_plot = cfg["labels"]
    colors_plot = cfg["colors"]
    N_plot = len(cols)
    width_plot = 0.30

    subj_mask_plot = subject_level_fdr_mask(all_pvals_array, cols, alpha=0.05)

    qvals_mat = np.full((n_roi, N_plot), np.nan)
    for j, col in enumerate(cols):
        if USE_ZVAL:
            value_col = zval_columns[col]
            p, q, t, rej = group_one_sample_test(
                dfs, roi_names, value_col=value_col,
                chance=cfg["chance"], use_zval=True, alpha=0.05
            )
        else:
            value_col = acc_columns[col]
            p, q, t, rej = group_one_sample_test(
                dfs, roi_names, value_col=value_col,
                chance=cfg["chance"], use_zval=False, alpha=0.05
            )
        qvals_mat[:, j] = q

    fig, ax = plt.subplots(figsize=(18, 5))

    for j, col in enumerate(cols):
        offset = (j - (N_plot - 1) / 2) * width_plot

        ax.bar(
            x + offset, mean_values[:, col], width_plot,
            label=labels_plot[j], color=colors_plot[j],
            edgecolor='black', linewidth=0.5
        )

        for s, marker in enumerate(markers_subjects[:n_subjects]):
            for r in range(n_roi):
                px = x[r] + offset
                py = all_array[s, r, col]
                if subj_mask_plot[r, j, s]:
                    ax.scatter(px, py, color='black', marker=marker, s=35,
                               edgecolor='black', linewidth=0.5, alpha=0.85, zorder=5)
                else:
                    ax.scatter(px, py, facecolors='none', edgecolors='black',
                               marker=marker, s=35, linewidth=1.5, alpha=0.85, zorder=5)

    ax.set_ylabel('Decoding accuracy', fontsize=16)
    ax.axhline(0.5, linestyle='--', color='gray', linewidth=1.5)
    ax.set_ylim(0.4, 1.0)
    yt = np.round(np.arange(0.4, 1.01, 0.1), 1)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{t:.1f}" for t in yt], fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, rotation=90, fontsize=13)

    ymin, ymax = ax.get_ylim()
    star_margin = (ymax - ymin) * 0.03
    star_base = ymax + star_margin * 0.6
    star_step = star_margin * 0.9
    for r in range(n_roi):
        for j in range(N_plot):
            stars = q_to_stars(qvals_mat[r, j])
            if stars:
                ax.text(x[r], star_base + j * star_step, stars,
                        ha='center', va='bottom', fontsize=10,
                        color='black', clip_on=False)

    filled = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                           markersize=8, label='Significant')
    outline = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                            markersize=8, markerfacecolor='none', label='Not significant')
    handles, labels_ = ax.get_legend_handles_labels()
    handles += [filled, outline]
    ax.legend(handles, labels_,
              loc='lower left',
              bbox_to_anchor=(-0.02, -0.9),
              frameon=True, fontsize=11,
              facecolor='white', framealpha=0.8,
              edgecolor='gray', borderaxespad=0.8)

    fig.subplots_adjust(bottom=0.22, top=0.88)
    fig.savefig(cfg["outfile"], dpi=300, bbox_inches="tight")
    plt.close(fig)

print("\n[All decoding figures and CSV stats saved under ./final_output]")
