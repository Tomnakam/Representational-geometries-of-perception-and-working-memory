#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection, multipletests

exclude = 0
atlas = 2#int(input("atlas small=1 big=2 ? "))
value = 2#int(input("logit(acc)=1 z-value=2 ? "))

ids = [1, 2, 4, 5, 6, 7, 8, 9]
markers_subjects = ['o', 's', '^', 'D', 'v', '*', '<', 'P']


def csv_path(prefix, sid, atlas, exclude):
    atlas_tag = "smallROI" if atlas == 1 else "bigROI"
    if (exclude == 1) and (sid in (2, 5)):
        return f"decoding_summary_subcategory/{prefix}_{atlas_tag}_{sid}_exclude.csv"
    return f"decoding_summary_subcategory/{prefix}_{atlas_tag}_{sid}.csv"


def safe_logit(p, eps=1e-6):
    p = np.asarray(p, float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def one_tailed_ttest_1samp_greater(x, popmean=0.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.nan, np.nan
    t = (x.mean() - popmean) / (x.std(ddof=1) / np.sqrt(n))
    p = stats.t.sf(t, df=n - 1)
    return float(t), float(p)


def fdr_bh_qvals(pvals, alpha=0.05):
    pvals = np.asarray(pvals, float)
    mask = np.isfinite(pvals)
    q = np.full_like(pvals, np.nan, dtype=float)
    if mask.sum() > 0:
        _, q_masked, _, _ = multipletests(pvals[mask], alpha=alpha, method='fdr_bh')
        q[mask] = q_masked
    return q


def q_to_stars(qvals):
    q = np.asarray(qvals, float)
    stars = np.zeros_like(q, dtype=int)
    finite = np.isfinite(q)
    stars[finite & (q < 0.05)] = 1
    stars[finite & (q < 0.01)] = 2
    stars[finite & (q < 0.001)] = 3
    return stars


def subject_level_roi_fdr_masks(all_pvals_array, alpha=0.05):
    n_subj, n_roi, n_feat = all_pvals_array.shape
    signif_mask = np.zeros((n_roi, n_feat, n_subj), dtype=bool)
    for f in range(n_feat):
        for s in range(n_subj):
            pvals = all_pvals_array[s, :, f]
            _, q = fdrcorrection(pvals, alpha=alpha)
            signif_mask[:, f, s] = q < alpha
    return signif_mask


def group_level_featurewise_fdr_from_scores(all_scores, alpha=0.05):
    n_subj, n_roi, n_feat = all_scores.shape
    tvals = np.full((n_roi, n_feat), np.nan, dtype=float)
    pvals = np.full((n_roi, n_feat), np.nan, dtype=float)
    qvals = np.full((n_roi, n_feat), np.nan, dtype=float)

    for f in range(n_feat):
        for r in range(n_roi):
            t, p = one_tailed_ttest_1samp_greater(all_scores[:, r, f], popmean=0.0)
            tvals[r, f] = t
            pvals[r, f] = p

        q_f = fdr_bh_qvals(pvals[:, f], alpha=alpha)
        qvals[:, f] = q_f

        means = np.nanmean(all_scores[:, :, f], axis=0)
        qvals[means <= 0.0, f] = np.nan

    stars = q_to_stars(qvals)
    stars[np.isnan(qvals)] = 0
    return tvals, pvals, qvals, stars


roi_names = [
    'Primary Visual', 'Early Visual', 'Dorsal Stream Visual', 'Ventral Stream Visual',
    'MT+ Complex', 'Somatosensory / Motor', 'Paracentral / Mid Cingulate', 'Premotor',
    'Posterior Opercular', 'Early Auditory', 'Auditory Association',
    'Insular / Frontal Opercular', 'Medial Temporal', 'Lateral Temporal',
    'Temporo-Parieto-Occipital Junction', 'Superior Parietal', 'Inferior Parietal',
    'Posterior Cingulate', 'Anterior Cingulate / Medial Prefrontal',
    'Orbital / Polar Frontal', 'Inferior Frontal', 'DorsoLateral Prefrontal'
]

cols_acc_A = ['Face_Accuracy', 'Scene_Accuracy']
cols_z_A = ['Face_z_value', 'Scene_z_value']
pcols_A = ['Face_p_value', 'Scene_p_value']

cols_acc_B = ['Race_Accuracy', 'Sex_Accuracy', 'Naturalness_Accuracy', 'Openness_Accuracy']
cols_z_B = ['Race_z_value', 'Sex_z_value', 'Naturalness_z_value', 'Openness_z_value']
pcols_B = ['Race_p_value', 'Sex_p_value', 'Naturalness_p_value', 'Openness_p_value']


csv_files_A = [csv_path("summary_category_decoding", sid, atlas, exclude) for sid in ids]
all_acc_A, all_z_A, all_pvals_A = [], [], []
roi_names_from_csv = None

for fp in csv_files_A:
    df = pd.read_csv(fp)
    all_acc_A.append(df[cols_acc_A])
    all_z_A.append(df[cols_z_A])
    all_pvals_A.append(df[pcols_A])
    if roi_names_from_csv is None and 'ROI' in df.columns:
        roi_names_from_csv = df['ROI'].tolist()

all_arrayA_acc = np.array([d.values for d in all_acc_A], dtype=float)
all_arrayA_z = np.array([d.values for d in all_z_A], dtype=float)
all_pvals_arrayA = np.array([p.values for p in all_pvals_A], dtype=float)
mean_valuesA = np.nanmean(all_arrayA_acc, axis=0)

if len(roi_names) != mean_valuesA.shape[0] and roi_names_from_csv is not None:
    roi_names = roi_names_from_csv

csv_files_B = [csv_path("summary_detailed_category_decoding", sid, atlas, exclude) for sid in ids]
all_acc_B, all_z_B, all_pvals_B = [], [], []

for fp in csv_files_B:
    df = pd.read_csv(fp)
    all_acc_B.append(df[cols_acc_B])
    all_z_B.append(df[cols_z_B])
    all_pvals_B.append(df[pcols_B])

all_arrayB_acc = np.array([d.values for d in all_acc_B], dtype=float)
all_arrayB_z = np.array([d.values for d in all_z_B], dtype=float)
all_pvals_arrayB = np.array([p.values for p in all_pvals_B], dtype=float)
mean_valuesB = np.nanmean(all_arrayB_acc, axis=0)


alpha = 0.05
signif_maskA = subject_level_roi_fdr_masks(all_pvals_arrayA, alpha=alpha)
signif_maskB = subject_level_roi_fdr_masks(all_pvals_arrayB, alpha=alpha)

if value == 1:
    scoresA = safe_logit(all_arrayA_acc) - safe_logit(0.25)
    scoresB = safe_logit(all_arrayB_acc) - safe_logit(0.5)
elif value == 2:
    scoresA = all_arrayA_z
    scoresB = all_arrayB_z
else:
    raise ValueError("value must be 1 or 2")

tA, pA, qA, starsA = group_level_featurewise_fdr_from_scores(scoresA, alpha=alpha)
tB, pB, qB, starsB = group_level_featurewise_fdr_from_scores(scoresB, alpha=alpha)

n_subjects = len(ids)
n_roi = len(roi_names)
x = np.arange(n_roi)

fig, (ax_top, ax_bot) = plt.subplots(nrows=2, figsize=(16, 10), sharex=True)

labels_top = ['Face', 'Scene']
colors_top = ['#4A9C4E', '#9242B1']
width_top = 0.35

for i in range(2):
    offset = (i - 0.5) * width_top
    ax_top.bar(
        x + offset, mean_valuesA[:, i], width_top,
        label=labels_top[i], color=colors_top[i], edgecolor='black'
    )
    for subj_idx, marker in enumerate(markers_subjects[:n_subjects]):
        for roi_idx in range(n_roi):
            px = x[roi_idx] + offset
            py = all_arrayA_acc[subj_idx, roi_idx, i]
            if signif_maskA[roi_idx, i, subj_idx]:
                ax_top.scatter(px, py, color='black', marker=marker, s=35,
                               edgecolor='black', linewidth=0.5, alpha=0.8, zorder=5)
            else:
                ax_top.scatter(px, py, facecolors='none', edgecolors='black',
                               marker=marker, s=35, linewidth=1.5, alpha=0.8, zorder=5)

ax_top.axhline(0.25, linestyle='--', color='gray', linewidth=1.5, label='Chance level')
ax_top.set_ylim(0.2, 0.45)
ax_top.set_ylabel('Decoding accuracy')
ax_top.grid(True, linestyle='--', alpha=0.5)
ax_top.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=11, frameon=True)
ax_top.text(-0.07, 1.05, "A", transform=ax_top.transAxes,
            ha="left", va="top", fontsize=16, fontname="Arial")

star_base_y = 1.02
row_gap = 0.035
horiz_dx = 0.15

for roi_idx in range(n_roi):
    for i_feat in range(2):
        n_stars = int(starsA[roi_idx, i_feat])
        if n_stars <= 0:
            continue
        x_center = x[roi_idx]
        total_w = (n_stars - 1) * horiz_dx
        x_start = x_center - total_w / 2.0
        y_pos = star_base_y + i_feat * row_gap
        for k in range(n_stars):
            ax_top.text(
                x_start + k * horiz_dx, y_pos, '*',
                ha='center', va='bottom', fontsize=13,
                color=colors_top[i_feat],
                transform=ax_top.get_xaxis_transform(),
                clip_on=False
            )

ax_top.set_xticks(x, roi_names)
plt.setp(ax_top.get_xticklabels(), visible=False)


labels_bot = ['Race', 'Sex', 'Naturalness', 'Openness']
colors_bot = ['#66BB6A', '#2E7D32', '#BA68C8', '#6A1B9A']
N = len(labels_bot)
width_bot = 0.8 / N

for i in range(N):
    offset = (i - (N - 1) / 2) * width_bot
    ax_bot.bar(
        x + offset, mean_valuesB[:, i], width_bot,
        label=labels_bot[i], color=colors_bot[i], edgecolor='black'
    )
    for subj_idx, marker in enumerate(markers_subjects[:n_subjects]):
        for roi_idx in range(n_roi):
            px = x[roi_idx] + offset
            py = all_arrayB_acc[subj_idx, roi_idx, i]
            if signif_maskB[roi_idx, i, subj_idx]:
                ax_bot.scatter(px, py, color='black', marker=marker, s=35,
                               edgecolor='black', linewidth=0.5, alpha=0.85, zorder=5)
            else:
                ax_bot.scatter(px, py, facecolors='none', edgecolors='black',
                               marker=marker, s=35, linewidth=1.5, alpha=0.85, zorder=5)

ax_bot.axhline(0.5, linestyle='--', color='gray', linewidth=1.5, label='Chance level')
ax_bot.set_ylim(0.4, 0.7)
ax_bot.set_ylabel('Decoding accuracy')
ax_bot.set_xticks(x, roi_names, rotation=45, ha='right')
ax_bot.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=11, frameon=True)
ax_bot.text(-0.07, 1.05, "B", transform=ax_bot.transAxes,
            ha="left", va="top", fontsize=16, fontname="Arial")
ax_bot.grid(True, linestyle='--', alpha=0.5)

star_base_y = 1.02
row_gap = 0.03
horiz_dx = 0.15

for roi_idx in range(n_roi):
    for i_cat in range(N):
        n_stars = int(starsB[roi_idx, i_cat])
        if n_stars <= 0:
            continue
        x_center = x[roi_idx]
        total_w = (n_stars - 1) * horiz_dx
        x_start = x_center - total_w / 2.0
        y_pos = star_base_y + i_cat * row_gap
        for k in range(n_stars):
            ax_bot.text(
                x_start + k * horiz_dx, y_pos, '*',
                ha='center', va='bottom', fontsize=13,
                color=colors_bot[i_cat],
                transform=ax_bot.get_xaxis_transform(),
                clip_on=False
            )

ax_top.set_axisbelow(True)
ax_bot.set_axisbelow(True)

plt.tight_layout()
plt.subplots_adjust(top=0.88, right=0.8, hspace=0.25)
plt.savefig("final_output/figure_subcategory_decoding.png", dpi=300, bbox_inches="tight")
plt.show()
