import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import nibabel as nib
from nilearn import plotting
from nilearn.datasets import fetch_surf_fsaverage

from scipy import stats
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests, fdrcorrection
from scipy.stats import ttest_1samp

from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory


exclude = 0
subIDs = [1, 2, 4, 5, 6, 7, 8, 9]

markers_subjects = ['o', 's', '^', 'D', 'v', '*', '<', 'P']

roi_names = [
    'Primary Visual', 'Early Visual', 'Dorsal Stream Visual', 'Ventral Stream Visual', 'MT+ Complex',
    'Somatosensory / Motor', 'Paracentral / Mid Cingulate', 'Premotor', 'Posterior Opercular',
    'Early Auditory', 'Auditory Association', 'Insular / Frontal Opercular', 'Medial Temporal',
    'Lateral Temporal', 'Temporo-Parieto-Occipital Junction', 'Superior Parietal', 'Inferior Parietal',
    'Posterior Cingulate', 'Anterior Cingulate / Medial Prefrontal', 'Orbital / Polar Frontal',
    'Inferior Frontal', 'DorsoLateral Prefrontal'
]

roi_names_2line = [
    'Primary\nVisual',
    'Early\nVisual',
    'Dorsal Stream\nVisual',
    'Ventral Stream\nVisual',
    'MT+ \nComplex',
    'Somatosensory /\nMotor',
    'Paracentral /\nMid Cingulate',
    'Premotor',
    'Posterior\nOpercular',
    'Early\nAuditory',
    'Auditory\nAssociation',
    'Insular /\nFrontal Opercular',
    'Medial\nTemporal',
    'Lateral\nTemporal',
    'Temporo-Parieto-\nOccipital Junction',
    'Superior\nParietal',
    'Inferior\nParietal',
    'Posterior\nCingulate',
    'Anterior Cingulate /\nMedial Prefrontal',
    'Orbital /\nPolar Frontal',
    'Inferior\nFrontal',
    'DorsoLateral\nPrefrontal'
]

ALPHA = 0.05
N_BOOT = 10000
CI = 95
RNG = np.random.default_rng(42)

OUTDIR = "final_output"
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.titlesize": 18
})

fsaverage = fetch_surf_fsaverage(mesh="fsaverage")


def _load_pickle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _use_exclude_name_rule(exclude_value, sid):
    return (float(exclude_value) == 1.0) and (sid in (2, 5))


def angle_path(kind, sid, exclude_value, atlas_tag, base_dir="angle_summary"):
    if kind not in ("z_scores", "empirical", "nf"):
        raise ValueError(f"Unknown kind: {kind}")

    if _use_exclude_name_rule(exclude_value, sid):
        fname = f"{kind}_{sid}_bootstrap_exclude_{atlas_tag}.pkl"
    else:
        fname = f"{kind}_{sid}_bootstrap_{atlas_tag}.pkl"

    return os.path.join(base_dir, fname)


def load_nf_emp(subIDs, exclude, atlas_tag="atlasBIG", base_dir="angle_summary"):
    nf_dict, emp_dict = {}, {}
    for sid in subIDs:
        nf_path = angle_path("nf", sid, exclude, atlas_tag, base_dir=base_dir)
        emp_path = angle_path("empirical", sid, exclude, atlas_tag, base_dir=base_dir)

        nf = np.asarray(_load_pickle(nf_path), dtype=float).ravel()
        emp = np.asarray(_load_pickle(emp_path), dtype=float)

        nf_dict[sid] = nf
        emp_dict[sid] = emp
    return nf_dict, emp_dict


def load_z_matrix(subIDs, exclude, roi_names, atlas_tag="atlasBIG", base_dir="angle_summary"):
    z_list = []
    sid_used = []
    for sid in subIDs:
        z_path = angle_path("z_scores", sid, exclude, atlas_tag, base_dir=base_dir)
        z = np.asarray(_load_pickle(z_path), dtype=float).ravel()
        if z.size != len(roi_names):
            raise ValueError(f"ROI count mismatch for sid={sid}: got {z.size}, expected {len(roi_names)}")
        z_list.append(z)
        sid_used.append(sid)
    z_matrix = np.vstack(z_list)
    return z_matrix, sid_used


def subjectwise_roi_fdr_from_z(z_matrix, alpha=0.05):
    pvals = norm.sf(z_matrix)
    significant = np.zeros_like(z_matrix, dtype=bool)

    for s in range(z_matrix.shape[0]):
        valid = np.isfinite(pvals[s])
        p = pvals[s, valid]
        if p.size == 0:
            continue
        rej, _, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
        significant[s, valid] = rej

    sig_counts = significant.sum(axis=0)

    categories = np.array([""] * z_matrix.shape[1], dtype=object)
    categories[sig_counts >= 5] = "universal"
    categories[sig_counts <= 2] = "non-universal"
    return significant, sig_counts, categories


def roi_stats_combined(z_matrix, roi_names, alpha=0.05, n_boot=10000, ci=95, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    ci_alpha = (100 - ci) / 2.0
    lo_q, hi_q = ci_alpha, 100 - ci_alpha

    rows = []
    for r, roi in enumerate(roi_names):
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

        idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = x[idx].mean(axis=1)
        ci_low = float(np.percentile(boot_means, lo_q))
        ci_high = float(np.percentile(boot_means, hi_q))

        tval, p_two = stats.ttest_1samp(x, popmean=0.0, nan_policy="omit")
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

    pvals = df["p_one_tailed"].to_numpy(float)
    valid = np.isfinite(pvals)

    p_corr = np.full_like(pvals, np.nan)
    reject = np.zeros_like(pvals, dtype=bool)

    if valid.any():
        rej, p_adj, _, _ = multipletests(pvals[valid], alpha=alpha, method="fdr_bh")
        p_corr[valid] = p_adj
        reject[valid] = rej

    df["p_one_tailed_FDR"] = p_corr
    df["significant_FDR"] = reject
    return df


def p_to_stars(p):
    if not np.isfinite(p) or p >= 0.05:
        return ""
    if p < 1e-4:
        return "****"
    elif p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    else:
        return "*"


def make_main_figure(
    z_matrix,
    roi_names,
    markers_subjects,
    significant_matrix_fdr,
    categories,
    df_stats,
    out_png
):
    means = df_stats["mean_z"].to_numpy(dtype=float)
    p_corr = df_stats["p_one_tailed_FDR"].to_numpy(dtype=float)

    x_bar = np.arange(len(roi_names))
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x_bar, means,
        capsize=3,
        color='lightgray',
        edgecolor='black',
        label="Mean"
    )

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    y_axes = 1.02
    for i in range(len(roi_names)):
        stars = p_to_stars(p_corr[i])
        if stars:
            ax.text(
                i, y_axes, stars,
                transform=trans,
                ha='center', va='bottom',
                fontsize=18, fontweight='bold',
                color='black',
                clip_on=False
            )
    fig.subplots_adjust(top=0.90)

    for roi_idx in range(len(roi_names) - 1):
        ax.axvline(roi_idx + 0.5, color='gray', linestyle=':', linewidth=1)

    colors_group = ['black' if subj_idx < 4 else 'dimgray' for subj_idx in range(len(z_matrix))]
    offsets = np.linspace(-0.2, 0.2, len(z_matrix))

    for subj_idx, subj_data in enumerate(z_matrix):
        marker = markers_subjects[subj_idx % len(markers_subjects)]
        color = colors_group[subj_idx]
        x_offset = offsets[subj_idx]

        for roi_idx, val in enumerate(subj_data):
            x_pos = x_bar[roi_idx] + x_offset

            if categories[roi_idx] == "universal":
                edge_c = "blue"
            elif categories[roi_idx] == "non-universal":
                edge_c = "red"
            else:
                edge_c = color

            if significant_matrix_fdr[subj_idx, roi_idx]:
                ax.scatter(
                    x_pos, val, color=color, marker=marker, s=55,
                    edgecolor=edge_c, linewidth=1.2, zorder=10
                )
            else:
                ax.scatter(
                    x_pos, val, facecolors='white', edgecolors=edge_c,
                    marker=marker, s=55, linewidth=1.5, zorder=10
                )

    ax.set_xticks(x_bar)
    ax.set_xticklabels(roi_names, rotation=45, ha='right')
    ax.set_ylabel('Z-score (rotation)', fontsize=14)

    ax.text(-0.05, 1.02, 'A', fontsize=24, transform=ax.transAxes, va='top', ha='left')

    plt.tight_layout()
    ax.set_xlim(-0.5, len(roi_names) - 0.5)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_row(ax, subset_ids, nf_dict, emp_dict, n_roi, sid2marker,
              row_title=None, show_xlabel=False, roi_names_2line=None):
    datasets = []
    nf_lines = []
    positions = []
    subidx_per_pos = []

    n_sub = len(subset_ids)
    pos = 1
    for roi in range(n_roi):
        for si, sid in enumerate(subset_ids):
            emp_vec = np.asarray(emp_dict[sid][roi, :], dtype=float)
            emp_vec = emp_vec[np.isfinite(emp_vec)]
            datasets.append(emp_vec)
            nf_lines.append(float(nf_dict[sid][roi]))
            positions.append(pos)
            subidx_per_pos.append(si)
            pos += 1

    vp = ax.violinplot(datasets, positions=positions, showextrema=False, widths=0.8)
    for b in vp['bodies']:
        b.set_facecolor('lightgray')
        b.set_edgecolor('none')
        b.set_alpha(1)

    for x0, y0, si in zip(positions, nf_lines, subidx_per_pos):
        sid = subset_ids[si]
        ax.scatter(
            x0, y0,
            marker=sid2marker[sid],
            s=50,
            facecolors='white',
            edgecolors='black',
            linewidths=1,
            zorder=4
        )

    for roi in range(1, n_roi):
        ax.axvline(roi * n_sub + 0.5, linestyle=':', linewidth=1)

    orth_line = ax.axhline(90, color='gray', linestyle='--', linewidth=1.5, label='orthogonal')

    ax.set_xlim(min(positions) - 0.5, max(positions) + 0.5)

    if show_xlabel:
        group_centers = [roi * n_sub + 1 + (n_sub - 1) / 2 for roi in range(n_roi)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(
            roi_names_2line,
            rotation=90,
            ha='center',
            multialignment='right'
        )
        ax.tick_params(axis='x', labelsize=14)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])

    ax.tick_params(axis='y', labelsize=14)

    if row_title:
        ax.set_title(row_title, loc='left', fontsize=14, pad=6)

    return orth_line


def make_noise_legend(ax, subset_ids, sid2marker, loc='lower right', ncol=2):
    handles = [
        Line2D(
            [], [], linestyle='None',
            marker=sid2marker[sid], markersize=10,
            markerfacecolor='white',
            markeredgecolor='black',
            label=f"Obs. {sid}"
        )
        for sid in subset_ids
    ]
    return ax.legend(
        handles,
        [f"Obs. {sid}" for sid in subset_ids],
        loc=loc,
        ncol=ncol,
        frameon=True,
        fontsize=12,
        handletextpad=0.4,
        columnspacing=0.8
    )


def make_supple_figure(nf_dict, emp_dict, roi_names_2line, all_subIDs, out_png):
    pilot_ids = all_subIDs[:4]
    new_ids = all_subIDs[4:]

    sid2marker = {sid: markers_subjects[i % len(markers_subjects)] for i, sid in enumerate(all_subIDs)}
    n_roi = next(iter(nf_dict.values())).shape[0]

    n_sub_row = len(pilot_ids)
    n_pos_row = n_roi * n_sub_row
    fig_w = max(14, n_pos_row * 0.23)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, sharey=True,
        figsize=(fig_w, 10), dpi=150,
        constrained_layout=True
    )

    orth_top = build_row(ax_top, pilot_ids, nf_dict, emp_dict, n_roi, sid2marker,
                         row_title="Pilot", show_xlabel=False, roi_names_2line=roi_names_2line)
    orth_bot = build_row(ax_bot, new_ids, nf_dict, emp_dict, n_roi, sid2marker,
                         row_title="New", show_xlabel=True, roi_names_2line=roi_names_2line)

    leg_orth = ax_top.legend([orth_top], ["orthogonal"], loc='upper left', frameon=True)
    ax_top.add_artist(leg_orth)

    make_noise_legend(ax_top, pilot_ids, sid2marker, loc='upper right', ncol=len(pilot_ids))
    make_noise_legend(ax_bot, new_ids, sid2marker, loc='upper right', ncol=len(new_ids))

    fig.supylabel("Angle between P and WM coding axes (deg)", y=0.63)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def surface_visualize(exclude, subIDs, outdir, lh_path, rh_path, atlas_tag="atlasSMALL", base_dir="angle_summary"):
    left_labels_array, _, _ = nib.freesurfer.read_annot(lh_path)
    right_labels_array, _, _ = nib.freesurfer.read_annot(rh_path)

    dfs_z = []
    for sid in subIDs:
        z_path = angle_path("z_scores", sid, exclude, atlas_tag, base_dir=base_dir)
        arr = np.asarray(_load_pickle(z_path)).squeeze()
        dfs_z.append(pd.Series(arr, name=f"sub{sid}"))

    file_stem = "figure_angle_finegrained"
    cbar_label = "t-value"
    z_mat = pd.concat(dfs_z, axis=1).values

    tvals, pvals_two = ttest_1samp(z_mat, popmean=0.0, axis=1, nan_policy='omit')

    pvals_one = pvals_two / 2.0
    pvals_one[tvals < 0] = 1.0

    reject, qvals = fdrcorrection(pvals_one, alpha=0.05)

    mask_all = np.isfinite(tvals)
    tval_1idx = np.concatenate([[np.nan], tvals])

    lh_tex = tval_1idx[left_labels_array]
    mask_left = mask_all[left_labels_array - 1]
    lh_tex_masked = np.where(mask_left, lh_tex, np.nan)

    lh_tex_masked = np.where(lh_tex_masked < 0, 0, lh_tex_masked)

    vmax = np.nanquantile(lh_tex_masked, 0.99)
    vmin = np.nanquantile(lh_tex_masked, 0.01)

    regions_indices = (np.where(reject)[0] + 1).tolist()
    contour_colors = ['magenta'] * len(regions_indices)

    views = ['lateral', 'dorsal', 'anterior', 'medial', 'ventral', 'posterior']

    axes_coords = {
        "anterior":  [0.01, 0.33, 0.44, 0.60],
        "posterior": [0.69, 0.33, 0.44, 0.60],
        "lateral":   [0.21, 0.51, 0.44, 0.5],
        "ventral":   [0.21, 0.25, 0.44, 0.5],
        "medial":    [0.49, 0.51, 0.44, 0.5],
        "dorsal":    [0.49, 0.25, 0.44, 0.5]
    }

    fig = plt.figure(figsize=(18, 12))

    for view in views:
        ax = fig.add_axes(axes_coords[view], projection='3d')

        plotting.plot_surf_stat_map(
            surf_mesh=fsaverage.infl_left,
            stat_map=lh_tex_masked,
            bg_map=fsaverage.sulc_left,
            hemi='left',
            view=view,
            cmap='viridis',
            symmetric_cbar=False,
            vmax=vmax, vmin=vmin,
            colorbar=False,
            axes=ax
        )
        ax.set_facecolor('none')

        if len(regions_indices) > 0:
            plotting.plot_surf_contours(
                surf_mesh=fsaverage.infl_left,
                roi_map=left_labels_array,
                levels=regions_indices,
                figure=fig,
                colors=contour_colors,
                view=view,
                axes=ax
            )

    norm_c = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cbar_ax = fig.add_axes([0.07, 0.45, 0.02, 0.45])
    colorbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap='viridis', norm=norm_c)
    colorbar.set_label(cbar_label, fontsize=25)
    colorbar.ax.tick_params(labelsize=20)

    fig.text(0.1, 0.95, 'B', fontsize=30, va='top', ha='left')

    out_png = os.path.join(outdir, f"{file_stem}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_part_a_angle_pipeline():
    atlas_tag = "atlasBIG"

    nf_dict, emp_dict = load_nf_emp(subIDs=subIDs, exclude=exclude, atlas_tag=atlas_tag, base_dir="angle_summary")
    z_matrix, _ = load_z_matrix(subIDs=subIDs, exclude=exclude, roi_names=roi_names, atlas_tag=atlas_tag, base_dir="angle_summary")

    significant_matrix_fdr, sig_counts, categories = subjectwise_roi_fdr_from_z(z_matrix, alpha=ALPHA)

    df_stats = roi_stats_combined(
        z_matrix=z_matrix,
        roi_names=roi_names,
        alpha=ALPHA,
        n_boot=N_BOOT,
        ci=CI,
        rng=RNG
    )
    df_stats["sig_count_subjectFDR"] = sig_counts
    df_stats["category"] = categories

    out_csv = os.path.join(OUTDIR, "angle_stats.csv")
    df_stats.to_csv(out_csv, index=False)
    print(df_stats)

    out_main_png = os.path.join(OUTDIR, "figure_angle_main.png")
    out_supp_png = os.path.join(OUTDIR, "figure_angle_supple.png")

    make_main_figure(
        z_matrix=z_matrix,
        roi_names=roi_names,
        markers_subjects=markers_subjects,
        significant_matrix_fdr=significant_matrix_fdr,
        categories=categories,
        df_stats=df_stats,
        out_png=out_main_png
    )

    make_supple_figure(
        nf_dict=nf_dict,
        emp_dict=emp_dict,
        roi_names_2line=roi_names_2line,
        all_subIDs=subIDs,
        out_png=out_supp_png
    )


def main():
    run_part_a_angle_pipeline()

    lh_path = os.path.expanduser('atlas_label/lh.HCP-MMP1.annot')
    rh_path = os.path.expanduser('atlas_label/rh.HCP-MMP1.annot')

    surface_visualize(
        exclude=exclude,
        subIDs=subIDs,
        outdir=OUTDIR,
        lh_path=lh_path,
        rh_path=rh_path,
        atlas_tag="atlasSMALL",
        base_dir="angle_summary"
    )

    print("\n[All outputs saved under ./final_output]")


if __name__ == "__main__":
    main()
