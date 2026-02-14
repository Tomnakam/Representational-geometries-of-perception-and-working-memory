#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---- formatting utilities ----

def format_ci(low, high):
    if not (np.isfinite(low) and np.isfinite(high)):
        return "n/a"
    return f"{low:.2f}–{high:.2f}"

def format_mean(mean):
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:.2f}"

def format_t(t):
    if not np.isfinite(t):
        return "n/a"
    return f"{t:.2f}"

def format_df(n):
    if not np.isfinite(n):
        return "n/a"
    df = int(n) - 1
    return f"{df}"

def format_dz(dz):
    if not np.isfinite(dz):
        return "n/a"
    return f"{dz:.2f}"

def format_p_or_q(x):
    if not np.isfinite(x):
        return "n/a"
    if x < 1e-4:
        return " < 1×10⁻⁴"
    elif x < 1e-3:
        s = f"{x:.1e}"
        s = s.replace("e-0", "×10⁻").replace("e-", "×10⁻")
        return f" {s}"
    else:
        return f" {x:.3f}"

def stars_from_p(p):
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "***"
    elif p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def draw_table_for_csv(in_csv, out_png,
                       fig_w=12, fig_h=10,
                       top_y=0.95, dy=0.03):
    """
    Read CSV and render statistical summary as a table-style figure.
    """

    if not os.path.exists(in_csv):
        print(f"[WARN] Not found: {in_csv} → skip")
        return

    df = pd.read_csv(in_csv)
    n_rows = df.shape[0]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # column positions
    x_roi = 0.02
    x_mean = 0.34
    x_ci   = 0.45
    x_dz   = 0.57
    x_t    = 0.65
    x_df   = 0.72
    x_p    = 0.81
    x_q    = 0.95

    # header line
    ax.hlines(top_y - dy*0.5, 0.0, 1.0, linewidth=1.6, color="black")

    # header text
    ax.text(x_roi, top_y, "ROI", ha="left", va="center", fontsize=15, weight="bold")
    ax.text(x_mean, top_y, "Mean z", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(x_ci,  top_y, "95% CI", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(x_dz,  top_y, "Cohen's dz", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(x_t,   top_y, "t", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(x_df,  top_y, "df", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(x_p,   top_y, "p", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(x_q,   top_y, "q", ha="right", va="center", fontsize=15, weight="bold")

    # table rows
    y = top_y - dy
    for _, row in df.iterrows():
        roi = row["ROI"]
        mean_z = row["mean_z"]
        ci_low = row["CI_low"]
        ci_high = row["CI_high"]
        tval = row["t_value"]
        n = row["N_subj_used"]
        p = row["p_one_tailed"]
        q = row["p_one_tailed_FDR"]

        # Cohen's dz from t and N
        if np.isfinite(tval) and np.isfinite(n) and n > 0:
            dz = tval / np.sqrt(n)
        else:
            dz = np.nan

        stars = stars_from_p(q)

        ax.text(x_roi,  y, roi,                 ha="left",   va="center", fontsize=13)
        ax.text(x_mean, y, format_mean(mean_z), ha="center", va="center", fontsize=13)
        ax.text(x_ci,   y, format_ci(ci_low, ci_high), ha="center", va="center", fontsize=13)
        ax.text(x_dz,   y, format_dz(dz),      ha="center", va="center", fontsize=13)
        ax.text(x_t,    y, format_t(tval),     ha="center", va="center", fontsize=13)
        ax.text(x_df,   y, format_df(n),       ha="center", va="center", fontsize=13)

        # significance stars (based on q)
        ax.text(x_q - 0.06, y, stars, ha="center", va="center", fontsize=13)

        # p / q
        ax.text(x_p, y, format_p_or_q(p), ha="center", va="center", fontsize=13)
        ax.text(x_q, y, format_p_or_q(q), ha="center", va="center", fontsize=13)

        y -= dy

    # bottom line
    ax.hlines(y + dy * 0.6, 0.0, 1.0, linewidth=1.6, color="black")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {in_csv} → {out_png}")


if __name__ == "__main__":
    # input/output pairs
    pairs = [
        ("final_output/angle_stats.csv",
         "final_output/figure_angle_table.png"),

        ("final_output/decoding_stats_feature.csv",
         "final_output/figure_feature_table.png"),

        ("final_output/decoding_stats_ccgp.csv",
         "final_output/figure_ccgp_table.png"),

        ("final_output/decoding_stats_xor.csv",
         "final_output/figure_xor_table.png"),
    ]

    for in_csv, out_png in pairs:
        draw_table_for_csv(in_csv, out_png)
