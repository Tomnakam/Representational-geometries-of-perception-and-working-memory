#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


# =========================
# Config
# =========================
SUB_IDS = [1, 2, 4, 5, 6, 7, 8, 9]
DATA_TEMPLATE = "rawdata/rawdata_PRMdecoding_exp1_{subID}.mat"

# Fixed RNG seed for full determinism (bootstrap + jitter)
SEED = 0

# =========================
# Core computation
# =========================
def process_subject(subID: int, rng: np.random.Generator):
    """
    Returns:
        result_sub: (2,3) float array [cueType, (correct, incorrect, no)]
        count_sub:  (2,3) int array   [cueType, (correct, incorrect, no)]
        rt_rows: list of tuples (rt, subject_label, cue_label)
    """
    mat_data = sio.loadmat(DATA_TEMPLATE.format(subID=subID))
    trial = mat_data["trial"]
    nTrial, nVar, nRun = trial.shape  # noqa: F841

    result_sub = np.zeros((2, 3), dtype=float)
    count_sub = np.zeros((2, 3), dtype=int)
    rt_rows = []

    for cueIdx in (1, 2):  # 1: face, 2: scene
        meanCorrect = []
        meanNo = []
        countCorrect = []
        countNo = []

        for run in range(nRun):
            cue_col = trial[:, 1, run].astype(int)
            col11 = trial[:, 10, run]
            rt_col = trial[:, 11, run]
            mask = (cue_col == cueIdx)

            odd_counts = np.sum(np.mod(col11[mask], 2) == 1)
            nan_counts = np.sum(np.isnan(col11[mask])) - 4

            is_bad_run_sub2 = (subID == 2 and (run + 1 in [1, 17]))

            meanCorrect.append(odd_counts / 4)
            meanNo.append((nan_counts - is_bad_run_sub2) / 4)

            countCorrect.append(odd_counts)
            countNo.append(nan_counts - is_bad_run_sub2)

            mask_rt = mask & (np.mod(col11, 2) == 1)
            if not is_bad_run_sub2:
                rts = rt_col[mask_rt]
                rt_rows.extend(
                    (float(rt), f"Obs. {subID}", "face" if cueIdx == 1 else "scene")
                    for rt in rts
                )

        if subID == 2:
            valid_runs = [r for r in range(nRun) if (r + 1) not in [1, 17]]
            meanCorrect = [meanCorrect[r] for r in valid_runs]
            meanNo = [meanNo[r] for r in valid_runs]
            countCorrect = [countCorrect[r] for r in valid_runs]
            countNo = [countNo[r] for r in valid_runs]

        meanC = float(np.mean(meanCorrect))
        meanN = float(np.mean(meanNo))
        meanI = float(1.0 - meanC - meanN)
        result_sub[cueIdx - 1, :] = [meanC, meanI, meanN]

        totalC = int(np.sum(countCorrect))
        totalN = int(np.sum(countNo))
        totalI = int((len(countCorrect) * 4) - totalC - totalN)
        count_sub[cueIdx - 1, :] = [totalC, totalI, totalN]

    return result_sub, count_sub, rt_rows


def build_all(subIDs, seed=0):
    rng = np.random.default_rng(seed)

    nSub = len(subIDs)
    resultMat = np.zeros((nSub, 2, 3), dtype=float)  # (subject, cueType, [correct, incorrect, no])
    countMat = np.zeros((nSub, 2, 3), dtype=int)

    allRT, allLabel, allCue = [], [], []

    for i, subID in enumerate(subIDs):
        result_sub, count_sub, rt_rows = process_subject(subID=subID, rng=rng)
        resultMat[i, :, :] = result_sub
        countMat[i, :, :] = count_sub

        for rt, lab, cue in rt_rows:
            allRT.append(rt)
            allLabel.append(lab)
            allCue.append(cue)

    df = pd.DataFrame({"RT": allRT, "Subject": allLabel, "Cue": allCue})
    return resultMat, countMat, df, rng


# =========================
# Stats
# =========================
def paired_acc_test(resultMat):
    nSub = resultMat.shape[0]
    face_acc = resultMat[:, 0, 0]
    scene_acc = resultMat[:, 1, 0]
    tval, pval = ttest_rel(face_acc, scene_acc)
    return tval, pval, face_acc, scene_acc, nSub


def rt_group_paired_geommean(df):
    epsilon = 1e-6
    subIDs = sorted(df["Subject"].unique())
    face_means_log, scene_means_log = [], []

    for sub in subIDs:
        rt_face = df[(df["Subject"] == sub) & (df["Cue"] == "face")]["RT"].to_numpy()
        rt_scene = df[(df["Subject"] == sub) & (df["Cue"] == "scene")]["RT"].to_numpy()
        face_means_log.append(float(np.log(rt_face + epsilon).mean()))
        scene_means_log.append(float(np.log(rt_scene + epsilon).mean()))

    tval, pval = ttest_rel(face_means_log, scene_means_log)

    face_means_geom = np.exp(face_means_log)
    scene_means_geom = np.exp(scene_means_log)

    return (
        float(tval),
        float(pval),
        face_means_geom,
        scene_means_geom,
        subIDs,
    )


# =========================
# Plot
# =========================
def plot_behavior(resultMat, df, face_means_geom, scene_means_geom, sub_list, rng, out_png):
    mean_incorrect_face = float(np.mean(resultMat[:, 0, 1]))
    mean_noresp_face = float(np.mean(resultMat[:, 0, 2]))
    mean_incorrect_scene = float(np.mean(resultMat[:, 1, 1]))
    mean_noresp_scene = float(np.mean(resultMat[:, 1, 2]))

    individual_incorrect_face = resultMat[:, 0, 1]
    individual_noresp_face = resultMat[:, 0, 2]
    individual_incorrect_scene = resultMat[:, 1, 1]
    individual_noresp_scene = resultMat[:, 1, 2]

    individuals_face = [individual_incorrect_face, individual_noresp_face]
    individuals_scene = [individual_incorrect_scene, individual_noresp_scene]

    group_face_mean = float(np.mean(face_means_geom))
    group_scene_mean = float(np.mean(scene_means_geom))

    markers_subjects = ["o", "s", "^", "D", "v", "*", "<", "P"]
    marker_map = {sub: markers_subjects[i] for i, sub in enumerate(sub_list)}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2, 1]},
    )
    ax1, ax2 = axes

    x = np.array([0, 1], dtype=float)
    bar_width = 0.35

    means_face = [mean_incorrect_face, mean_noresp_face]
    means_scene = [mean_incorrect_scene, mean_noresp_scene]

    ax1.bar(x - bar_width / 2, means_face, width=bar_width, label="Face", color="0.3")
    ax1.bar(x + bar_width / 2, means_scene, width=bar_width, label="Scene", color="0.7")

    jitter = 0.04
    for i in range(2):
        for sub_idx, val in enumerate(individuals_face[i]):
            sub = sub_list[sub_idx]
            ax1.plot(
                x[i] - bar_width / 2 + rng.uniform(-jitter, jitter),
                float(val),
                marker_map[sub],
                color="black",
                alpha=0.8,
            )
        for sub_idx, val in enumerate(individuals_scene[i]):
            sub = sub_list[sub_idx]
            ax1.plot(
                x[i] + bar_width / 2 + rng.uniform(-jitter, jitter),
                float(val),
                marker_map[sub],
                color="black",
                alpha=0.8,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Incorrect", "No response"], fontsize=14)
    ax1.set_ylabel("Proportion", fontsize=14)
    ax1.set_ylim(0, 0.2)
    ax1.tick_params(axis="y", labelsize=14)
    ax1.grid(False)
    ax1.legend(fontsize=12)

    x2 = np.array([0, 1], dtype=float)
    ax2.bar(
        x2,
        [group_face_mean, group_scene_mean],
        capsize=5,
        width=0.6,
        color=["0.3", "0.7"],
        edgecolor="black",
        linewidth=1.0,
    )

    jitter2 = 0.04
    for sub, f_g, s_g in zip(sub_list, face_means_geom, scene_means_geom):
        ax2.scatter(
            x2[0] + rng.uniform(-jitter2, jitter2),
            float(f_g),
            s=60,
            marker=marker_map[sub],
            color="0.0",
            zorder=3,
        )
        ax2.scatter(
            x2[1] + rng.uniform(-jitter2, jitter2),
            float(s_g),
            s=60,
            marker=marker_map[sub],
            color="0.0",
            zorder=3,
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(["Face", "Scene"], fontsize=14)
    ax2.set_ylabel("mean RT (s)", fontsize=14)
    ax2.tick_params(axis="y", labelsize=14)
    ax2.grid(False)
    ax2.set_ylim(0, 2)

    ax1.text(-0.2, 1.05, "A", transform=ax1.transAxes, fontsize=18, va="top", ha="left")
    ax2.text(-0.3, 1.05, "B", transform=ax2.transAxes, fontsize=18, va="top", ha="left")

    fig.savefig(out_png, bbox_inches="tight")
    plt.show()


# =========================
# Main
# =========================
def main():
    resultMat, countMat, df, rng = build_all(SUB_IDS, seed=SEED)

    tval, pval, face_acc, scene_acc, nSub = paired_acc_test(resultMat)
    print(f"t({nSub-1}) = {tval:.3f}, p = {pval:.4f}")
    print(f"Mean(Face)={face_acc.mean():.3f}, Mean(Scene)={scene_acc.mean():.3f}")

    t_rt, p_rt, face_means_geom, scene_means_geom, sub_list = rt_group_paired_geommean(df)
    print(f"Paired t-test on mean logRTs: t({len(sub_list)-1}) = {t_rt:.3f}, p = {p_rt:.4f}")
    print(f"Geometric mean RT (Face) = {np.mean(face_means_geom):.3f}")
    print(f"Geometric mean RT (Scene) = {np.mean(scene_means_geom):.3f}")

    correct_per_sub = np.mean(resultMat[:, :, 0], axis=1)
    noresp_per_sub = np.mean(resultMat[:, :, 2], axis=1)
    print(f"Mean correct response rate across participants: {np.mean(correct_per_sub):.3f}")
    print(f"Mean no-response rate across participants: {np.mean(noresp_per_sub):.3f}")

    plot_behavior(
        resultMat=resultMat,
        df=df,
        face_means_geom=face_means_geom,
        scene_means_geom=scene_means_geom,
        sub_list=sub_list,
        rng=rng,
        out_png="figure_behavioral_performance.png",
    )


if __name__ == "__main__":
    main()
