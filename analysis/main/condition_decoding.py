"""
ROI-wise decoding & CCGP with permutation nulls (surface beta).

- Load GLMsingle betas per session (npz), average over 10 samples per rep
- Optionally exclude specific poor runs
- ROI loop (parallel, demo=0):
    - ROI mask from HCP-MMP1 annot (or Glasser_mask_new.json)
    - Standardize within each run across 4 conditions
    - GroupKFold by run
    - Train/test with PCA(train only) + LogisticRegression
    - Permutation nulls (within-run shuffling)
    - Save summary CSV

DEMO special mode (demo=1):
- Fixed params: subID=2, exclude=0, atlas=2
- File pattern: beta_condition_specific_sub{subID}_ses{ses}_demo.npz
- No ROI loop, no 2nd ProcessPoolExecutor
- Use X = beta_array (no ROI masking)
- Permutations reduced to 10
- Print to terminal only:
    Feature_Decoding_Accuracy
    XOR_Decoding_Accuracy
    CCGP_Feature_Decoding_Accuracy
    Tradeoff index = log(XOR/CCGP_Feature)
- No saving

NOTE (spawn-safe on macOS):
- All ProcessPoolExecutor usage is inside main()
- Workers do not rely on globals; everything passed via args
"""

# =========================
# Imports (top-level; safe)
# =========================
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import nibabel as nib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold


# =========================
# Load and preprocess betas (per session)
# =========================
def load_and_process_beta(ses: int, subID: int, base_dir: Path, demo: int):
    """
    Load betas for one session and average within last dimension (=10).

    Expected file:
      demo=0: single_beta/beta_condition_specific_sub{subID}_ses{ses}.npz
      demo=1: single_beta/beta_condition_specific_sub{subID}_ses{ses}_demo.npz

    Returns:
        beta_FP, beta_SP, beta_FWM, beta_SWM (each: vox x (runs*reps))
    """
    if demo == 1:
        filename = base_dir / f"beta_condition_specific_sub{subID}_ses{ses}_demo.npz"
    else:
        filename = base_dir / f"beta_condition_specific_sub{subID}_ses{ses}.npz"

    if not filename.exists():
        print(f"File {filename} not found. Skipping.")
        return None

    beta_npz = np.load(filename)

    def _avg10(arr):
        return arr.reshape(arr.shape[0], -1, 10).mean(axis=2)

    beta_FP = _avg10(beta_npz["arr_0"])
    beta_SP = _avg10(beta_npz["arr_1"])
    beta_FWM = _avg10(beta_npz["arr_2"])
    beta_SWM = _avg10(beta_npz["arr_3"])

    print(f"Loaded {filename}")
    return beta_FP, beta_SP, beta_FWM, beta_SWM


# =========================
# Permutation helpers
# =========================
def permute_within_run_all4(condition_list, run_list, n_perm):
    """Within each run, permute labels among all 4 conditions."""
    out = np.empty((n_perm, len(condition_list)), dtype=condition_list.dtype)
    runs = np.unique(run_list)
    for p in range(n_perm):
        shuf = condition_list.copy()
        for r in runs:
            idx = np.where(run_list == r)[0]
            shuf[idx] = np.random.permutation(condition_list[idx])
        out[p] = shuf
    return out


def permute_within_run_split12_34(condition_list, run_list, n_perm):
    """Within each run, permute only within {1,2} and within {3,4} separately."""
    out = np.empty((n_perm, len(condition_list)), dtype=condition_list.dtype)
    runs = np.unique(run_list)
    for p in range(n_perm):
        shuf = condition_list.copy()
        for r in runs:
            idx_run = np.where(run_list == r)[0]
            idx_12 = idx_run[np.isin(condition_list[idx_run], [1, 2])]
            idx_34 = idx_run[np.isin(condition_list[idx_run], [3, 4])]
            shuf[idx_12] = np.random.permutation(condition_list[idx_12])
            shuf[idx_34] = np.random.permutation(condition_list[idx_34])
        out[p] = shuf
    return out


def permute_within_run_split13_24(condition_list, run_list, n_perm):
    """Within each run, permute only within {1,3} and within {2,4} separately."""
    out = np.empty((n_perm, len(condition_list)), dtype=condition_list.dtype)
    runs = np.unique(run_list)
    for p in range(n_perm):
        shuf = condition_list.copy()
        for r in runs:
            idx_run = np.where(run_list == r)[0]
            idx_13 = idx_run[np.isin(condition_list[idx_run], [1, 3])]
            idx_24 = idx_run[np.isin(condition_list[idx_run], [2, 4])]
            shuf[idx_13] = np.random.permutation(condition_list[idx_13])
            shuf[idx_24] = np.random.permutation(condition_list[idx_24])
        out[p] = shuf
    return out


# =========================
# Core decoding (single "unit": ROI or whole-brain)
# =========================
def run_decoding_unit(
    X: np.ndarray,
    condition_list: np.ndarray,
    run_list: np.ndarray,
    nRuns_total: int,
    n_reps: int,
    nFolds: int,
    decodingPCA_n_components,
    n_permutations: int
):
    """
    IMPORTANT:
    - This function standardizes X in-place.
      If you need to preserve original X, pass X.copy().
    """

    # ---- standardize within each run across 4 conditions ----
    for run_idx in range(nRuns_total):
        trial_indices = []
        for cond_idx in range(4):
            for rep_idx in range(n_reps):
                trial_idx = cond_idx * nRuns_total * n_reps + run_idx * n_reps + rep_idx
                trial_indices.append(trial_idx)
        block = X[trial_indices]
        X[trial_indices] = StandardScaler().fit_transform(block)

    cv = GroupKFold(n_splits=nFolds)

    mask_FP = condition_list == 1
    mask_SP = condition_list == 2
    mask_FWM = condition_list == 3
    mask_SWM = condition_list == 4

    accuraciesFeature = []
    nullFeature = np.zeros((n_permutations, nFolds), dtype=np.float32)

    accuraciesTask = []
    nullTask = np.zeros((n_permutations, nFolds), dtype=np.float32)

    accuraciesXOR = []
    nullXOR = np.zeros((n_permutations, nFolds), dtype=np.float32)

    accuraciesPtoWM = []
    nullPtoWM = np.zeros((n_permutations, nFolds), dtype=np.float32)

    accuraciesWMtoP = []
    nullWMtoP = np.zeros((n_permutations, nFolds), dtype=np.float32)

    accuraciesFtoS = []
    nullFtoS = np.zeros((n_permutations, nFolds), dtype=np.float32)

    accuraciesStoF = []
    nullStoF = np.zeros((n_permutations, nFolds), dtype=np.float32)

    accuraciesFeatureInP = []
    nullFeatureInP = np.zeros((n_permutations, nFolds), dtype=np.float32)

    accuraciesFeatureInWM = []
    nullFeatureInWM = np.zeros((n_permutations, nFolds), dtype=np.float32)

    accuraciesTaskInF = []
    nullTaskInF = np.zeros((n_permutations, nFolds), dtype=np.float32)

    accuraciesTaskInS = []
    nullTaskInS = np.zeros((n_permutations, nFolds), dtype=np.float32)

    # ==========
    # Block 1
    # ==========
    all_shuf_4 = permute_within_run_all4(condition_list, run_list, n_permutations)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, condition_list, groups=run_list)):
        X_train = X[train_idx]
        y_train = condition_list[train_idx]
        X_test = X[test_idx]
        y_test = condition_list[test_idx]

        pca = PCA(n_components=decodingPCA_n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Feature: F vs S
        y_train_bin = np.isin(y_train, [2, 4]).astype(int)
        y_test_bin = np.isin(y_test, [2, 4]).astype(int)
        clf = LogisticRegression(solver="liblinear")
        clf.fit(X_train_pca, y_train_bin)
        accuraciesFeature.append(accuracy_score(y_test_bin, clf.predict(X_test_pca)))

        # Task: P vs WM
        y_train_bin = np.isin(y_train, [3, 4]).astype(int)
        y_test_bin = np.isin(y_test, [3, 4]).astype(int)
        clf = LogisticRegression(solver="liblinear")
        clf.fit(X_train_pca, y_train_bin)
        accuraciesTask.append(accuracy_score(y_test_bin, clf.predict(X_test_pca)))

        # XOR: (FP+SWM) vs (SP+FWM)
        y_train_bin = np.isin(y_train, [2, 3]).astype(int)
        y_test_bin = np.isin(y_test, [2, 3]).astype(int)
        clf = LogisticRegression(solver="liblinear")
        clf.fit(X_train_pca, y_train_bin)
        accuraciesXOR.append(accuracy_score(y_test_bin, clf.predict(X_test_pca)))

        # nulls
        for perm in range(n_permutations):
            y_train_shuf = all_shuf_4[perm, train_idx]
            y_test_shuf = all_shuf_4[perm, test_idx]

            clf_null = LogisticRegression(solver="liblinear")
            clf_null.fit(X_train_pca, np.isin(y_train_shuf, [2, 4]).astype(int))
            nullFeature[perm, fold_idx] = accuracy_score(
                np.isin(y_test_shuf, [2, 4]).astype(int),
                clf_null.predict(X_test_pca),
            )

            clf_null = LogisticRegression(solver="liblinear")
            clf_null.fit(X_train_pca, np.isin(y_train_shuf, [3, 4]).astype(int))
            nullTask[perm, fold_idx] = accuracy_score(
                np.isin(y_test_shuf, [3, 4]).astype(int),
                clf_null.predict(X_test_pca),
            )

            clf_null = LogisticRegression(solver="liblinear")
            clf_null.fit(X_train_pca, np.isin(y_train_shuf, [2, 3]).astype(int))
            nullXOR[perm, fold_idx] = accuracy_score(
                np.isin(y_test_shuf, [2, 3]).astype(int),
                clf_null.predict(X_test_pca),
            )

    # ==========
    # Block 2
    # ==========
    all_shuf_12_34 = permute_within_run_split12_34(condition_list, run_list, n_permutations)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, condition_list, groups=run_list)):
        # Feature in P (FP vs SP)
        train_mask = (mask_FP | mask_SP) & np.isin(np.arange(len(condition_list)), train_idx)
        train_index = np.where(train_mask)[0]
        X_train = X[train_index]
        y_train = condition_list[train_index]

        test_mask = (mask_FP | mask_SP) & np.isin(np.arange(len(condition_list)), test_idx)
        test_index = np.where(test_mask)[0]
        X_test = X[test_index]
        y_test = condition_list[test_index]

        pca = PCA(n_components=decodingPCA_n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        clf = LogisticRegression(solver="liblinear")
        clf.fit(X_train_pca, (y_train == 2).astype(int))
        accuraciesFeatureInP.append(accuracy_score((y_test == 2).astype(int), clf.predict(X_test_pca)))

        for perm in range(n_permutations):
            y_train_shuf = all_shuf_12_34[perm, train_index]
            y_test_shuf = all_shuf_12_34[perm, test_index]
            clf_null = LogisticRegression(solver="liblinear")
            clf_null.fit(X_train_pca, (y_train_shuf == 2).astype(int))
            nullFeatureInP[perm, fold_idx] = accuracy_score((y_test_shuf == 2).astype(int), clf_null.predict(X_test_pca))

        # P->WM (test on FWM vs SWM)
        test_mask = (mask_FWM | mask_SWM) & np.isin(np.arange(len(condition_list)), test_idx)
        test_index_wm = np.where(test_mask)[0]
        X_test_wm = X[test_index_wm]
        y_test_wm = condition_list[test_index_wm]
        X_test_wm_pca = pca.transform(X_test_wm)

        accuraciesPtoWM.append(accuracy_score((y_test_wm == 4).astype(int), clf.predict(X_test_wm_pca)))

        for perm in range(n_permutations):
            y_train_shuf = all_shuf_12_34[perm, train_index]
            y_test_shuf_wm = all_shuf_12_34[perm, test_index_wm]
            clf_null = LogisticRegression(solver="liblinear")
            clf_null.fit(X_train_pca, (y_train_shuf == 2).astype(int))
            nullPtoWM[perm, fold_idx] = accuracy_score((y_test_shuf_wm == 4).astype(int), clf_null.predict(X_test_wm_pca))

        # Feature in WM (FWM vs SWM)
        train_mask = (mask_FWM | mask_SWM) & np.isin(np.arange(len(condition_list)), train_idx)
        train_index = np.where(train_mask)[0]
        X_train = X[train_index]
        y_train = condition_list[train_index]

        test_mask = (mask_FWM | mask_SWM) & np.isin(np.arange(len(condition_list)), test_idx)
        test_index = np.where(test_mask)[0]
        X_test = X[test_index]
        y_test = condition_list[test_index]

        pca = PCA(n_components=decodingPCA_n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        clf = LogisticRegression(solver="liblinear")
        clf.fit(X_train_pca, (y_train == 4).astype(int))
        accuraciesFeatureInWM.append(accuracy_score((y_test == 4).astype(int), clf.predict(X_test_pca)))

        for perm in range(n_permutations):
            y_train_shuf = all_shuf_12_34[perm, train_index]
            y_test_shuf = all_shuf_12_34[perm, test_index]
            clf_null = LogisticRegression(solver="liblinear")
            clf_null.fit(X_train_pca, (y_train_shuf == 4).astype(int))
            nullFeatureInWM[perm, fold_idx] = accuracy_score((y_test_shuf == 4).astype(int), clf_null.predict(X_test_pca))

        # WM->P (test on FP vs SP)
        test_mask = (mask_FP | mask_SP) & np.isin(np.arange(len(condition_list)), test_idx)
        test_index_p = np.where(test_mask)[0]
        X_test_p = X[test_index_p]
        y_test_p = condition_list[test_index_p]
        X_test_p_pca = pca.transform(X_test_p)

        accuraciesWMtoP.append(accuracy_score((y_test_p == 2).astype(int), clf.predict(X_test_p_pca)))

        for perm in range(n_permutations):
            y_train_shuf = all_shuf_12_34[perm, train_index]
            y_test_shuf_p = all_shuf_12_34[perm, test_index_p]
            clf_null = LogisticRegression(solver="liblinear")
            clf_null.fit(X_train_pca, (y_train_shuf == 4).astype(int))
            nullWMtoP[perm, fold_idx] = accuracy_score((y_test_shuf_p == 2).astype(int), clf_null.predict(X_test_p_pca))

    # ==========
    # Block 3
    # ==========
    all_shuf_13_24 = permute_within_run_split13_24(condition_list, run_list, n_permutations)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, condition_list, groups=run_list)):
        # Task in F (FP vs FWM)
        train_mask = (mask_FP | mask_FWM) & np.isin(np.arange(len(condition_list)), train_idx)
        train_index = np.where(train_mask)[0]
        X_train = X[train_index]
        y_train = condition_list[train_index]

        test_mask = (mask_FP | mask_FWM) & np.isin(np.arange(len(condition_list)), test_idx)
        test_index = np.where(test_mask)[0]
        X_test = X[test_index]
        y_test = condition_list[test_index]

        pca = PCA(n_components=decodingPCA_n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        clf = LogisticRegression(solver="liblinear")
        clf.fit(X_train_pca, (y_train == 3).astype(int))
        accuraciesTaskInF.append(accuracy_score((y_test == 3).astype(int), clf.predict(X_test_pca)))

        for perm in range(n_permutations):
            y_train_shuf = all_shuf_13_24[perm, train_index]
            y_test_shuf = all_shuf_13_24[perm, test_index]
            clf_null = LogisticRegression(solver="liblinear")
            clf_null.fit(X_train_pca, (y_train_shuf == 3).astype(int))
            nullTaskInF[perm, fold_idx] = accuracy_score((y_test_shuf == 3).astype(int), clf_null.predict(X_test_pca))

        # F->S (test on SP vs SWM)
        test_mask = (mask_SP | mask_SWM) & np.isin(np.arange(len(condition_list)), test_idx)
        test_index_s = np.where(test_mask)[0]
        X_test_s = X[test_index_s]
        y_test_s = condition_list[test_index_s]
        X_test_s_pca = pca.transform(X_test_s)

        accuraciesFtoS.append(accuracy_score((y_test_s == 4).astype(int), clf.predict(X_test_s_pca)))

        for perm in range(n_permutations):
            y_train_shuf = all_shuf_13_24[perm, train_index]
            y_test_shuf_s = all_shuf_13_24[perm, test_index_s]
            clf_null = LogisticRegression(solver="liblinear")
            clf_null.fit(X_train_pca, (y_train_shuf == 3).astype(int))
            nullFtoS[perm, fold_idx] = accuracy_score((y_test_shuf_s == 4).astype(int), clf_null.predict(X_test_s_pca))

        # Task in S (SP vs SWM)
        train_mask = (mask_SP | mask_SWM) & np.isin(np.arange(len(condition_list)), train_idx)
        train_index = np.where(train_mask)[0]
        X_train = X[train_index]
        y_train = condition_list[train_index]

        test_mask = (mask_SP | mask_SWM) & np.isin(np.arange(len(condition_list)), test_idx)
        test_index = np.where(test_mask)[0]
        X_test = X[test_index]
        y_test = condition_list[test_index]

        pca = PCA(n_components=decodingPCA_n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        clf = LogisticRegression(solver="liblinear")
        clf.fit(X_train_pca, (y_train == 4).astype(int))
        accuraciesTaskInS.append(accuracy_score((y_test == 4).astype(int), clf.predict(X_test_pca)))

        for perm in range(n_permutations):
            y_train_shuf = all_shuf_13_24[perm, train_index]
            y_test_shuf = all_shuf_13_24[perm, test_index]
            clf_null = LogisticRegression(solver="liblinear")
            clf_null.fit(X_train_pca, (y_train_shuf == 4).astype(int))
            nullTaskInS[perm, fold_idx] = accuracy_score((y_test_shuf == 4).astype(int), clf_null.predict(X_test_pca))

        # S->F (test on FP vs FWM)
        test_mask = (mask_FP | mask_FWM) & np.isin(np.arange(len(condition_list)), test_idx)
        test_index_f = np.where(test_mask)[0]
        X_test_f = X[test_index_f]
        y_test_f = condition_list[test_index_f]
        X_test_f_pca = pca.transform(X_test_f)

        accuraciesStoF.append(accuracy_score((y_test_f == 3).astype(int), clf.predict(X_test_f_pca)))

        for perm in range(n_permutations):
            y_train_shuf = all_shuf_13_24[perm, train_index]
            y_test_shuf_f = all_shuf_13_24[perm, test_index_f]
            clf_null = LogisticRegression(solver="liblinear")
            clf_null.fit(X_train_pca, (y_train_shuf == 4).astype(int))
            nullStoF[perm, fold_idx] = accuracy_score((y_test_shuf_f == 3).astype(int), clf_null.predict(X_test_f_pca))

    # =========================
    # Summaries
    # =========================
    def summary_stats(acc_list, null_mat):
        mean_acc = float(np.mean(acc_list))
        mean_null_per_perm = np.mean(null_mat, axis=1)
        p = float(np.mean(mean_null_per_perm >= mean_acc))
        mu = float(np.mean(mean_null_per_perm))
        sd = float(np.std(mean_null_per_perm))
        z = (mean_acc - mu) / sd if sd > 0 else np.nan
        return mean_acc, p, z, mean_null_per_perm

    mean_acc_feature, feature_p, feature_z, mean_nullFeature = summary_stats(accuraciesFeature, nullFeature)
    mean_acc_featureP, featureP_p, featureP_z, mean_nullFeatureInP = summary_stats(accuraciesFeatureInP, nullFeatureInP)
    mean_acc_featureWM, featureWM_p, featureWM_z, mean_nullFeatureInWM = summary_stats(accuraciesFeatureInWM, nullFeatureInWM)
    mean_acc_task, task_p, task_z, mean_nullTask = summary_stats(accuraciesTask, nullTask)
    mean_acc_taskF, taskF_p, taskF_z, mean_nullTaskInF = summary_stats(accuraciesTaskInF, nullTaskInF)
    mean_acc_taskS, taskS_p, taskS_z, mean_nullTaskInS = summary_stats(accuraciesTaskInS, nullTaskInS)
    mean_acc_XOR, XOR_p, XOR_z, mean_nullXOR = summary_stats(accuraciesXOR, nullXOR)
    mean_acc_PtoWM, PtoWM_p, PtoWM_z, mean_nullPtoWM = summary_stats(accuraciesPtoWM, nullPtoWM)
    mean_acc_WMtoP, WMtoP_p, WMtoP_z, mean_nullWMtoP = summary_stats(accuraciesWMtoP, nullWMtoP)
    mean_acc_FtoS, FtoS_p, FtoS_z, mean_nullFtoS = summary_stats(accuraciesFtoS, nullFtoS)
    mean_acc_StoF, StoF_p, StoF_z, mean_nullStoF = summary_stats(accuraciesStoF, nullStoF)

    CCGP_Feature = (mean_acc_PtoWM + mean_acc_WMtoP) / 2
    CCGP_Feature_null = (mean_nullPtoWM + mean_nullWMtoP) / 2
    CCGP_Feature_p = float(np.mean(CCGP_Feature_null >= CCGP_Feature))
    mu = float(np.mean(CCGP_Feature_null))
    sd = float(np.std(CCGP_Feature_null))
    CCGP_Feature_z = (CCGP_Feature - mu) / sd if sd > 0 else np.nan

    CCGP_Task = (mean_acc_FtoS + mean_acc_StoF) / 2
    CCGP_Task_null = (mean_nullFtoS + mean_nullStoF) / 2
    CCGP_Task_p = float(np.mean(CCGP_Task_null >= CCGP_Task))
    mu = float(np.mean(CCGP_Task_null))
    sd = float(np.std(CCGP_Task_null))
    CCGP_Task_z = (CCGP_Task - mu) / sd if sd > 0 else np.nan

    return (
        mean_acc_feature, feature_p, feature_z,
        mean_acc_featureP, featureP_p, featureP_z,
        mean_acc_featureWM, featureWM_p, featureWM_z,
        mean_acc_task, task_p, task_z,
        mean_acc_taskF, taskF_p, taskF_z,
        mean_acc_taskS, taskS_p, taskS_z,
        mean_acc_XOR, XOR_p, XOR_z,
        CCGP_Feature, CCGP_Feature_p, CCGP_Feature_z,
        CCGP_Task, CCGP_Task_p, CCGP_Task_z
    )


# =========================
# ROI worker (spawn-safe: NO globals)
# =========================
def process_roi(args):
    (
        mask_index, mask_name, mask_id,
        left_labels_array, right_labels_array,
        beta_array, condition_list, run_list,
        nRuns_total, n_reps, nFolds,
        decodingPCA_n_components, n_permutations
    ) = args

    # ROI mask (your original order: [R, L])
    roi_indiceL = np.isin(left_labels_array, mask_id)
    roi_indiceR = np.isin(right_labels_array, mask_id)
    roi_indice = np.concatenate([roi_indiceR, roi_indiceL])

    X = beta_array[:, roi_indice].astype(np.float32)

    metrics = run_decoding_unit(
        X=X,
        condition_list=condition_list,
        run_list=run_list,
        nRuns_total=nRuns_total,
        n_reps=n_reps,
        nFolds=nFolds,
        decodingPCA_n_components=decodingPCA_n_components,
        n_permutations=n_permutations
    )

    return (mask_name,) + metrics


# =========================
# MAIN (spawn-safe)
# =========================
def main():
    # ---- keep default behavior for demo=0 (same prompts, same defaults) ----
    nSes = 4
    nRuns = 8
    nFolds = 8
    n_reps = 8
    n_proc = 4

    demo = int(float(input("demo? 0:No 1:Yes ")))

    # ---- DEMO special: force fixed params ----
    if demo == 1:
        subID = 2
        exclude = 0.0
        excluded_runID = []
        atlas = 2
    else:
        subID = int(input("SubID=? "))
        exclude = float(input("exclude runs? 0:No 1:Yes "))
        if exclude == 1:
            if subID == 2:
                excluded_runID = [3]
            elif subID == 5:
                excluded_runID = [12]
            else:
                excluded_runID = []
        else:
            excluded_runID = []

        atlas = int(float(input("Atlas 1:small 2:big ")))

    decodingPCA_n_components = 0.95

    np.random.seed(subID)

    # permutations: demo only -> 10, else -> 1000
    n_permutations = 10 if demo == 1 else 2

    base_dir = Path("single_beta")
    save_dir = Path("decoding_summary")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- load betas per session (parallel is fine) ----
    with ProcessPoolExecutor(max_workers=n_proc) as executor:
        fn = partial(load_and_process_beta, subID=subID, base_dir=base_dir, demo=demo)
        results = list(executor.map(fn, range(1, nSes + 1)))

    all_beta_FP, all_beta_SP, all_beta_FWM, all_beta_SWM = [], [], [], []
    for res in results:
        if res is None:
            continue
        bFP, bSP, bFWM, bSWM = res
        all_beta_FP.append(bFP)
        all_beta_SP.append(bSP)
        all_beta_FWM.append(bFWM)
        all_beta_SWM.append(bSWM)

    if len(all_beta_FP) == 0:
        raise RuntimeError("No beta files were loaded (all sessions missing?).")

    beta_FP_concat = np.concatenate(all_beta_FP, axis=1)
    beta_SP_concat = np.concatenate(all_beta_SP, axis=1)
    beta_FWM_concat = np.concatenate(all_beta_FWM, axis=1)
    beta_SWM_concat = np.concatenate(all_beta_SWM, axis=1)

    # ---- exclude run columns ----
    excluded_cols = []
    for run in excluded_runID:
        start = (run - 1) * n_reps
        end = run * n_reps
        excluded_cols.extend(range(start, end))

    def exclude_columns(mat, cols):
        if len(cols) == 0:
            return mat
        keep = np.setdiff1d(np.arange(mat.shape[1]), np.array(cols, dtype=int))
        return mat[:, keep]

    beta_FP_concat = exclude_columns(beta_FP_concat, excluded_cols)
    beta_SP_concat = exclude_columns(beta_SP_concat, excluded_cols)
    beta_FWM_concat = exclude_columns(beta_FWM_concat, excluded_cols)
    beta_SWM_concat = exclude_columns(beta_SWM_concat, excluded_cols)

    nBeta_cond = beta_FP_concat.shape[1]
    nRuns_total = nSes * nRuns - len(excluded_runID)

    # ---- build decoding arrays ----
    beta_array = np.concatenate(
        [beta_FP_concat.T, beta_SP_concat.T, beta_FWM_concat.T, beta_SWM_concat.T],
        axis=0
    ).astype(np.float32)

    condition_list = np.concatenate([
        np.ones(nBeta_cond, dtype=int),
        np.full(nBeta_cond, 2, dtype=int),
        np.full(nBeta_cond, 3, dtype=int),
        np.full(nBeta_cond, 4, dtype=int),
    ])

    one_cond_run_list = np.repeat(np.arange(nRuns_total), n_reps)
    run_list = np.tile(one_cond_run_list, 4)

    n_groups = len(np.unique(run_list))
    if nFolds > n_groups:
        raise ValueError(f"nFolds={nFolds} is larger than #runs={n_groups} after exclusion.")

    # =========================
    # DEMO: NO ROI LOOP, NO SAVE
    # =========================
    if demo == 1:
        X = beta_array.astype(np.float32)  # as requested (no ROI masking)

        metrics = run_decoding_unit(
            X=X,
            condition_list=condition_list,
            run_list=run_list,
            nRuns_total=nRuns_total,
            n_reps=n_reps,
            nFolds=nFolds,
            decodingPCA_n_components=decodingPCA_n_components,
            n_permutations=n_permutations
        )

        # indices in return tuple
        Feature_Acc = metrics[0]
        XOR_Acc = metrics[18]
        CCGP_Feature_Acc = metrics[21]

        eps = 1e-12
        tradeoff = float(np.log((XOR_Acc + eps) / (CCGP_Feature_Acc + eps)))

        print("\n=== DEMO RESULTS ===")
        print("subject:", subID) 
        print(f"Feature_Decoding_Accuracy          = {Feature_Acc:.6f}")
        print(f"XOR_Decoding_Accuracy              = {XOR_Acc:.6f}")
        print(f"CCGP_Feature_Decoding_Accuracy     = {CCGP_Feature_Acc:.6f}")
        print(f"Tradeoff index log(XOR/CCGP_Feat)  = {tradeoff:.6f}")
        print("====================\n")
        return

    # =========================
    # NON-DEMO: ROI LOOP (parallel), behavior unchanged
    # =========================
    left_labels_array, _, _ = nib.freesurfer.read_annot("atlas_label/lh.HCP-MMP1.annot")
    right_labels_array, _, names = nib.freesurfer.read_annot("atlas_label/rh.HCP-MMP1.annot")
    left_labels_array = np.where(left_labels_array == 0, np.nan, left_labels_array)
    right_labels_array = np.where(right_labels_array == 0, np.nan, right_labels_array)

    if atlas == 1:
        names = [n.decode("utf-8").removeprefix("R_").removesuffix("_ROI") for n in names][1:]
        mask = list(range(1, 181))
        roi_items = list(zip(names, mask))
        out_tag = "smallROI"
    elif atlas == 2:
        with open("Glasser_mask_new.json", "r") as f:
            mask = json.load(f)
        roi_items = list(mask.items())
        out_tag = "bigROI"
    else:
        raise ValueError("atlas must be 1 (small) or 2 (big)")

    args_list = [
        (
            i, name, roi_id,
            left_labels_array, right_labels_array,
            beta_array, condition_list, run_list,
            nRuns_total, n_reps, nFolds,
            decodingPCA_n_components, n_permutations
        )
        for i, (name, roi_id) in enumerate(roi_items)
    ]

    with ProcessPoolExecutor(max_workers=n_proc) as executor:
        results = list(executor.map(process_roi, args_list))

    df_results = pd.DataFrame(results, columns=[
        "ROI",
        "Feature_Decoding_Accuracy", "Feature_Decoding_pval", "Feature_Decoding_zval",
        "Feature_Decoding_Accuracy(P)", "Feature_Decoding_pval(P)", "Feature_Decoding_zval(P)",
        "Feature_Decoding_Accuracy(WM)", "Feature_Decoding_pval(WM)", "Feature_Decoding_zval(WM)",
        "Task_Decoding_Accuracy", "Task_Decoding_pval", "Task_Decoding_zval",
        "Task_Decoding_Accuracy(F)", "Task_Decoding_pval(F)", "Task_Decoding_zval(F)",
        "Task_Decoding_Accuracy(S)", "Task_Decoding_pval(S)", "Task_Decoding_zval(S)",
        "XOR_Decoding_Accuracy", "XOR_Decoding_pval", "XOR_Decoding_zval",
        "CCGP_Feature_Decoding_Accuracy", "CCGP_Feature_Decoding_pval", "CCGP_Feature_Decoding_zval",
        "CCGP_Task_Decoding_Accuracy", "CCGP_Task_Decoding_pval", "CCGP_Task_Decoding_zval"
    ])

    out_csv = save_dir / f"summary_{out_tag}_{subID}.csv"
    df_results.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
