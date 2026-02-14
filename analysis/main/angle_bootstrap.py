"""
Bootstrap-based angle vs noise-floor analysis (ROI-wise).
- Loads run-wise betas (4 conditions) from runwise_beta/beta_{subID}.pkl
- Excludes specific runs (optional)
- Chooses atlas:
    1) SMALL: HCP-MMP1 parcels (1..180)
    2) BIG: custom coarse Glasser groups from Glasser_mask_new.json
- For each ROI:
    - z-score within each run across 4 conditions
    - define VP = FP-SP and VWM = FWM-SWM across runs
    - bootstrap run means to estimate distributions and noise-floor (approx)
    - compute empirical angle distribution and z-score of (empirical - noise_floor)
- Saves: z-scores, empirical angles, noise floor
"""

# =========================
# Settings / inputs
# =========================
n_boot = 10000

demo = int(input("demo? 0:No 1:Yes ").strip())
if demo == 1:
    subID = 2
    exclude = 0
    atlas = 2
else:
    subID = int(input("subID="))
    exclude = float(input("exclude runs? 0:No 1:Yes"))
    atlas = int(float(input("Atlas 1:small 2:big")))

if exclude == 1:
    if subID == 2:
        excluded_runID = [3]
    elif subID == 5:
        excluded_runID = [12]
    else:
        excluded_runID = []
else:
    excluded_runID = []

nRuns = 32
keep_runs = [r for r in range(1, nRuns + 1) if r not in set(excluded_runID)]
n_keep = len(keep_runs)


# =========================
# Imports
# =========================
import json
import pickle
import random
from pathlib import Path

import numpy as np
import nibabel as nib

# RNG
rng = np.random.default_rng(subID)
random.seed(subID)

# =========================
# Paths
# =========================
cwd = Path.cwd()
beta_dir = cwd / "runwise_beta"

if demo == 1:
    beta_path = beta_dir / f"beta_{subID}_demo.pkl"
else:
    beta_path = beta_dir / f"beta_{subID}.pkl"


out_dir = cwd / "angle_summary"
out_dir.mkdir(parents=True, exist_ok=True)

# =========================
# Load beta
# =========================
with open(beta_path, "rb") as f:
    beta = pickle.load(f)

# =========================
# Load atlas labels / ROI definitions
# =========================
left_labels_array, ctab, names = nib.freesurfer.read_annot("atlas_label/lh.HCP-MMP1.annot")
right_labels_array, ctab, names = nib.freesurfer.read_annot("atlas_label/rh.HCP-MMP1.annot")

if atlas == 1:
    # SMALL atlas: 180 parcels
    names = [n.decode("utf-8").removeprefix("R_").removesuffix("_ROI") for n in names]
    names = names[1:]  # drop background label
    roi_iter = list(zip(names, range(1, 181)))  # (roi_name, roi_id)
    atlas_tag = "atlasSMALL"

elif atlas == 2:
    # BIG atlas: custom coarse groups from json
    mask_json = cwd / "Glasser_mask_new.json"
    with open(mask_json, "r") as f:
        mask = json.load(f)
    roi_iter = list(mask.items())  # (roi_name, roi_id)
    atlas_tag = "atlasBIG"

else:
    raise ValueError("atlas must be 1 (small) or 2 (big)")

# =========================
# Extract betamaps (keep_runs only)
# =========================
n_vox = beta.shape[1]

betamapFP  = np.empty((n_keep, n_vox), dtype=float)
betamapSP  = np.empty((n_keep, n_vox), dtype=float)
betamapFWM = np.empty((n_keep, n_vox), dtype=float)
betamapSWM = np.empty((n_keep, n_vox), dtype=float)

print("keep_runs:", keep_runs)
for i, r in enumerate(keep_runs):
    betamapFP[i, :]  = beta[0, :, r - 1]
    betamapSP[i, :]  = beta[1, :, r - 1]
    betamapFWM[i, :] = beta[3, :, r - 1]
    betamapSWM[i, :] = beta[4, :, r - 1]

# =========================
# Allocate outputs
# =========================
n_rois = len(roi_iter)

empirical_cos = np.empty((n_rois, n_boot))
ang_diff = np.empty((n_rois, n_boot))
noisefloor = np.empty((n_rois,))
z_scores_cos = np.empty((n_rois,))

# =========================
# ROI loop
# =========================
for roi_index, (roi_name, roi_id) in enumerate(roi_iter):
    print(roi_name)

    if demo==1:
        masked_FP  = betamapFP
        masked_SP  = betamapSP
        masked_FWM = betamapFWM
        masked_SWM = betamapSWM

    else:
        roi_indiceL = np.isin(left_labels_array, roi_id)
        roi_indiceR = np.isin(right_labels_array, roi_id)
        roi_indice = np.concatenate([roi_indiceL, roi_indiceR])

        masked_FP  = betamapFP[:, roi_indice]
        masked_SP  = betamapSP[:, roi_indice]
        masked_FWM = betamapFWM[:, roi_indice]
        masked_SWM = betamapSWM[:, roi_indice]

    nVoxels = masked_FP.shape[1]
    print(f"Number of Voxels: {nVoxels}")

    # --- z-scoring within each run across 4 conditions ---
    for r in range(n_keep):
        group = np.vstack([masked_FP[r, :], masked_SP[r, :], masked_FWM[r, :], masked_SWM[r, :]])
        mu = np.mean(group)
        sd = np.std(group)
        tmp = (group - mu) / sd

        masked_FP[r, :]  = tmp[0, :]
        masked_SP[r, :]  = tmp[1, :]
        masked_FWM[r, :] = tmp[2, :]
        masked_SWM[r, :] = tmp[3, :]

    # Dimension count (as in your code: number of voxels after masking)
    n_components_actual = masked_FP.shape[1]

    # VP and VWM across runs (n_keep x nVoxels)
    VP = masked_FP - masked_SP
    VWM = masked_FWM - masked_SWM

    n = masked_FP.shape[0]
    Bootstrap_Indices = rng.integers(0, n, size=(n_boot, n))  # (B, n)

    # --- bootstrap mean distribution for VP ---
    means_boot = []
    for j in range(nVoxels):
        x = VP[:, j]
        means_boot.append(x[Bootstrap_Indices].mean(axis=1))  # (B,)
    vp_dist = np.column_stack(means_boot)  # (B, k)
    ep = vp_dist - vp_dist.mean(axis=0)

    # --- bootstrap mean distribution for VWM ---
    means_boot = []
    for j in range(nVoxels):
        x = VWM[:, j]
        means_boot.append(x[Bootstrap_Indices].mean(axis=1))  # (B,)
    vm_dist = np.column_stack(means_boot)  # (B, k)
    em = vm_dist - vm_dist.mean(axis=0)

    # trace-like quantities (as in your code)
    trSigma_p = np.einsum("bk,bk->", ep, ep) / (n_boot - 1)
    trSigma_m = np.einsum("bk,bk->", em, em) / (n_boot - 1)
    noise_cov = np.einsum("bk,bk->", ep, em) / (n_boot - 1)

    # approximation (clipped at 0)
    Sp = max(np.linalg.norm(VP.mean(axis=0)) ** 2 - trSigma_p, 0)
    Swm = max(np.linalg.norm(VWM.mean(axis=0)) ** 2 - trSigma_m, 0)

    cos_nf = (np.sqrt(Sp * Swm) + noise_cov) / np.sqrt((Sp + trSigma_p) * (Swm + trSigma_m))
    noisefloor[roi_index] = np.degrees(np.arccos(cos_nf))

    # empirical angles across bootstraps
    for b in range(n_boot):
        VP_mean = vp_dist[b, :]
        VWM_mean = vm_dist[b, :]

        normP = np.linalg.norm(VP_mean)
        normWM = np.linalg.norm(VWM_mean)
        empirical_cos[roi_index, b] = np.degrees(np.arccos(np.dot(VP_mean, VWM_mean) / (normP * normWM)))

    # z-score of (empirical - noise floor)
    ang_diff[roi_index, :] = empirical_cos[roi_index, :] - noisefloor[roi_index]
    z_scores_cos[roi_index] = np.mean(ang_diff[roi_index, :]) / np.std(ang_diff[roi_index, :])

    if demo == 1:           
        break                 


# =========================
# Save outputs
# =========================
if demo == 1:
    print("\n=== DEMO RESULTS ===")
    print(" subject:", subID) 
    print(" z-value(primary visual):", z_scores_cos[0])
    print("====================\n")

else:
    if exclude == 1 and subID in (2, 5):
        z_path = out_dir / f"z_scores_{subID}_bootstrap_exclude_{atlas_tag}.pkl"
        with open(z_path, "wb") as f:
            pickle.dump(z_scores_cos, f)

        emp_path = out_dir / f"empirical_{subID}_bootstrap_exclude_{atlas_tag}.pkl"
        with open(emp_path, "wb") as f:
            pickle.dump(empirical_cos, f)

        nf_path = out_dir / f"nf_{subID}_bootstrap_exclude_{atlas_tag}.pkl"
        with open(nf_path, "wb") as f:
            pickle.dump(noisefloor, f)
    

    else:
        z_path = out_dir / f"z_scores_{subID}_bootstrap_{atlas_tag}.pkl"
        with open(z_path, "wb") as f:
            pickle.dump(z_scores_cos, f)

        emp_path = out_dir / f"empirical_{subID}_bootstrap_{atlas_tag}.pkl"
        with open(emp_path, "wb") as f:
            pickle.dump(empirical_cos, f)

        nf_path = out_dir / f"nf_{subID}_bootstrap_{atlas_tag}.pkl"
        with open(nf_path, "wb") as f:
            pickle.dump(noisefloor, f)
    

    print("Saved:")
    print(" ", z_path)
    print(" ", emp_path)
    print(" ", nf_path)