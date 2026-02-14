"""
ROI-wise split-half reliability (z-scoring enabled).
- Load run-wise betas (4 conditions) from runwise_beta/beta_{subID}.pkl
- Generate many odd/even run assignments (stratified by 4 blocks of 8 runs)
- ROI mask: Glasser coarse masks (Glasser_mask_new.json) + HCP-MMP1 annot
- For each ROI and each assignment:
    - z-score betas within run across 4 conditions
    - compute condition means for odd/even halves
    - compute reliability correlations
- Save outputs
"""

# =========================
# Settings / inputs
# =========================
rep = 10000
n_components = 4

subID = int(input("subID="))
exclude = float(input("exclude runs? 0:No 1:Yes"))

if exclude == 1:
    if subID == 2:
        excluded_runID = [3]
    elif subID == 5:
        excluded_runID = [12]
    else:
        excluded_runID = []
else:
    excluded_runID = []

# =========================
# Imports
# =========================
import json
import pickle
import random
from pathlib import Path

import numpy as np
import nibabel as nib

random.seed(subID)

# =========================
# Output dirs
# =========================
cwd = Path.cwd()
out_dir = cwd / "test_retest_data"

for d in [out_dir]:
    d.mkdir(parents=True, exist_ok=True)

# =========================
# Load beta
# =========================
base_dir = Path("runwise_beta")
beta_path = base_dir / f"beta_{subID}.pkl"

with open(beta_path, "rb") as f:
    beta = pickle.load(f)

# =========================
# Run list and random assignments (odd/even)
# =========================
nRuns = 32
all_runs = list(range(1, nRuns + 1))

group1 = [r for r in range(1, 9)   if r not in excluded_runID]
group2 = [r for r in range(9, 17)  if r not in excluded_runID]
group3 = [r for r in range(17, 25) if r not in excluded_runID]
group4 = [r for r in range(25, 33) if r not in excluded_runID]

valid_assignments = []
for _ in range(rep):
    odd = sorted(
        random.sample(group1, 4)
        + random.sample(group2, 4)
        + random.sample(group3, 4)
        + random.sample(group4, 4)
    )
    even = sorted(list(set(all_runs) - set(odd)))
    valid_assignments.append({"run_odd": tuple(odd), "run_even": tuple(even)})

# =========================
# ROI definition (Glasser coarse mask + HCP-MMP1 annot)
# =========================
mask_json = "Glasser_mask_new.json"
with open(mask_json, "r") as f:
    mask = json.load(f)

left_labels_array, ctab, names = nib.freesurfer.read_annot("atlas_label/lh.HCP-MMP1.annot")
right_labels_array, ctab, names = nib.freesurfer.read_annot("atlas_label/rh.HCP-MMP1.annot")

# decode names
names = [n.decode("utf-8").replace("_R", "").replace("/", "_") for n in names]
names = names[0::2]

# treat label 0 as background
left_labels_array = np.where(left_labels_array == 0, np.nan, left_labels_array)
right_labels_array = np.where(right_labels_array == 0, np.nan, right_labels_array)

n_rois = len(mask)

# =========================
# Extract run-wise betamaps (run x vertex)
# =========================
n_vertices = len(right_labels_array) + len(left_labels_array)

betamapFP  = np.empty((nRuns, n_vertices))
betamapSP  = np.empty((nRuns, n_vertices))
betamapFWM = np.empty((nRuns, n_vertices))
betamapSWM = np.empty((nRuns, n_vertices))

for run in range(nRuns):
    betamapFP[run, :]  = beta[0, :, run]
    betamapSP[run, :]  = beta[1, :, run]
    betamapFWM[run, :] = beta[3, :, run]
    betamapSWM[run, :] = beta[4, :, run]

# =========================
# Allocate outputs
# =========================
n_perm = len(valid_assignments)

R_FSP_FSP = np.empty((n_rois, n_perm))
R_FSWM_FSWM = np.empty((n_rois, n_perm))

# =========================
# ROI loop
# =========================
for roi_idx, (roi_name, roi_id) in enumerate(mask.items()):
    print(roi_name)

    roi_indiceL = np.isin(left_labels_array, roi_id)
    roi_indiceR = np.isin(right_labels_array, roi_id)
    roi_indice = np.concatenate([roi_indiceL, roi_indiceR])

    masked_betamapFP  = betamapFP[:, roi_indice]
    masked_betamapSP  = betamapSP[:, roi_indice]
    masked_betamapFWM = betamapFWM[:, roi_indice]
    masked_betamapSWM = betamapSWM[:, roi_indice]

    # --- z-scoring within each run across 4 conditions ---
    for r in range(nRuns):
        group = np.vstack([
            masked_betamapFP[r, :],
            masked_betamapSP[r, :],
            masked_betamapFWM[r, :],
            masked_betamapSWM[r, :],
        ])
        mu = np.mean(group)
        sd = np.std(group)
        tmp = (group - mu) / sd

        masked_betamapFP[r, :]  = tmp[0, :]
        masked_betamapSP[r, :]  = tmp[1, :]
        masked_betamapFWM[r, :] = tmp[2, :]
        masked_betamapSWM[r, :] = tmp[3, :]

    # --- permutations ---
    for p, assignment in enumerate(valid_assignments):
        odd_idx = np.array(assignment["run_odd"]) - 1
        even_idx = np.array(assignment["run_even"]) - 1

        odd_FP  = masked_betamapFP[odd_idx, :]
        odd_SP  = masked_betamapSP[odd_idx, :]
        odd_FWM = masked_betamapFWM[odd_idx, :]
        odd_SWM = masked_betamapSWM[odd_idx, :]

        even_FP  = masked_betamapFP[even_idx, :]
        even_SP  = masked_betamapSP[even_idx, :]
        even_FWM = masked_betamapFWM[even_idx, :]
        even_SWM = masked_betamapSWM[even_idx, :]

        # Condition-wise mean vectors
        odd_mean = [x.mean(axis=0) for x in (odd_FP, odd_SP, odd_FWM, odd_SWM)]
        even_mean = [x.mean(axis=0) for x in (even_FP, even_SP, even_FWM, even_SWM)]
        FPodd_nopc, SPodd_nopc, FWModd_nopc, SWModd_nopc = odd_mean
        FPeven_nopc, SPeven_nopc, FWMeven_nopc, SWMeven_nopc = even_mean

        # Reliability-style correlations
        R_FSP_FSP[roi_idx, p] = np.corrcoef(FPodd_nopc - SPodd_nopc, FPeven_nopc - SPeven_nopc)[0, 1]
        R_FSWM_FSWM[roi_idx, p] = np.corrcoef(FWModd_nopc - SWModd_nopc, FWMeven_nopc - SWMeven_nopc)[0, 1]

# =========================
# Save outputs
# =========================

def dump_pkl(obj, name):
    if exclude == 1 and subID in (2, 5):
        p = out_dir / f"{name}_{subID}_exclude.pkl"
        with open(p, "wb") as f:
            pickle.dump(obj, f)
    else:
        p = out_dir / f"{name}_{subID}.pkl"
        with open(p, "wb") as f:
            pickle.dump(obj, f)

dump_pkl(R_FSP_FSP, "CorrP")
dump_pkl(R_FSWM_FSWM, "CorrWM")