"""
Run-wise surface GLM with optional motion censoring (interpolation).
- Builds design matrices (SPM HRF + polynomial drift + motion regressors)
- Loads fsaverage/fsnative surface time series (L/R), concatenates hemispheres
- Optionally interpolates censored volumes based on FD threshold
- Fits nilearn.run_glm per run
- Saves voxelwise R^2 and beta estimates

"""

# =========================
# User settings
# =========================
subID = input("subID: ")
motion_censoring = 1  # set 0 to skip interpolation-based censoring
brain_space = "fsaverage"  # or "fsnative"

TR = 1.4
n_volumes = 319
nRuns = 8
nSes = 4
motion_threshold = 0.5  # FD threshold

# =========================
# Imports
# =========================
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

from sklearn.metrics import r2_score
from nilearn.plotting import plot_design_matrix
from nilearn import surface
from nilearn.glm.first_level import run_glm, make_first_level_design_matrix


# =========================
# Helper functions
# =========================
def interpolate_censored_data(func_data, valid_indices, censored_indices):
    """Interpolate surface functional data at censored timepoints (per vertex)."""
    interpolated_data = func_data.copy()
    for vtx in range(func_data.shape[1]):
        ts = func_data[:, vtx]
        f = interpolate.interp1d(
            valid_indices,
            ts[valid_indices],
            bounds_error=False,
            fill_value="extrapolate",
        )
        interpolated_data[censored_indices, vtx] = f(censored_indices)
    return interpolated_data


def ensure_dir(path: Path):
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


# =========================
# Main
# =========================
start_time = time.time()

subID_int = int(subID)
frame_times = np.arange(n_volumes) * TR

cwd = Path.cwd()
fs_dir = cwd / brain_space
space_suffix = f"space-{brain_space}"

# Output dirs
design_dir = cwd / "design_matrix"
fd_dir = cwd / "FD"
output_dir = cwd / "runwise_beta"
ensure_dir(design_dir)
ensure_dir(fd_dir)
ensure_dir(output_dir)

# Determine run counts per session (special-case: sub-09)
if subID_int == 9:
    run_counts = {1: 8, 2: 8, 3: 7, 4: 9}
else:
    run_counts = {ses: nRuns for ses in range(1, nSes + 1)}

# Enumerate all runs in order
all_runs = [(ses, run) for ses, max_run in run_counts.items() for run in range(1, max_run + 1)]
n_total_runs = len(all_runs)

# =========================
# Read event files (assumes event_sub{subID}_run{k}.csv is sequentially numbered)
# =========================
event_files = [cwd / "event" / f"event_sub{subID}_run{i}.csv" for i in range(1, nSes * nRuns + 1)]
events = [pd.read_csv(f) for f in event_files]

# =========================
# Build design matrices
# =========================
design_matrices = []
motion_columns = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]

for i, (ses, run) in enumerate(all_runs):
    motion_df = pd.read_csv(
        cwd / "tsv" / f"sub-0{subID}_ses-0{ses}_task-prm_run-0{run}_desc-confounds_timeseries.tsv",
        sep="\t",
    )
    motion = motion_df[motion_columns].values

    dm = make_first_level_design_matrix(
        frame_times,
        events[i],
        hrf_model="spm",
        drift_model="polynomial",
        drift_order=3,
        add_regs=motion,
        add_reg_names=motion_columns,
    )
    design_matrices.append(dm)

# Print first design matrix (raw)
dm0 = design_matrices[0]
print(dm0)

# =========================
# Rename columns + plot (plot-only reorder; does not affect GLM fitting)
# =========================
dm_plot = dm0.copy()
dm_plot.columns = [
    "Face Perception",
    "Scene Perception",
    "Test",
    "Face Working Memory",
    "Scene Working Memory",
    "Translation(x)",
    "Translation(y)",
    "Translation(z)",
    "Rotation(x)",
    "Rotation(y)",
    "Rotation(z)",
    "Drift(1st-order)",
    "Drift(2nd-order)",
    "Drift(3rd-order)",
    "constant",
]

# Swap columns for visualization only
cols = list(dm_plot.columns)
cols[2], cols[4] = cols[4], cols[2]
cols[3], cols[2] = cols[2], cols[3]
dm_plot = dm_plot[cols]

fig_dm, ax_dm = plt.subplots(figsize=(10, 6), constrained_layout=True)
plot_design_matrix(dm_plot, axes=ax_dm)
ax_dm.set_xlabel(ax_dm.get_xlabel(), fontsize=14)
ax_dm.set_ylabel("Volume", fontsize=14)
ax_dm.tick_params(axis="both", labelsize=12)
fig_dm.savefig(design_dir / f"design_matrix_sub{subID}.png", dpi=300)
plt.close(fig_dm)

# =========================
# Prepare functional file lists (surface)
# =========================
file_namesR = [
    fs_dir / f"sub-0{subID}_ses-0{ses}_task-prm_run-{run:02d}_hemi-R_{space_suffix}_bold.func.gii"
    for ses, run in all_runs
]
file_namesL = [
    fs_dir / f"sub-0{subID}_ses-0{ses}_task-prm_run-{run:02d}_hemi-L_{space_suffix}_bold.func.gii"
    for ses, run in all_runs
]

labels = [None] * n_total_runs
estimates = [None] * n_total_runs

# Will be initialized after first run is loaded (to get dimensions)
beta_all = None
residuals_all = None
r2_all_voxels = None

# =========================
# Run-wise GLM
# =========================
for i, (ses, run) in enumerate(all_runs):
    motion_file = cwd / "tsv" / f"sub-0{subID}_ses-0{ses}_task-prm_run-{run:02d}_desc-confounds_timeseries.tsv"
    if not motion_file.exists():
        raise FileNotFoundError(f"Missing motion file: {motion_file}")

    motion_data = pd.read_csv(motion_file, sep="\t")
    fd = motion_data["framewise_displacement"].fillna(0)
    censor_mask = fd > motion_threshold

    # Save FD plot
    plt.figure()
    plt.plot(fd)
    plt.title(f"FD_run{i + 1}")
    plt.savefig(fd_dir / f"FD_sub{subID}_run{i + 1}.png")
    plt.close()

    # Load surface functional data
    func_right = surface.load_surf_data(file_namesR[i])
    func_left = surface.load_surf_data(file_namesL[i])

    censored_indices = np.where(censor_mask)[0]
    valid_indices = np.where(~censor_mask)[0]

    # Motion censoring (interpolation) toggle
    if motion_censoring == 1:
        func_right = interpolate_censored_data(func_right, valid_indices, censored_indices)
        func_left = interpolate_censored_data(func_left, valid_indices, censored_indices)

    # Force float64 + concatenate hemispheres
    func_right = func_right.astype(np.float64)
    func_left = func_left.astype(np.float64)
    func_concat = np.concatenate([func_left, func_right])

    # Initialize output arrays on first run
    if i == 0:
        n_regressors = design_matrices[0].values.shape[1]
        n_timepoints = func_concat.T.shape[0]
        n_voxels = func_concat.T.shape[1]

        r2_all_voxels = np.zeros((n_total_runs, n_voxels))
        beta_all = np.full((n_regressors, n_voxels, n_total_runs), np.nan)
        residuals_all = np.full((n_timepoints, n_voxels, n_total_runs), np.nan)

    # Fit GLM
    print(f"fitting run #{i + 1} (ses {ses}, run {run})...")
    X = design_matrices[i].values
    Y = func_concat.T
    labels[i], estimates[i] = run_glm(Y, X, n_jobs=6)

    # Fill arrays per label
    for lab, res in estimates[i].items():
        mask = (labels[i] == lab)  # Voxels belonging to this label
        beta_all[:, mask, i] = res.theta
        residuals_all[:, mask, i] = res.residuals

    # Compute voxelwise R^2
    for lbl in np.unique(labels[i]):
        res = estimates[i][lbl]
        idx = np.where(labels[i] == lbl)[0]  # Voxel indices in the original Y

        # Predict BOLD time course from the model
        yhat = X @ res.theta

        # Compute voxelwise R^2
        for col_in_label, col_in_Y in enumerate(idx):
            r2_all_voxels[i, col_in_Y] = r2_score(Y[:, col_in_Y], yhat[:, col_in_label])

# =========================
# Save outputs
# =========================
np.save(output_dir / f"r2_all_voxels_sub{subID}.npy", r2_all_voxels)

suffix = "" if brain_space == "fsaverage" else f"_{brain_space}"
with open(output_dir / f"beta{suffix}_{subID}.pkl", "wb") as f:
    pickle.dump(beta_all, f)

elapsed_time = time.time() - start_time
print(f"elapsed time: {elapsed_time:.3f} seconds")
