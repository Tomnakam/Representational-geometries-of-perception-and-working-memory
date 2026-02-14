"""
Run GLMsingle on surface fMRI (fsaverage/fsnative) for a given subject/session.
- Builds binary design matrices from event CSVs (optionally upsampled)
- Loads surface time series (L/R), concatenates hemispheres
- Optional temporal upsampling via pchip interpolation
- Runs GLMsingle and saves condition-specific betas as .npz
"""

# =========================
# User settings / inputs
# =========================
subID = input("subID: ")
sesID = input("sesID: ")
sesID_int = int(sesID)

test = float(input("test? no=0 or yes=1: "))
upsampling = float(input("upsampling? no=0 or yes=1: "))

brain_space = "fsaverage"   # or "fsnative"

TR = 1.4
newTR = 1.0
n_volumes = 319
stimdur = 1

# =========================
# Imports
# =========================
import time
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd

from nilearn import surface
from nilearn.datasets import fetch_surf_fsaverage

from glmsingle.glmsingle import GLM_single
from glmsingle.utils.alt_round import alt_round

from scipy.interpolate import pchip

# fsaverage mesh (not strictly required unless you use it later, but kept as in your code)
fsaverage = fetch_surf_fsaverage(mesh="fsaverage")

# =========================
# Paths
# =========================
cwd = Path.cwd()
fs_dir = cwd / brain_space
space_suffix = f"space-{brain_space}"


def ensure_dir(p: Path):
    """Create directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


# =========================
# Time-series interpolation utilities (as provided) 
# adapted from https://github.com/Charestlab/pyslicetime/blob/master/slicetime/tseriesinterp.py 
# see also Kay, K., Jamison, K. W., Zhang, R. Y., & Uğurbil, K. (2020). 
# A temporal decomposition method for identifying venous effects in task-based fMRI. Nature methods, 17(10), 1033-1039.
# =========================
def tseriesinterp(m, trorig, trnew, dim=None, numsamples=None,
                  fakeout=0, wantreplicate=False, interpmethod='pchip', slicetimes=None):
    """
    Interpolate <m> along the time dimension using pchip extrapolation.
    Keeps original alignment at the first timepoint; supports optional padding.
    """

    # internal constants
    numchunks = 20

    # infer time dimension
    if dim is None:
        dim = len(m.shape) - 1

    # prep 2D
    msize = np.asarray(m.shape)
    if len(msize) > 1:
        mflat = reshape2D(m, dim)

    # determine number of output samples
    if numsamples is None:
        numsamples0 = int(np.ceil((mflat.shape[0] * trorig) / trnew))
    else:
        numsamples0 = numsamples

    # original time points
    if wantreplicate:
        pre_pad = np.array((-3, -2, -1)) * trorig
        tps = np.arange(0, trorig * mflat.shape[0], trorig)
        post_pad = (mflat.shape[0] - 1) * trorig + np.array((1, 2, 3)) * trorig
        timeorig = np.r_[pre_pad, tps, post_pad]
    else:
        timeorig = [0.0 + x * (trorig * mflat.shape[0]) / len(mflat) for x in range(len(mflat))]

    # new time points
    timenew = np.array([0.0 + x * (trnew * numsamples0) / numsamples0 for x in range(int(numsamples0))]) - fakeout

    # interpolate in chunks (memory friendly)
    chunks = chunking(list(range(mflat.shape[1])), int(np.ceil(mflat.shape[1] / numchunks)))

    temp = []
    for chunk in chunks:
        if wantreplicate:
            this_ts = mflat[:, chunk]
            pre_pad = np.tile(mflat[0, chunk], (3, 1))
            post_pad = np.tile(mflat[-1, chunk], (3, 1))
            dat = np.r_[pre_pad, this_ts, post_pad]
            temp.append(pchip(timeorig, dat, extrapolate=True)(timenew))
        else:
            temp.append(pchip(timeorig, mflat[:, chunk], extrapolate=True)(timenew))

    stacktemp = np.hstack(temp)

    # restore original shape
    msize[dim] = numsamples0
    newm = reshape2D_undo(stacktemp, dim, msize)
    return newm


def reshape2D(m, dim):
    """Move dimension <dim> to axis 0 and reshape to 2D."""
    return np.moveaxis(m, dim, 0).reshape((m.shape[dim], -1), order='F')


def reshape2D_undo(f, dim, msize):
    """Undo reshape2D operation."""
    msizetmp = list(msize)
    msizetmp.remove(msize[dim])
    msizetmp.insert(0, msize[dim])
    return np.moveaxis(f.reshape(msizetmp, order='F'), 0, dim)


def chunking(vect, num, chunknum=None):
    """Split a vector into chunks of length <num>."""
    if chunknum is None:
        nchunk = int(np.ceil(len(vect) / num))
        return np.array_split(vect, nchunk)
    else:
        nchunk = int(np.ceil(len(vect) / num))
        f = np.array_split(vect, nchunk)
        xbegin = (chunknum - 1) * num + 1
        xend = np.min((len(vect), chunknum * num))
        return f[num - 1], xbegin, xend


# =========================
# Run counts and timing grid
# =========================
if int(subID) == 9:
    run_counts = {1: 8, 2: 8, 3: 7, 4: 9}
else:
    run_counts = {1: 8, 2: 8, 3: 8, 4: 8}

# Number of runs in this session
nRuns = run_counts[sesID_int]

# Time axis for design matrices
if upsampling == 0:
    time_points = np.arange(0, TR * n_volumes, TR)
else:
    time_points = np.arange(0, TR * n_volumes, newTR)

frame_times = np.arange(n_volumes) * TR

# Compute offset to map (ses, run) -> global run index used in event filenames
offset = sum(run_counts[s] for s in range(1, sesID_int))

# =========================
# Read event files
# =========================
event_files = [
    cwd / "event" / f"event_sub{subID}_run{offset + i}.csv"
    for i in range(1, nRuns + 1)
]
events = [pd.read_csv(f) for f in event_files]

# =========================
# Build binary design matrices
# =========================
trial_types = ["PFace", "PScene", "WMFace", "WMScene", "TestResp"]
design_matrices = []

for i in range(nRuns):
    event_df = pd.DataFrame(events[i])
    dm = pd.DataFrame(0, index=time_points, columns=trial_types)

    # Fill regressors (boxcars)
    for _, row in event_df.iterrows():
        onset, duration, trial_type = row["onset"], row["duration"], row["trial_type"]
        start_idx = np.searchsorted(time_points, onset)
        end_idx = np.searchsorted(time_points, onset + duration)
        dm.loc[time_points[start_idx:end_idx], trial_type] = 1

    design_matrices.append(np.array(dm.fillna(0)))

print(design_matrices[0])

# =========================
# Index vectors (which beta corresponds to which condition)
# =========================
index_vectors = [dm.argmax(axis=1) * (dm.sum(axis=1) > 0) - (dm.sum(axis=1) == 0) for dm in design_matrices]
index_vectors = np.concatenate([indices[indices != -1] for indices in index_vectors])

# =========================
# Load surface functional data (per run)
# =========================
file_namesR = [
    fs_dir / f"sub-0{subID}_ses-0{sesID}_task-prm_run-0{i}_hemi-R_{space_suffix}_bold.func.gii"
    for i in range(1, nRuns + 1)
]
file_namesL = [
    fs_dir / f"sub-0{subID}_ses-0{sesID}_task-prm_run-0{i}_hemi-L_{space_suffix}_bold.func.gii"
    for i in range(1, nRuns + 1)
]

func_right_list = [surface.load_surf_data(f).astype(np.float32) for f in file_namesR]
func_left_list  = [surface.load_surf_data(f).astype(np.float32) for f in file_namesL]

# Concatenate hemispheres (kept as in your code)
func_combined_list = [
    np.vstack([
        surface.load_surf_data(file_R).astype(np.float32),
        surface.load_surf_data(file_L).astype(np.float32),
    ])
    for file_R, file_L in zip(file_namesR, file_namesL)
]

# Optional temporal upsampling
if upsampling == 1:
    numsamples = np.array(int(np.ceil(n_volumes * TR / newTR)))
    print(func_combined_list[0].shape)
    for i in range(nRuns):
        func_combined_list[i] = tseriesinterp(func_combined_list[i], TR, newTR, numsamples=numsamples)
    print(func_combined_list[0].shape)

# =========================
# GLMsingle options / output locations
# =========================
opt = dict()

if test == 0:
    opt["wantlibrary"] = 1
    opt["wantglmdenoise"] = 1
    opt["wantfracridge"] = 1
    opt["wantfileoutputs"] = [0, 1, 1, 1]
    opt["wantmemoryoutputs"] = [0, 1, 1, 1]
    outputdir = str(cwd / "GLMsingleOutput" / f"sub{subID}_ses{sesID}")
    figuredir = str(cwd / "GLMsingleOutput" / f"sub{subID}_ses{sesID}_figure")
else:
    opt["wantlibrary"] = 0
    opt["wantglmdenoise"] = 0
    opt["wantfracridge"] = 0
    opt["wantfileoutputs"] = [0, 1, 0, 0]
    opt["wantmemoryoutputs"] = [0, 1, 0, 0]
    outputdir = str(cwd / "GLMsingleOutput" / f"test_sub{subID}_ses{sesID}")
    figuredir = str(cwd / "GLMsingleOutput" / f"test_sub{subID}_ses{sesID}_figure")

opt["maxpolydeg"] = [[4]] * nRuns #print([alt_round(((design_matrices[r].shape[0] * TR) / 60) / 2) + 1 for r in range(nRuns)])
opt["n_jobs"] = 4

# Make sure output dirs exist
ensure_dir(Path(outputdir))
ensure_dir(Path(figuredir))


# =========================
# Run GLMsingle
# =========================
glmsingle_obj = GLM_single(opt)
pprint(glmsingle_obj.params)

start_time = time.time()
results_glmsingle = glmsingle_obj.fit(
    design_matrices,
    func_combined_list,
    stimdur,
    TR,
    outputdir=outputdir,
    figuredir=figuredir
)

elapsed_time = time.time() - start_time
print("\telapsed time: ", f"{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")

# =========================
# Extract betas and save (condition-specific)
# =========================
if test == 0:
    plot_data = np.squeeze(results_glmsingle["typed"]["betasmd"]).astype(float)
    r2 = np.squeeze(results_glmsingle["typed"]["R2"].astype(float))
    FRAC = np.squeeze(results_glmsingle["typed"]["FRACvalue"].astype(float))
    HRF = np.squeeze(results_glmsingle["typed"]["HRFindex"])
else:
    plot_data = np.squeeze(results_glmsingle["typeb"]["betasmd"])
    r2 = np.squeeze(results_glmsingle["typeb"]["R2"])

print(plot_data.shape)  # voxel * event
print(np.max(r2))
print(np.min(r2))
print(np.max(FRAC))
print(np.min(FRAC))

# Only focusing on PFace, PScene, WMFace, WMScene (first 4 regressors)
if test==0:
    beta_condition_specific = [plot_data[:, index_vectors == i] for i in range(4)]

    beta_dir = cwd / "single_beta"
    ensure_dir(beta_dir)

    suffix = "" if brain_space == "fsaverage" else f"_{brain_space}"

    # ---- save R2 ----
    r2_path = beta_dir / f"R2{suffix}_sub{subID}_ses{sesID}.npy"
    np.save(r2_path, r2)
    
    # ---- save FRAC ----
    frac_path = beta_dir / f"FRAC{suffix}_sub{subID}_ses{sesID}.npy"
    np.save(frac_path, FRAC)
    
    # ---- save betas ----
    beta_path = beta_dir / f"beta_condition_specific{suffix}_sub{subID}_ses{sesID}.npz"
    np.savez(beta_path, *beta_condition_specific)
