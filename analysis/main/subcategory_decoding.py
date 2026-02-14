# =========================
# Parameters
# =========================
nSes = 4
nRuns = 8
nFolds = 8
n_reps = 8  # repeat of condition within a run
n_components = 0.95
n_permutations = 1000
n_processors = 8

subID = int(input("SubID=? "))
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

import numpy as np
np.random.seed(subID)
import pandas as pd
import nibabel as nib
from nilearn.datasets import fetch_surf_fsaverage
fsaverage = fetch_surf_fsaverage(mesh="fsaverage")

atlas = float(input("Atlas 1:small 2:big"))

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# =========================
# Paths
# =========================
base_dir = Path("single_beta")
save_dir = Path("decoding_summary_subcategory")
save_dir.mkdir(parents=True, exist_ok=True)


# =========================
# Load beta (parallel over sessions)
# =========================
def load_and_process_beta(ses):
    filename = base_dir / f'beta_condition_specific_sub{subID}_ses{ses}.npz'
    try:
        beta_condition_specific = np.load(filename)

        beta_FP  = beta_condition_specific['arr_0'].reshape(beta_condition_specific['arr_0'].shape[0], -1, 10).mean(axis=2)
        beta_SP  = beta_condition_specific['arr_1'].reshape(beta_condition_specific['arr_1'].shape[0], -1, 10).mean(axis=2)
        beta_FWM = beta_condition_specific['arr_2'].reshape(beta_condition_specific['arr_2'].shape[0], -1, 10).mean(axis=2)
        beta_SWM = beta_condition_specific['arr_3'].reshape(beta_condition_specific['arr_3'].shape[0], -1, 10).mean(axis=2)

        print(f"Loaded {filename}")
        return beta_FP, beta_SP, beta_FWM, beta_SWM
    except FileNotFoundError:
        print(f"File {filename} not found. Skipping.")
        return None

all_beta_FP, all_beta_SP, all_beta_FWM, all_beta_SWM = [], [], [], []

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(load_and_process_beta, range(1, nSes + 1)))

for res in results:
    if res is not None:
        beta_FP, beta_SP, beta_FWM, beta_SWM = res
        all_beta_FP.append(beta_FP)
        all_beta_SP.append(beta_SP)
        all_beta_FWM.append(beta_FWM)
        all_beta_SWM.append(beta_SWM)

beta_FP_concat  = np.concatenate(all_beta_FP, axis=1)
beta_SP_concat  = np.concatenate(all_beta_SP, axis=1)
beta_FWM_concat = np.concatenate(all_beta_FWM, axis=1)
beta_SWM_concat = np.concatenate(all_beta_SWM, axis=1)


# =========================
# Load behavior & exclude runs (rows)
# =========================
df = pd.read_csv(f'behav_rawdata/rawdata_PRMdecoding_exp1_{subID}.csv')

if exclude == 1:
    excluded_rows = []
    for run in excluded_runID:
        start = (run - 1) * 16
        end   = run * 16
        excluded_rows.extend(range(start, end))
    print("Excluded rows:", excluded_rows)
    df_clean = df.drop(excluded_rows).reset_index(drop=True)
else:
    df_clean = df

cueF_df = df_clean[df_clean['Cue'] == 1]
categoryF = list(zip(cueF_df['FaceCategory'], cueF_df['SceneCategory']))

cueS_df = df_clean[df_clean['Cue'] == 2]
categoryS = list(zip(cueS_df['FaceCategory'], cueS_df['SceneCategory']))


# =========================
# Exclude beta columns (runs) if needed
# =========================
excluded_cols = []
for run in excluded_runID:
    start = (run - 1) * n_reps
    end   = run * n_reps
    excluded_cols.extend(range(start, end))
print("Excluded columns:", excluded_cols)

def exclude_columns(mat, excluded_cols):
    keep = np.setdiff1d(np.arange(mat.shape[1]), excluded_cols)
    return mat[:, keep]

beta_FP_concat  = exclude_columns(beta_FP_concat,  excluded_cols)
beta_SP_concat  = exclude_columns(beta_SP_concat,  excluded_cols)
beta_FWM_concat = exclude_columns(beta_FWM_concat, excluded_cols)
beta_SWM_concat = exclude_columns(beta_SWM_concat, excluded_cols)


# =========================
# Stratify by (FaceCategory, SceneCategory) combination
# =========================
n_voxel, n_trials = beta_FP_concat.shape

unique_conditions = sorted(set(categoryF))
n_cond = len(unique_conditions)

trials_per_condF = [categoryF.count(cond) for cond in unique_conditions]
trials_per_condS = [categoryS.count(cond) for cond in unique_conditions]

beta_FP_stratified  = np.empty((n_voxel, n_trials))
beta_SP_stratified  = np.empty((n_voxel, n_trials))
beta_FWM_stratified = np.empty((n_voxel, n_trials))
beta_SWM_stratified = np.empty((n_voxel, n_trials))

run_indicesF = np.empty((n_trials), dtype=int)
run_indicesS = np.empty((n_trials), dtype=int)

end = 0
for cond_idx, cond in enumerate(unique_conditions):
    trial_indices = [i for i, cat in enumerate(categoryF) if cat == cond]
    start = end
    end = start + int(trials_per_condF[cond_idx])

    beta_FP_stratified[:, start:end]  = beta_FP_concat[:, trial_indices]
    beta_FWM_stratified[:, start:end] = beta_FWM_concat[:, trial_indices]
    run_indicesF[start:end] = np.array([i // n_reps for i in trial_indices], dtype=int)

end = 0
for cond_idx, cond in enumerate(unique_conditions):
    trial_indices = [i for i, cat in enumerate(categoryS) if cat == cond]
    start = end
    end = start + int(trials_per_condS[cond_idx])

    beta_SP_stratified[:, start:end]  = beta_SP_concat[:, trial_indices]
    beta_SWM_stratified[:, start:end] = beta_SWM_concat[:, trial_indices]
    run_indicesS[start:end] = np.array([i // n_reps for i in trial_indices], dtype=int)

# multi-class labels (4-class)
condition_list_F = np.repeat([i[0] for i in unique_conditions], trials_per_condF)
condition_list_S = np.repeat([i[1] for i in unique_conditions], trials_per_condS)

# binary labels
condition_list_F_race = np.where(condition_list_F < 3, 1, 2)
condition_list_F_sex  = np.where(condition_list_F % 2 == 1, 1, 2)
condition_list_S_naturalness = np.where(condition_list_S < 3, 1, 2)
condition_list_S_openness    = np.where(condition_list_S % 2 == 1, 1, 2)


# =========================
# Atlas / ROI mask
# =========================
left_labels_array, ctab, names = nib.freesurfer.read_annot('atlas_label/lh.HCP-MMP1.annot')
right_labels_array, ctab, names = nib.freesurfer.read_annot('atlas_label/rh.HCP-MMP1.annot')

if atlas == 1:
    names = [name.decode('utf-8').removeprefix('R_').removesuffix('_ROI') for name in names]
    names = names[1:]
    mask = range(1, 181)
elif atlas == 2:
    import json
    file_path = 'Glasser_mask_new.json'
    with open(file_path, 'r') as json_file:
        mask = json.load(json_file)
    labels = list(mask.keys())


# =========================
# Decoding tools
# =========================
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

cv = GroupKFold(n_splits=nFolds)

def _permute_within_run(condition_list, run_list, n_permutations):
    """(n_permutations, N) label permutations within each run."""
    all_shuffled = np.empty((n_permutations, len(condition_list)), dtype=condition_list.dtype)
    runs = np.unique(run_list)
    for perm in range(n_permutations):
        shuf = condition_list.copy()
        for r in runs:
            idx = np.where(run_list == r)[0]
            shuf[idx] = np.random.permutation(condition_list[idx])
        all_shuffled[perm] = shuf
    return all_shuffled

def _standardize_within_run(X, beta_array_ROI, run_list):
    """standardize within each run across conditions/trials (in-place on X)."""
    for r in np.unique(run_list):
        idx = np.where(run_list == r)[0]
        block = beta_array_ROI[idx]
        X[idx] = StandardScaler().fit_transform(block)
    return X


# =========================
# ROI worker: run both (1) face/scene 4-class and (2) detailed binary boundaries
# =========================
category_boundary = ['race', 'sex', 'naturalness', 'openness']

def decode_roi(args):
    mask_index, mask_name, mask_id = args
    print(f"{mask_name=}")

    roi_indiceL = np.isin(left_labels_array, mask_id)
    roi_indiceR = np.isin(right_labels_array, mask_id)
    roi_indice = np.concatenate([roi_indiceR, roi_indiceL])

    out = {}

    # ---------- (A) 4-class decoding: face / scene ----------
    for cb in ["face", "scene"]:
        if cb == "face":
            beta_array_ROI = np.concatenate([beta_FP_stratified.T, beta_FWM_stratified.T], axis=0)[:, roi_indice]
            condition_list = np.tile(condition_list_F, 2)
            run_list = np.concatenate([run_indicesF, run_indicesF])
        else:
            beta_array_ROI = np.concatenate([beta_SP_stratified.T, beta_SWM_stratified.T], axis=0)[:, roi_indice]
            condition_list = np.tile(condition_list_S, 2)
            run_list = np.concatenate([run_indicesS, run_indicesS])

        X = beta_array_ROI.astype(np.float32)
        X = _standardize_within_run(X, beta_array_ROI, run_list)
        all_shuf = _permute_within_run(condition_list, run_list, n_permutations)

        acc_list = []
        null = np.zeros((n_permutations, nFolds))

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, condition_list, groups=run_list)):
            X_train = X[train_idx]
            y_train = condition_list[train_idx]
            X_test  = X[test_idx]
            y_test  = condition_list[test_idx]

            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca  = pca.transform(X_test)

            clf = LogisticRegression(solver='lbfgs', max_iter=10000)
            clf.fit(X_train_pca, y_train)
            acc_list.append(accuracy_score(y_test, clf.predict(X_test_pca)))

            for perm in range(n_permutations):
                y_train_shuf = all_shuf[perm, train_idx]
                y_test_shuf  = all_shuf[perm, test_idx]

                clf_null = LogisticRegression(solver='lbfgs', max_iter=10000)
                clf_null.fit(X_train_pca, y_train_shuf)
                null[perm, fold_idx] = accuracy_score(y_test_shuf, clf_null.predict(X_test_pca))

        acc_mean = float(np.mean(acc_list))
        null_mean = null.mean(axis=1)
        p_value = float(np.mean(null_mean >= acc_mean))
        z_value = (acc_mean - float(null_mean.mean())) / float(null_mean.std(ddof=1))

        out[cb] = {"accuracy": acc_mean, "p_value": p_value, "z_value": float(z_value)}
        print(f"ROI: {mask_name}, category: {cb}, result: {out[cb]}")

    # ---------- (B) detailed binary boundaries ----------
    for cb in category_boundary:
        print('category boundary:', cb)

        if cb == 'race':
            beta_array_ROI = np.concatenate([beta_FP_stratified.T, beta_FWM_stratified.T], axis=0)[:, roi_indice]
            condition_list = np.tile(condition_list_F_race, 2)
            run_list = np.concatenate([run_indicesF, run_indicesF])
        elif cb == 'sex':
            beta_array_ROI = np.concatenate([beta_FP_stratified.T, beta_FWM_stratified.T], axis=0)[:, roi_indice]
            condition_list = np.tile(condition_list_F_sex, 2)
            run_list = np.concatenate([run_indicesF, run_indicesF])
        elif cb == 'naturalness':
            beta_array_ROI = np.concatenate([beta_SP_stratified.T, beta_SWM_stratified.T], axis=0)[:, roi_indice]
            condition_list = np.tile(condition_list_S_naturalness, 2)
            run_list = np.concatenate([run_indicesS, run_indicesS])
        elif cb == 'openness':
            beta_array_ROI = np.concatenate([beta_SP_stratified.T, beta_SWM_stratified.T], axis=0)[:, roi_indice]
            condition_list = np.tile(condition_list_S_openness, 2)
            run_list = np.concatenate([run_indicesS, run_indicesS])

        X = beta_array_ROI.astype(np.float32)
        X = _standardize_within_run(X, beta_array_ROI, run_list)
        all_shuf = _permute_within_run(condition_list, run_list, n_permutations)

        acc_list = []
        null = np.zeros((n_permutations, nFolds))

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, condition_list, groups=run_list)):
            X_train = X[train_idx]
            y_train = condition_list[train_idx]
            X_test  = X[test_idx]
            y_test  = condition_list[test_idx]

            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca  = pca.transform(X_test)

            clf = LogisticRegression(solver='lbfgs', max_iter=10000)
            clf.fit(X_train_pca, y_train)
            acc_list.append(accuracy_score(y_test, clf.predict(X_test_pca)))

            for perm in range(n_permutations):
                y_train_shuf = all_shuf[perm, train_idx]
                y_test_shuf  = all_shuf[perm, test_idx]

                clf_null = LogisticRegression(solver='lbfgs', max_iter=10000)
                clf_null.fit(X_train_pca, y_train_shuf)
                null[perm, fold_idx] = accuracy_score(y_test_shuf, clf_null.predict(X_test_pca))

        acc_mean = float(np.mean(acc_list))
        null_mean = null.mean(axis=1)
        p_value = float(np.mean(null_mean >= acc_mean))
        z_value = (acc_mean - float(null_mean.mean())) / float(null_mean.std(ddof=1))

        out[cb] = {"accuracy": acc_mean, "p_value": p_value, "z_value": float(z_value)}
        print(f"ROI: {mask_name}, category: {cb}, result: {out[cb]}")

    return mask_index, mask_name, out


# =========================
# Run (parallel over ROIs)
# =========================
if atlas == 1:
    args_list = [(i, name, id_) for i, (name, id_) in enumerate(zip(names, mask))]
elif atlas == 2:
    args_list = [(i, name, id_) for i, (name, id_) in enumerate(mask.items())]

with ProcessPoolExecutor(max_workers=n_processors) as executor:
    results = list(executor.map(decode_roi, args_list))

# CSV順を安定化
results = sorted(results, key=lambda x: x[0])


# =========================
# Save CSV (4-class face/scene)
# =========================
roi_names = []
accuracy_face, p_face, z_face = [], [], []
accuracy_scene, p_scene, z_scene = [], [], []

for mask_index, mask_name, res in results:
    roi_names.append(mask_name)
    accuracy_face.append(res['face']['accuracy'])
    p_face.append(res['face']['p_value'])
    z_face.append(res['face']['z_value'])
    accuracy_scene.append(res['scene']['accuracy'])
    p_scene.append(res['scene']['p_value'])
    z_scene.append(res['scene']['z_value'])

df_results = pd.DataFrame({
    'ROI': roi_names,
    'Face_Accuracy': accuracy_face,
    'Face_p_value': p_face,
    'Face_z_value': z_face,
    'Scene_Accuracy': accuracy_scene,
    'Scene_p_value': p_scene,
    'Scene_z_value': z_scene,
})

if atlas == 1:
    if exclude == 1 and subID in (2, 5):
        out_csv = save_dir /f'summary_category_decoding_smallROI_{subID}_exclude.csv'    
    else:
        out_csv = save_dir /f'summary_category_decoding_smallROI_{subID}.csv'   
elif atlas == 2:
    if exclude == 1 and subID in (2, 5):
        out_csv = save_dir /f'summary_category_decoding_bigROI_{subID}_exclude.csv' 
    else:
        out_csv = save_dir /f'summary_category_decoding_bigROI_{subID}.csv' 
df_results.to_csv(out_csv, index=False)

# =========================
# Save CSV (binary boundaries)
# =========================
accuracy_race, p_race, z_race = [], [], []
accuracy_sex, p_sex, z_sex = [], [], []
accuracy_naturalness, p_naturalness, z_naturalness = [], [], []
accuracy_openness, p_openness, z_openness = [], [], []

for mask_index, mask_name, res in results:
    accuracy_race.append(res['race']['accuracy'])
    p_race.append(res['race']['p_value'])
    z_race.append(res['race']['z_value'])

    accuracy_sex.append(res['sex']['accuracy'])
    p_sex.append(res['sex']['p_value'])
    z_sex.append(res['sex']['z_value'])

    accuracy_naturalness.append(res['naturalness']['accuracy'])
    p_naturalness.append(res['naturalness']['p_value'])
    z_naturalness.append(res['naturalness']['z_value'])

    accuracy_openness.append(res['openness']['accuracy'])
    p_openness.append(res['openness']['p_value'])
    z_openness.append(res['openness']['z_value'])

df_results = pd.DataFrame({
    'ROI': roi_names,
    'Race_Accuracy': accuracy_race,
    'Race_p_value': p_race,
    'Race_z_value': z_race,
    'Sex_Accuracy': accuracy_sex,
    'Sex_p_value': p_sex,
    'Sex_z_value': z_sex,
    'Naturalness_Accuracy': accuracy_naturalness,
    'Naturalness_p_value': p_naturalness,
    'Naturalness_z_value': z_naturalness,
    'Openness_Accuracy': accuracy_openness,
    'Openness_p_value': p_openness,
    'Openness_z_value': z_openness,
})

if atlas == 1:
    if exclude == 1 and subID in (2, 5):
        out_csv = save_dir /f'summary_detailed_category_decoding_smallROI_{subID}_exclude.csv'
    else:
        out_csv = save_dir /f'summary_detailed_category_decoding_smallROI_{subID}.csv'
elif atlas == 2:
    if exclude == 1 and subID in (2, 5):
        out_csv = save_dir /f'summary_detailed_category_decoding_bigROI_{subID}_exclude.csv'
    else:
        out_csv = save_dir /f'summary_detailed_category_decoding_bigROI_{subID}.csv'
df_results.to_csv(out_csv, index=False)