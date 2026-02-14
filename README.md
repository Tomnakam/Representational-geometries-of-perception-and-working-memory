# Representational-geometries-of-perception-and-working-memory

## Preprint

This is the code associated with the preprint:
* [Nakamura, T., Yoo, S.B.M., Kay, K., Lau, H., Moharramipour, A. Representational geometries of perception and working memory. bioRxiv (2026)](https://doi.org/10.1101/2025.09.07.674590)

If you use the code in your research, please cite this preprint.

## Installation Guide
```bash
git clone https://github.com/Tomnakam/Representational-geometries-of-perception-and-working-memory
```
It will take a couple of minutes to install on normal desktop computer.

## System Requirements
This project requires **Conda**.
To create the environment:

```bash
conda env create -f environment.yml
conda activate analysis
```

To run singl-trial GLM, [GLMsingle toolbox](https://github.com/cvnlab/GLMsingle) introduced in the following paper is alos required:
* [Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr, M., Kay, K.N. Improving the accuracy of single-trial fMRI response estimates using GLMsingle. *eLife* (2022).](https://doi.org/10.7554/eLife.77599)

To run tradeoff_figure.py (Supplementary figures), Connectome Workbench (https://www.humanconnectome.org/software/get-connectome-workbench) is required.

## Demo
To quickly demonstrate how the analysis code is run,
1) Run `analysis/main/angle_boostrap.py` with `demo = 1`  
   → Produces the z-value in the angle analysis (square marker in Figure 2A; ~1 minute)
2) Run `analysis/main/condition_decoding.py` with `demo = 1`  
   → Produces the face-vs-scene decoding accuracy, CCGP accuracy, XOR decoding accuracy (square marker in Figure 3) and tradeoff index (square marker in Figure 4; taking ~10 minutes) 

These demo analyses are performed for sub-02 in V1.
To reproduce the entire results, follow the instructions below (all betamaps shared on zenodo is required). 

## Rawdata
All raw MR data are available in [OpenNeuro](https://openneuro.org/datasets/ds007082).
To reproduce preprocessing and GLM anlayses, this raw data is required.

## Preprocessing
To reproduce the preporcessing steps by fMRIPrep, run analysis/preprocessing/run_fmriprep_all.sh
You have to put all subjects' fsaverage.gii together under the directory named "fsaverage" to make the following GLM steps work. 

## GLM
To obatin betamaps used in angle analysis, run analysis/GLM/surfaceGLM.py. Subject id (1-9) will be asked.
To obtain betamaps used in decoding, run analysis/GLM/surfaceGLMsingle.py. Subject id (1-9) and session id (1-4) will be asked.

## Betamaps
All beta maps generated during the GLM steps are available on [Zenodo] (the link will be made publicly accessible upon publication).
After downloading, place the files in the following directories:

```
analysis/main/runwise_beta
analysis/main/single_beta
```
These directories are required for the subsequent analyses.

### 1. run-wise beta maps:
      File name format: beta_{subID}.pkl
      Format: Pickle file
      Shape: (n_regressors, n_vertices, n_runs)
These are used for angle analysis.

### 2. trial-wise beta maps:
      File name format: beta_condition_specific_{subID}_{sesID}.npz
      Format: Compressed NumPy archive (.npz).
      Each file contains four NumPy arrays, corresponding to:
         Face Perception, Scene Perception, Face Working Memory, Scene Working Memory.
      Each array has shape: (n_vertices, n_TRs)
These are used for decoding analysis.

## Main Analyses and Figures
The main analysis scripts use the beta maps as input.
(These files are not required when running the demo codes.)

When executing the scripts, you will be prompted to enter several integer values:

    subID: 1-9
    excluded: 1  → exclude predefined poor runs  
              0  → use all runs
    atlas: 1  → coarse-grained atlas (180 parcels)  
           2  → fine-grained atlas (22 ROIs)
    tradeoff index: 1  → simple log ratio  (Figure 4)  
                    2  → probit ratio      (Figure S12, S13)
    decoding test statistics: 1  → logit-transformed accuracy  
                              2  → z-value

### 1. angle
To reproduce Figure 2 and S4,
run analysis/main/angle_bootstrap.py, then run analysis/main/angle_figure.py

### 2. decoding
To reproduce Figure 3, S5 and S6,
run analysis/main/condition_decoding.py, then run analysis/main/condition_decoding_figure.py

### 3. tradeoff
To reproduce Figure 4 and S7,
run analysis/main/tradeoff_figure.py with the input of tradeoff index = 1 (simple)

To reproduce Figure S12 and S13,
run analysis/main/tradeoff_figure.py with the input of tradeoff index = 2 (probit)

### 4. behavioral data analysis
To reproduce Figure S1,
run analysis/behavior/behav_summary.py

### 5. test-retest reliability
To reproduce Figure S3,
run analysis/main/test-retest_compute.py, then run analysis/main/test-retest_figure.py

### 6. subcategory decoding
To reproduce Figure S8,
run analysis/main/subcategory_decoding.py, then run analysis/main/subcategory_decoding_figure.py

### 7. simulation of tradeoff
To reproduce Figure S11,
run analysis/main/simulation_tradeoff.py
