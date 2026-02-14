#!/bin/sh
proj_top=/SINLAB/SIN/tom/all_BIDS
work_dir=${proj_top}/data/fmriprep_wdr

singularity run --cleanenv -B ${proj_top}:${proj_top} /SINLAB/SIN/fmriprep/fmriprep-23-2-1.simg \
${proj_top}/data/BIDS ${proj_top}/data/derivatives participant --participant-label 01 02 04 05 06 07 08 09 \
	--fs-license-file ${proj_top}/license.txt \
	--fs-subjects-dir ${proj_top}/data/FreeSurfer \
	--output-spaces fsaverage \
	--mem_mb 25600 \
	--n_cpus 4 \
	-w ${work_dir} \
	--derivatives ${proj_top}/data/derivatives \
	--omp-nthreads 4 \
    	--random-seed 1 \
	--skip-bids-validation \
	--write-graph 