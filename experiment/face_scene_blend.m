clear; close all;

id = input('id=');

% ---- load parameters ----
load('PRM_OverallParameters.mat');

olddir = pwd;

% ================================
% read correspondance table
% ================================
faceTbl  = readtable(fullfile(olddir, 'face_corresp.csv'));
sceneTbl = readtable(fullfile(olddir, 'scene_corresp.csv'));

% lookup function
getFaceFile  = @(cat, ex) ...
    faceTbl.FileName{faceTbl.FaceCategory==cat & faceTbl.FaceExemplar==ex};

getSceneFile = @(cat, ex) ...
    sceneTbl.FileName{sceneTbl.SceneCategory==cat & sceneTbl.SceneExemplar==ex};

% ---- load trial from saved rawdata ----
cd('rawdata');
folder_name = ['sub', num2str(id)];
cd(folder_name);
load(['rawdata_PRMdecoding_exp1_', num2str(id)], 'trial');

% ---- ensure stimuli directory exists ----
if ~exist(fullfile(pwd, 'stimuli'), 'dir')
    mkdir('stimuli');
end
cd(olddir);

% ---- generate and save blended images per trial ----
for run = 1:32
    for n = 1:16

        % ===== Face =====
        fcat = face_category{trial(n,3,run)};
        fname = sprintf('%s%d.jpg', fcat, trial(n,4,run));

        cd('hsv-matched');
        cd('face');
        cd(fcat);

        f_img_sample = im2double(imread(fname));

        if trial(n,8,run)==1
            filename = sprintf('%s%d.jpg', fcat, trial(n,9,run));
            f_img_test=im2double(imread(filename));
        end
        cd(olddir);

        % ===== Scene =====
        scat = scene_category{trial(n,5,run)};
        sname = sprintf('%s%d.jpg', scat, trial(n,6,run));

        cd('hsv-matched');
        cd('scene');
        cd(scat);

        s_img_sample = im2double(imread(sname));

        if trial(n,8,run) == 1
            filename = sprintf('%s%d.jpg', scat, trial(n,10,run));
            s_img_test=im2double(imread(filename));
        end
        cd(olddir);

        % ===== Save blended images =====
        cd('rawdata');
        cd(folder_name);
        cd('stimuli');

        blended_img_sample = face_alpha * f_img_sample + (1 - face_alpha) * s_img_sample;
        imwrite(blended_img_sample, sprintf('sample_run%d_trial%d.png', run, n));

        if trial(n,8,run) == 1
            blended_img_test = face_alpha * f_img_test + (1 - face_alpha) * s_img_test;
            imwrite(blended_img_test, sprintf('test_run%d_trial%d.png', run, n));
        end

        cd(olddir);
    end
end
