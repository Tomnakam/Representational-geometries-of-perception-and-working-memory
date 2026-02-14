%% =========================
% Face: match HSV(V) with face_corresp.csv (col2=assigned number, col3=filename)
%% =========================
olddir = pwd;
load('PRM_OverallParameters.mat');

inRoot  = fullfile(olddir, 'raw_stimuli', 'cropped_face');
outRoot = fullfile(olddir, 'hsv-matched', 'face');

numExemplarFace = [104,93,90,93];
imgSize  = [256, 256];

% ellipse parameters
centerX = 128; centerY = 128; a = 100; b = 126;
[x, y] = meshgrid(1:imgSize(2), 1:imgSize(1));
ellipse = ((x - centerX).^2) / a^2 + ((y - centerY).^2) / b^2 > 1;

% --- read CSV ---
csvPath = fullfile(olddir, 'face_corresp.csv');

% ヘッダあり/なし両対応で読む
opts = detectImportOptions(csvPath);
T = readtable(csvPath, opts);

% もしヘッダなしで Var1,Var2,Var3 になってても列番号で取る
assignedNum = T{:,2};     % 2列目：出力番号
srcNameCol  = T{:,3};     % 3列目：入力ファイル名

% 文字列化
srcFile = string(srcNameCol);

nFaceTotal = sum(numExemplarFace);
hsv_img = zeros(imgSize(1), imgSize(2), 3, nFaceTotal);
v_mean  = zeros(1, nFaceTotal);

face_count = 0;

for i = 1:length(face_category)
    cat = face_category{i};
    for j = 1:numExemplarFace(i)
        face_count = face_count + 1;

        % 入力ファイル名：CSV 3列目
        inName = srcFile(face_count);

        % 典型: inRoot/<category>/<filename>
        inPath = fullfile(inRoot, cat, inName);
        if ~isfile(inPath)
            % 念のため fallback（CSVが相対パスを持ってる場合など）
            inPath = fullfile(inRoot, inName);
        end

        img = im2double(imread(inPath));
        hsv = rgb2hsv(img);
        hsv_img(:,:,:,face_count) = hsv;

        v = hsv(:,:,3);
        v_mean(face_count) = mean(v(ellipse), 'all');  
    end
end

% V set to min(v_mean)
v_target = min(v_mean);
for k = 1:nFaceTotal
    scale = v_target / v_mean(k);

    v = hsv_img(:,:,3,k);
    v = v * scale; 
    hsv_img(:,:,3,k) = min(max(v,0),1);
end

% --- save ---
if ~exist(outRoot, 'dir'), mkdir(outRoot); end

face_count = 0;
for i = 1:length(face_category)
    cat = face_category{i};

    outDir = fullfile(outRoot, cat);
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    for j = 1:numExemplarFace(i)
        face_count = face_count + 1;

        % 出力番号：CSV 2列目
        outNum = assignedNum(face_count);

        outName = sprintf('%s%d.jpg', cat, outNum);
        outPath = fullfile(outDir, outName);

        new_rgb = hsv2rgb(hsv_img(:,:,:,face_count));
        new_rgb = min(max(new_rgb,0),1);
        imwrite(new_rgb, outPath);
    end
end

cd(olddir);

%% =========================
% Scene: match HSV(V) with scene_corresp.csv (col2=assigned number, col3=filename)
%% =========================
olddir = pwd;
load('PRM_OverallParameters.mat');

inRoot  = fullfile(olddir, 'raw_stimuli', 'scene');
outRoot = fullfile(olddir, 'hsv-matched', 'scene');

numExemplarScene = [68,68,68,68];
nSceneTotal = sum(numExemplarScene);

imgSize  = [256, 256];

% --- read CSV ---
csvPath = fullfile(olddir, 'scene_corresp.csv');

opts = detectImportOptions(csvPath);
T = readtable(csvPath, opts);

assignedNum = T{:,2};    % 2列目：出力番号
srcNameCol  = T{:,3};    % 3列目：入力ファイル名
srcFile = string(srcNameCol);

% --- compute HSV + V mean ---
hsv_img = zeros(imgSize(1), imgSize(2), 3, nSceneTotal);
v_mean  = zeros(1, nSceneTotal);

scene_count = 0;
for i = 1:length(scene_category)
    cat = scene_category{i};

    for j = 1:numExemplarScene(i)
        scene_count = scene_count + 1;

        % 入力ファイル名（CSV 3列目）
        inName = srcFile(scene_count);

        % 典型: inRoot/<category>/<filename>
        inPath = fullfile(inRoot, cat, inName);
        if ~isfile(inPath)
            % fallback: inRoot/<filename>
            inPath = fullfile(inRoot, inName);
        end

        hsv = rgb2hsv(im2double(imread(inPath)));
        hsv_img(:,:,:,scene_count) = hsv;

        v_mean(scene_count) = mean(hsv(:,:,3), 'all');
    end
end

% --- match V to min ---
v_target = min(v_mean);
for k = 1:nSceneTotal
    hsv_img(:,:,3,k) = hsv_img(:,:,3,k) / v_mean(k) * v_target;
    hsv_img(:,:,3,k) = min(max(hsv_img(:,:,3,k),0),1);
end

% --- save ---
if ~exist(outRoot, 'dir'), mkdir(outRoot); end

scene_count = 0;
for i = 1:length(scene_category)
    cat = scene_category{i};

    outDir = fullfile(outRoot, cat);
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    for j = 1:numExemplarScene(i)
        scene_count = scene_count + 1;

        outNum = assignedNum(scene_count);  % CSV 2列目

        outName = sprintf('%s%d.jpg', cat, outNum);
        outPath = fullfile(outDir, outName);

        rgb = hsv2rgb(hsv_img(:,:,:,scene_count));
        rgb = min(max(rgb,0),1);
        imwrite(rgb, outPath);
    end
end

cd(olddir);