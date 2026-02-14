%% CFD image crop + resize + ellipse mask
% only filenames containing 'N'

rootDir   = pwd;
inputDir  = fullfile(rootDir, 'raw_stimuli', 'face');
outputRoot = fullfile(rootDir, 'raw_stimuli','cropped_face');


imgFiles = [ ...
    dir(fullfile(inputDir, '**', '*.jpg')); ...
    dir(fullfile(inputDir, '**', '*.png')) ...
];

for i = 1:length(imgFiles)

    % N を含むファイルのみ
    if ~contains(imgFiles(i).name, 'N')
        continue;
    end

    inPath = fullfile(imgFiles(i).folder, imgFiles(i).name);

    % 第1階層のみ保持
    relPath = erase(imgFiles(i).folder, [inputDir filesep]);
    parts   = strsplit(relPath, filesep);
    firstLevel = parts{1};

    outDir = fullfile(outputRoot, firstLevel);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    outPath = fullfile(outDir, imgFiles(i).name);

    %% read
    img = imread(inPath);

    %% fixed crop
    img = img(201:1510, 568:1877, :);

    %% resize
    img = imresize(img, [256 256]);

    %% ellipse mask (Python Circle 相当)
    imgSize = 256;
    a = 10800;
    b = 16000;

    [x, y] = meshgrid(1:imgSize, 1:imgSize);
    mask = (((x - 128).^2)/a + ((y - 128).^2)/b) >= 1;

    for c = 1:size(img,3)
        tmp = img(:,:,c);
        tmp(mask) = 255;
        img(:,:,c) = tmp;
    end

    %% save
    imwrite(img, outPath);
end
