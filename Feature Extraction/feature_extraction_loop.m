
if isempty(gcp('nocreate'))
    parpool('Processes', 2, 'SpmdEnabled', false);
end

dataDir = 'Data';
files = dir(fullfile(dataDir,'*.h5'));

%%
parfor k = 1:numel(files)
    filePath = fullfile(files(k).folder, files(k).name);
    fprintf("Processing %s\n", filePath);
    extract_gc_features(filePath);
end
