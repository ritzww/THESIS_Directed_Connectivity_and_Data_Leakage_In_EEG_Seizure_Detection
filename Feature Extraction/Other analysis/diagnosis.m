%% ==========================================================
%  HDF5 EEG DIAGNOSTICS
%% ==========================================================

clear; clc;

fp = fullfile('Data','chb04_19.h5');

%% ----------------------------------------------------------
% Load first dataset
%% ----------------------------------------------------------

info = h5info(fp);

fprintf('Datasets found:\n');
for k = 1:length(info.Datasets)
    fprintf('  %d : %s\n',k,info.Datasets(k).Name);
end

ds = info.Datasets(1).Name;

raw = h5read(fp,['/' ds]);

fprintf('\nDataset: %s\n',ds);
fprintf('Size    : [%s]\n',num2str(size(raw)));

%% ----------------------------------------------------------
% Convert to double
%% ----------------------------------------------------------

X = double(raw);

% Put channels in rows
if size(X,1) > size(X,2)
    X = X.';
end

X2 = X;

nCh = size(X2,1);
nSamples = size(X2,2);

fprintf('\nChannels : %d\n',nCh);
fprintf('Samples  : %d\n',nSamples);

%% ----------------------------------------------------------
% Check for NaN / Inf
%% ----------------------------------------------------------

nNaN = sum(isnan(X2(:)));
nInf = sum(isinf(X2(:)));

fprintf('\nNaN values : %d\n',nNaN);
fprintf('Inf values : %d\n',nInf);

if nNaN > 0 || nInf > 0

    fprintf('\nChannels containing NaN/Inf:\n');

    [r,~] = find(~isfinite(X2));

    disp(unique(r)');

end

%% ----------------------------------------------------------
% Variance check
%% ----------------------------------------------------------

v = var(X2,0,2);

fprintf('\nMin variance : %.3e\n',min(v));
fprintf('Max variance : %.3e\n',max(v));

dead = find(~isfinite(v) | v < 1e-12);

if isempty(dead)

    fprintf('No dead channels found.\n');

else

    fprintf('Dead channels:\n');
    disp(dead');

end

%% ----------------------------------------------------------
% Remove bad channels for diagnostics
%% ----------------------------------------------------------

keep = isfinite(v) & (v > 1e-12);

Xclean = X2(keep,:);

fprintf('\nRemaining channels after cleaning: %d\n',size(Xclean,1));

%% ----------------------------------------------------------
% Correlation matrix
%% ----------------------------------------------------------

R = corrcoef(Xclean.');

fprintf('NaNs in correlation matrix: %d\n',sum(isnan(R(:))));

nChClean = size(Xclean,1);

R(1:nChClean+1:end) = 0;

[ii,jj] = find(triu(abs(R) > 0.9999));

if isempty(ii)

    fprintf('No duplicate channels found.\n');

else

    fprintf('\nDuplicate channels:\n');

    for k = 1:length(ii)

        fprintf('Ch %d <-> Ch %d   r = %.8f\n', ...
            ii(k), jj(k), R(ii(k),jj(k)));

    end

end

%% ----------------------------------------------------------
% Covariance diagnostics
%% ----------------------------------------------------------

C = cov(Xclean.');

fprintf('\nNaNs in covariance matrix: %d\n', ...
    sum(isnan(C(:))));

fprintf('Infs in covariance matrix: %d\n', ...
    sum(isinf(C(:))));

if all(isfinite(C(:)))

    fprintf('\nCovariance rank      : %d / %d\n', ...
        rank(C), size(C,1));

    fprintf('Covariance condition : %.3e\n', ...
        cond(C));

    e = sort(eig(C));

    fprintf('Smallest eigenvalue  : %.3e\n', e(1));
    fprintf('Largest eigenvalue   : %.3e\n', e(end));

else

    fprintf('\nCovariance matrix contains NaN/Inf.\n');

end

%% ----------------------------------------------------------
% Maximum channel correlation
%% ----------------------------------------------------------

Rtmp = abs(R);
Rtmp(1:size(Rtmp,1)+1:end) = 0;

[maxCorr,idx] = max(Rtmp(:));

[r,c] = ind2sub(size(Rtmp),idx);

fprintf('\nMaximum correlation: %.6f\n',maxCorr);
fprintf('Between channels %d and %d\n',r,c);

%% ----------------------------------------------------------
% Optional heatmap
%% ----------------------------------------------------------

figure;
imagesc(abs(R));
axis square;
colorbar;
title('Absolute Channel Correlation');
xlabel('Channel');
ylabel('Channel');

%% ----------------------------------------------------------
% Channel labels (if available)
%% ----------------------------------------------------------

try

    labels = h5read(fp,'/channel_labels');

    fprintf('\nChannel labels:\n');
    disp(labels);

catch

    fprintf('\n(no channel_labels dataset found)\n');

end

%%
fp = fullfile('Data','chb04_19.h5');

X = double(h5read(fp,'/signals'));

size(X)

if size(X,1) > size(X,2)
    X = X.';
end

fprintf('Channels = %d\n',size(X,1));
fprintf('Samples  = %d\n',size(X,2));

fprintf('NaNs = %d\n',sum(isnan(X(:))));
fprintf('Infs = %d\n',sum(isinf(X(:))));

v = var(X,0,2);

fprintf('Min variance = %.3e\n',min(v));
fprintf('Max variance = %.3e\n',max(v));

C = cov(X.');

fprintf('Rank = %d / %d\n',rank(C),size(C,1));
fprintf('Cond = %.3e\n',cond(C));