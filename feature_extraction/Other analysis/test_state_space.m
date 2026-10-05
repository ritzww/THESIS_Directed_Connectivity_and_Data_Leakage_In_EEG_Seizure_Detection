%% Parameters
filePath  = fullfile('Data', 'chb01_03.h5');
fs        = 256;
win_sec   = 10;
win_samps = win_sec * fs;                        % 2560
step_samps = round(win_samps * 0.25);            % 75% overlap = 640 step
morder    = 5;                                   % Model order
regmode   = 'LWR';
fres      = [];                                  % auto-calculated by state-space method

%% Load & preprocess
X = h5read(filePath, '/signals');
X = X';
X = double(X);
X = X ./ std(X,0,2);

%% Single window test
w = 1200;
start_idx = (w-1)*step_samps + 1;
end_idx   = start_idx + win_samps - 1;
% Per window:
data = X(:, start_idx:end_idx);
data = demean(data);              % zero-mean (required by MVGC)


%% VAR model estimation (<mvgc_schema.html#3 |A2|>)

% Estimate VAR model of selected order from data.

ptic('\n*** tsdata_to_var... ');
[A,SIG] = tsdata_to_var(data,morder,regmode);
assert(~isbad(A),'VAR estimation failed - bailing out');
ptoc;


info = var_info(A,SIG);
assert(~info.error,'VAR error(s) found - bailing out');


%% Granger causality calculation: frequency domain  (<mvgc_schema.html#3 |A14|>)

% If not specified, we set the frequency resolution to something sensible. Warn if
% resolution is very large, as this may lead to excessively long computation times,
% and/or out-of-memory issues.

if isempty(fres)
    fres = 2^nextpow2(info.acdec); % based on autocorrelation decay; alternatively, you could try fres = 2^nextpow2(nobs);
    fprintf('\nfrequency resolution auto-calculated as %d (increments ~ %.2gHz)\n',fres,fs/2/fres);
end
if fres > 20000 % adjust to taste
    fprintf(2,'\nWARNING: large frequency resolution = %d - may cause computation time/memory usage problems\nAre you sure you wish to continue [y/n]? ',fres);
    istr = input(' ','s'); if isempty(istr) || ~strcmpi(istr,'y'); fprintf(2,'Aborting...\n'); return; end
end

% Calculate spectral pairwise-conditional causalities at given frequency
% resolution by state-space method.

ptic('\n*** var_to_spwcgc... ');
f = var_to_spwcgc(A,SIG,fres);
assert(~isbad(f,false),'spectral GC calculation failed - bailing out');
ptoc;

%% Setting up for frequency band
% After f is computed...
nvars      = 18;                         

nfreqs = size(f, 3);
freq_axis = linspace(0, fs/2, nfreqs);

bands = [0.5  4;
    4    8;
    8   13;
    13  30;
    30  45];

%% Average over frequency bands
F_bands = zeros(nvars, nvars, 5);
for b = 1:5
    bins = find(freq_axis >= bands(b,1) & freq_axis <= bands(b,2));
    F_bands(:,:,b) = mean(f(:,:,bins), 3);
end

n = nvars;
mask = ~eye(n);
feat = zeros(5 * n*(n-1), 1);
for b = 1:5
    mat = F_bands(:,:,b);
    feat((b-1)*n*(n-1)+1 : b*n*(n-1)) = mat(mask);
end


fprintf('Feature vector length: %d\n', length(feat));
fprintf('Min: %.4f\n', min(feat));
fprintf('Max: %.4f\n', max(feat));
fprintf('NaNs: %d\n', sum(isnan(feat)));

%% Integrate frequency bands
F_bands = zeros(nvars, nvars, 5);
for b = 1:5
    bins = find(freq_axis >= bands(b,1) & freq_axis <= bands(b,2));
    band_width = bands(b,2) - bands(b,1);
    F_bands(:,:,b) = trapz(freq_axis(bins), f(:,:,bins), 3) / band_width;
end

n = nvars;
mask = ~eye(n);
feat = zeros(5 * n*(n-1), 1);
for b = 1:5
    mat = F_bands(:,:,b);
    feat((b-1)*n*(n-1)+1 : b*n*(n-1)) = mat(mask);
end

fprintf('Feature vector length: %d\n', length(feat));
fprintf('Min: %.4f\n', min(feat));
fprintf('Max: %.4f\n', max(feat));
fprintf('NaNs: %d\n', sum(isnan(feat)));


%% Plot the figure
%
figure;
imagesc(F_bands(:,:,2));  % theta band
colorbar;
title('Theta band GC - window w');
xlabel('Source channel');
ylabel('Target channel');







%% TEST: high -----------------------------------------------------------------------------

filePath  = fullfile('Data', 'chb01_03.h5');
fs        = 256;
win_sec   = 10;
win_samps = win_sec * fs;                        % 2560
step_samps = round(win_samps * 0.25);            % 75% overlap = 640 step
morder    = 5;                                   % Model order
regmode   = 'LWR';
fres      = 1024;                                  % auto-calculated by state-space method

X = h5read(filePath, '/signals');
X = X';
X = double(X);
X = X ./ std(X,0,2);


w = 1;
start_idx = (w-1)*step_samps + 1;
end_idx   = start_idx + win_samps - 1;
data = X(:, start_idx:end_idx);
data = demean(data);              % zero-mean (required by MVGC)


ptic('\n*** tsdata_to_var... ');
[A,SIG] = tsdata_to_var(data,morder,regmode);
assert(~isbad(A),'VAR estimation failed - bailing out');
ptoc;


info = var_info(A,SIG);
assert(~info.error,'VAR error(s) found - bailing out');


ptic('\n*** var_to_spwcgc... ');
f = var_to_spwcgc(A,SIG,fres);
assert(~isbad(f,false),'spectral GC calculation failed - bailing out');
ptoc;


nfreqs = size(f, 3);
freq_axis = linspace(0, fs/2, nfreqs);

bands = [0.5  4;
    4    8;
    8   13;
    13  30;
    30  45];


F_bands = zeros(nvars, nvars, 5);
for b = 1:5
    bins = find(freq_axis >= bands(b,1) & freq_axis <= bands(b,2));
    band_width = bands(b,2) - bands(b,1);
    F_bands(:,:,b) = trapz(freq_axis(bins), f(:,:,bins), 3) / band_width;
end

n = nvars;
mask = ~eye(n);
feat = zeros(5 * n*(n-1), 1);
for b = 1:5
    mat = F_bands(:,:,b);
    feat((b-1)*n*(n-1)+1 : b*n*(n-1)) = mat(mask);
end

fprintf('Feature vector length: %d\n', length(feat));
fprintf('Min: %.4f\n', min(feat));
fprintf('Max: %.4f\n', max(feat));
fprintf('NaNs: %d\n', sum(isnan(feat)));

feat_high = feat;

%% TEST: low -----------------------------------------------------------------------------

filePath  = fullfile('Data', 'chb01_03.h5');
fs        = 256;
win_sec   = 10;
win_samps = win_sec * fs;                        % 2560
step_samps = round(win_samps * 0.25);            % 75% overlap = 640 step
morder    = 5;                                   % Model order
regmode   = 'LWR';
fres      = 256;                                  % auto-calculated by state-space method

X = h5read(filePath, '/signals');
X = X';
X = double(X);
X = X ./ std(X,0,2);


w = 1;
start_idx = (w-1)*step_samps + 1;
end_idx   = start_idx + win_samps - 1;
data = X(:, start_idx:end_idx);
data = demean(data);              % zero-mean (required by MVGC)


ptic('\n*** tsdata_to_var... ');
[A,SIG] = tsdata_to_var(data,morder,regmode);
assert(~isbad(A),'VAR estimation failed - bailing out');
ptoc;


info = var_info(A,SIG);
assert(~info.error,'VAR error(s) found - bailing out');


ptic('\n*** var_to_spwcgc... ');
f = var_to_spwcgc(A,SIG,fres);
assert(~isbad(f,false),'spectral GC calculation failed - bailing out');
ptoc;

nvars      = 18;                         
n_bands    = 5;   

nfreqs = size(f, 3);
freq_axis = linspace(0, fs/2, nfreqs);

bands = [0.5  4;
    4    8;
    8   13;
    13  30;
    30  45];


F_bands = zeros(nvars, nvars, 5);
for b = 1:5
    bins = find(freq_axis >= bands(b,1) & freq_axis <= bands(b,2));
    band_width = bands(b,2) - bands(b,1);
    F_bands(:,:,b) = trapz(freq_axis(bins), f(:,:,bins), 3) / band_width;
end

n = nvars;
mask = ~eye(n);
feat = zeros(5 * n*(n-1), 1);
for b = 1:5
    mat = F_bands(:,:,b);
    feat((b-1)*n*(n-1)+1 : b*n*(n-1)) = mat(mask);
end

fprintf('Feature vector length: %d\n', length(feat));
fprintf('Min: %.4f\n', min(feat));
fprintf('Max: %.4f\n', max(feat));
fprintf('NaNs: %d\n', sum(isnan(feat)));

feat_low = feat;

%%
C = corrcoef(feat_high, feat_low);
r = C(1,2);
fmae = mean(abs(feat_high - feat_low));

fprintf('Correlation: %.6f\n', r);
fprintf('MAE: %.6f\n', fmae);
fprintf('MAE as %% of feature range: %.2f%%\n', ...
    100 * fmae / (max(feat_high) - min(feat_high)));