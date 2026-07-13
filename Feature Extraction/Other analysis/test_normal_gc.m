%% Parameters
filePath  = fullfile('Data', 'chb01_03.h5');
fs        = 256;
win_sec   = 10;
win_samps = win_sec * fs;                        % 2560
step_samps = round(win_samps * 0.25);            % 75% overlap = 640 step
morder    = 5;                                   % Model order
regmode   = 'LWR';
acmaxlags = [];                                 
fres      = [];                                 

%% Load & preprocess
X = h5read(filePath, '/signals');             
X = X';                             
X = double(X);                              

%% Sliding window setup
n_samples = size(X, 2);
n_wins    = floor((n_samples - win_samps) / step_samps) + 1;

%% Preallocate output
nvars = size(X, 1);                              % 22 channels
F_all = nan(nvars, nvars, n_wins);               % time-domain GC
f_all = cell(n_wins, 1);                         % spectral GC (freq x nvars x nvars)

%%
w = 1200;  % pick any window number to test

start_idx = (w-1)*step_samps + 1;
end_idx   = start_idx + win_samps - 1;

% Extract and demean
data = demean(X(:, start_idx:end_idx));





%% VAR model estimation (<mvgc_schema.html#3 |A2|>)

% Estimate VAR model of selected order from data.

ptic('\n*** tsdata_to_var... ');
[A,SIG] = tsdata_to_var(data,morder,regmode);
ptoc;

% Check for failed regression

assert(~isbad(A),'VAR estimation failed');

% NOTE: at this point we have a model and are finished with the data! - all
% subsequent calculations work from the estimated VAR parameters A and SIG.

%% Autocovariance calculation (<mvgc_schema.html#3 |A5|>)
ptic('*** var_to_autocov... ');
[G,info] = var_to_autocov(A,SIG,acmaxlags);
ptoc;

var_acinfo(info,true); % report results (and bail out on error)

%% Granger causality calculation: frequency domain  (<mvgc_schema.html#3 |A14|>)


ptic('\n*** autocov_to_spwcgc... ');
f = autocov_to_spwcgc(G,1024);
ptoc;


assert(~isbad(f,false),'spectral GC calculation failed');

%%
nfreqs = size(f, 3);
freq_axis = linspace(0, fs/2, nfreqs);
fprintf('Frequency resolution: %.3f Hz\n', freq_axis(2) - freq_axis(1));

%%
bands = [0.5  4;
    4    8;
    8   13;
    13  30;
    30  45];
band_names = {'delta','theta','alpha','beta','gamma'};

for b = 1:5
    bins = find(freq_axis >= bands(b,1) & freq_axis <= bands(b,2));
    fprintf('%s: %d bins (%.2f to %.2f Hz)\n', ...
        band_names{b}, numel(bins), freq_axis(bins(1)), freq_axis(bins(end)));
end

%%
% Preallocate [18 x 18 x 5] — one matrix per band
F_bands = zeros(18, 18, 5);

for b = 1:5
    bins = find(freq_axis >= bands(b,1) & freq_axis <= bands(b,2));
    % Sum over frequency bins and normalise by number of bins
    F_bands(:,:,b) = mean(f(:,:,bins), 3);
end

% Check
fprintf('Feature matrix size: %s\n', mat2str(size(F_bands)));

%%
% Remove diagonal (self-causality = NaN) and flatten
n = 18;
mask = ~eye(n);  % exclude diagonal
feat = zeros(5 * n*(n-1), 1);

for b = 1:5
    mat = F_bands(:,:,b);
    feat((b-1)*n*(n-1)+1 : b*n*(n-1)) = mat(mask);
end

fprintf('Feature vector length: %d\n', length(feat));
% Should be 5 * 18 * 17 = 1530

%%
fprintf('Feature vector length: %d\n', length(feat));
fprintf('Min: %.4f\n', min(feat));
fprintf('Max: %.4f\n', max(feat));
fprintf('NaNs: %d\n', sum(isnan(feat)));
fprintf('Zeros: %d\n', sum(feat == 0));

%%
figure;
imagesc(F_bands(:,:,2));  % theta band
colorbar;
title('Theta band GC - window w');
xlabel('Source channel');
ylabel('Target channel');

%%
% Set the new parameters
acmaxlags = 500;
fres = 256;

% Time each step
tic;
[G, info] = var_to_autocov(A, SIG, acmaxlags);
t1 = toc; fprintf('var_to_autocov: %.2f sec\n', t1);

tic;
f = autocov_to_spwcgc(G, fres);
t2 = toc; fprintf('autocov_to_spwcgc: %.2f sec\n', t2);

fprintf('Total: %.2f sec\n', t1+t2);

%%
tic;
[G, info] = var_to_autocov(A, SIG, []);
t1 = toc; fprintf('var_to_autocov: %.2f sec\n', t1);

tic;
f = autocov_to_spwcgc(G, 1024);
t2 = toc; fprintf('autocov_to_spwcgc: %.2f sec\n', t2);

fprintf('Total: %.2f sec\n', t1+t2);