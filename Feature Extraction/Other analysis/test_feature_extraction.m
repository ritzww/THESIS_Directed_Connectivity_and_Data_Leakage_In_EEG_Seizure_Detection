%% Parameters
filePath     = fullfile('Data', 'chb04_28.h5'); %
fs           = 256; 
win_sec      = 10;
win_samps    = win_sec * fs;                      % 2560
step_samps   = round(win_samps * 0.25);           % 640 (75% overlap)
morder       = 5;
regmode      = 'LWR';
% fres         = [];
label_thresh = 0.60;                              % 60% seizure → label 1


% For frequency band integration
nvars        = 18;
n_bands      = 5;
feat_len     = n_bands * nvars * (nvars - 1);     % final feature length 
bands = [0.5  4;
        4    8;
        8   13;
        13  30;
        30  45];


%% Load signals & Preprocess
X = h5read(filePath, '/signals');
X = X';
X = double(X);
X = X ./ std(X,0,2);
%X=X*1e6;


nsamples = size(X, 2);


%% Load seizure annotations 
try
    sz           = h5read(filePath, '/seizure_intervals');  % MATLAB reads as [2 × n_seizures]
    onset_samps  = sz(1,:)' + 1;   % +1: Python is 0-indexed, MATLAB is 1-indexed
    offset_samps = sz(2,:)';       % offset stays as-is (exclusive end → last valid sample)
catch
    onset_samps = []; offset_samps = [];
    warning('No seizure annotations found.');
end

% Sanity check — print in seconds
for s = 1:numel(onset_samps)
    fprintf('  Seizure %d: %.1fs → %.1fs (%.1fs)\n', s, ...
        onset_samps(s)/fs, offset_samps(s)/fs, (offset_samps(s)-onset_samps(s))/fs);
end

%% Pre-allocate
n_windows  = floor((nsamples - win_samps) / step_samps) + 1;
all_feats  = NaN(feat_len, n_windows, 'double');  % single = half the memory
all_labels = zeros(n_windows, 1, 'uint8');

fprintf('Record: %s | Windows: %d\n', filePath, n_windows); % Check the number of windows

%% Compute fres from pilot window (middle of record)
% pilot_idx  = round(nsamples / 2);
% pilot_start = max(1, pilot_idx - round(win_samps/2));
% pilot_end   = pilot_start + win_samps - 1;
% 
% pilot_data  = X(:, pilot_start:pilot_end);
% pilot_data  = demean(pilot_data);
% 
% [A_pilot, SIG_pilot] = tsdata_to_var(pilot_data, morder, regmode);
% assert(~isbad(A_pilot), 'Pilot VAR estimation failed');
% 
% info_pilot = var_info(A_pilot, SIG_pilot);
% assert(~info_pilot.error, 'Pilot VAR error - bailing out');
% 
% fres = 2^nextpow2(info_pilot.acdec);
% fprintf('Frequency resolution fixed from pilot window: %d (increments ~ %.2f Hz)\n', ...
%     fres, fs/2/fres);
% 
% % Guard against unreasonably large fres
% if fres > 20000 % adjust to taste
%     fprintf(2,'\nWARNING: large frequency resolution = %d - may cause computation time/memory usage problems\nAre you sure you wish to continue [y/n]? ',fres);
%     istr = input(' ','s'); if isempty(istr) || ~strcmpi(istr,'y'); fprintf(2,'Aborting...\n'); return; end
% end


% Fixed fres=256
fres=256;



%% Sliding window loop (parallel)
parfor w = 1:n_windows
    fprintf('→ Window: %d\n', w);
    start_idx = (w-1)*step_samps + 1;
    end_idx   = start_idx + win_samps - 1;
    data = X(:, start_idx:end_idx);
    data = demean(data);              % zero-mean (required by MVGC)
    
    % ── LABEL: count seizure samples in this window ──────────────────────
    seizure_samps = 0;
    for s = 1:numel(onset_samps)
        ov_start = max(start_idx, onset_samps(s));
        ov_end   = min(end_idx,   offset_samps(s));
        if ov_start <= ov_end
            seizure_samps = seizure_samps + (ov_end - ov_start + 1);
        end
    end
    all_labels(w) = uint8((seizure_samps / win_samps) >= label_thresh);
    % label = uint8((seizure_samps / win_samps) >= label_thresh);

    % ── FEATURE EXTRACTION ───────────────────────────────────────────────
     % Estimate VAR model of selected order from data.

    ptic('\n*** tsdata_to_var... ');
    [A,SIG] = tsdata_to_var(data,morder,regmode);
    assert(~isbad(A),'VAR estimation failed - bailing out');
    ptoc;


    info = var_info(A,SIG, false);
    assert(~info.error,'VAR error(s) found - bailing out');

    
    % Calculate spectral pairwise-conditional causalities at given frequency
    % resolution by state-space method.
    
    ptic('\n*** var_to_spwcgc... ');
    f = var_to_spwcgc(A,SIG,fres);
    assert(~isbad(f,false),'spectral GC calculation failed - bailing out');
    ptoc;

    % After f is computed...
    nfreqs = size(f, 3);
    freq_axis = linspace(0, fs/2, nfreqs);

    % Integrate frequency bands
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

    % fprintf('Feature vector length: %d\n', length(feat));
    % fprintf('Min: %.4f\n', min(feat));
    % fprintf('Max: %.4f\n', max(feat));
    fprintf('NaNs: %d\n', sum(isnan(feat)));

    % Store results for this window
    all_feats(:,w) = feat;
    %all_labels(w)   = label;
end

%% ── SAVE ────────────────────────────────────────────────────────────────
if ~exist('Features', 'dir')
    mkdir('Features');
end

[~, recName] = fileparts(filePath);

outFile = fullfile('Features', [recName '_gc_features_128.mat']);

save(outFile, ...
    'all_feats', 'all_labels', ...
    'fres', 'win_samps', 'step_samps', ...
    'morder', 'bands', 'feat_len', ...
    '-v7.3');

fprintf('\nSaved → %s\n', outFile);
fprintf('Total: %d windows | Seizure: %d | Non-seizure: %d\n', ...
    n_windows, sum(all_labels==1), sum(all_labels==0));

n_valid = sum(~all(isnan(all_feats), 1));
fprintf('Total: %d windows | valid: %d | failed(NaN): %d\n', ...
    n_windows, n_valid, n_windows - n_valid);



%% Plot
% figure;
% imagesc(F_bands(:,:,2));  % theta band
% colorbar;
% title('Theta band GC - window w');
% xlabel('Source channel');
% ylabel('Target channel');
