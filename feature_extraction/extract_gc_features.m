function extract_gc_features(filePath)
    % Parameters
    fs           = 256; 
    win_sec      = 10;
    win_samps    = win_sec * fs;                      % 2560
    step_samps   = round(win_samps * 0.25);           % 640 (75% overlap)
    morder       = 5;
    regmode      = 'LWR';
    label_thresh = 0.60;                              % 60% seizure → label 1
    
    
    % For frequency band integration
    nvars        = 18;
    n_bands      = 5;
    feat_len     = n_bands * nvars * (nvars - 1);   
    
    bands = [0.5  4;
            4    8;
            8   13;
            13  30;
            30  45];
    
    % ===========================================
    % -------- Load signals & Preprocess -------
    % ===========================================
    
    X = h5read(filePath, '/signals');
    X = X';
    X = double(X);
    X = X ./ std(X,0,2);
    nsamples = size(X, 2);
    
    % ===========================================
    % -------- Load seizure annotations -------
    % ===========================================
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
    
    % Pre-allocate
    n_windows  = floor((nsamples - win_samps) / step_samps) + 1;
    all_feats  = NaN(feat_len, n_windows, 'double');  % single = half the memory
    all_labels = zeros(n_windows, 1, 'uint8');
    
    fprintf('Record: %s | Windows: %d\n', filePath, n_windows); % Check the number of windows
    
    % Fixed fres=256
    fres=256;
    
    
    % ===========================================
    % ----------- SLIDING WINDOW LOOP -----------
    % ===========================================
    
    for w = 1:n_windows
        fprintf('→ Window: %d\n', w);
        
        start_idx = (w-1)*step_samps + 1;
        end_idx   = start_idx + win_samps - 1;
    
        data = X(:, start_idx:end_idx);
        data = demean(data);              
        
        % ====================================================
        % --- LABEL: Count seizure samples in this window ---
        % ====================================================
        
        seizure_samps = 0;
    
        for s = 1:numel(onset_samps)
            ov_start = max(start_idx, onset_samps(s));
            ov_end   = min(end_idx,   offset_samps(s));
            if ov_start <= ov_end
                seizure_samps = seizure_samps + (ov_end - ov_start + 1);
            end
        end
        
        % Save label
        all_labels(w) = uint8((seizure_samps / win_samps) >= label_thresh);
    
        % ===========================================
        % ----------- FEATURE EXTRACTION -----------
        % ===========================================
    
        % VAR Model ----------------------------------
        [A,SIG] = tsdata_to_var(data,morder,regmode);
        % info = var_info(A, SIG, false);
        % assert(~info.error);

        % Spectral GC -------------------------------
        ptic('\n*** var_to_spwcgc... ');
        f = var_to_spwcgc(A, SIG, fres);
        ptoc;

    
        % Integrate Bands with Trapzoid ---------------------------
        nfreqs = size(f, 3);
        freq_axis = linspace(0, fs/2, nfreqs);
    
        F_bands = zeros(nvars, nvars, 5);
        for b = 1:5
            bins = find(freq_axis >= bands(b,1) & freq_axis <= bands(b,2));
            band_width = bands(b,2) - bands(b,1);
            F_bands(:,:,b) = trapz(freq_axis(bins), f(:,:,bins), 3) / band_width;
        end
    
        % Remove diagonal and turn into a vector ----
        n = nvars;
        mask = ~eye(n);
        feat = zeros(5 * n*(n-1), 1);
        for b = 1:5
            mat = F_bands(:,:,b);
            feat((b-1)*n*(n-1)+1 : b*n*(n-1)) = mat(mask);
        end
    
        % Store results for this window -------------
        all_feats(:,w) = feat;

        fprintf('Min: %.4f\n', min(feat));
        fprintf('Max: %.4f\n', max(feat));
        fprintf('NaNs: %d\n', sum(isnan(feat)));
    end
    
    
    % ===========================================
    % -------------- SAVE FEATURE --------------
    % ===========================================
    
    if ~exist('Features', 'dir')
        mkdir('Features');
    end
    
    [~, recName] = fileparts(filePath);
    
    outFile = fullfile('Features', [recName '_features.mat']);
    
    save(outFile, ...
        'all_feats', 'all_labels', 'fres', 'win_samps', 'step_samps', 'morder', 'bands', 'feat_len', '-v7.3');
    
    fprintf('\nSaved → %s\n', outFile);
    fprintf('Total: %d windows | Seizure: %d | Non-seizure: %d\n', ...
        n_windows, sum(all_labels==1), sum(all_labels==0));
end


