% Setup
fs         = 256;
win_sec    = 10;
win_samps  = win_sec * fs;
step_samps = round(win_samps * 0.25);

n_per_file = 10;
n_files_max = 10;

low_fres = 256;

% Get order summary
T = readtable('var_order_summary.csv');
order_map = containers.Map(T.patient, T.median_order);


rng(42);   

% Results storage
corr_all = [];
mae_all  = [];

max_windows = 24 * n_files_max * n_per_file;
corr_all = NaN(max_windows,1);
mae_all  = NaN(max_windows,1);
k = 0;


% =====================================================
%  LOOP PATIENT
% =====================================================
for pid = 1:24
    patient = sprintf('chb%02d', pid);
    files = dir(fullfile('Data', [patient '_*.h5']));

    if isempty(files)
        fprintf('No files for %s, skipping\n', patient); continue
    end

    % Randomly choose up to n_files_max files
    files = files(randperm(numel(files), min(n_files_max, numel(files))));

    fprintf('\nProcessing %s\n', patient);

    morder = order_map(patient);

    % =====================================================
    %  LOOP FILES
    % =====================================================

    for f = 1:numel(files)

        filePath = fullfile(files(f).folder, files(f).name);

        X = h5read(filePath,'/signals');
        X=X';
        X = double(X); 
        X = X ./ std(X, 0, 2); 

        n_samples = size(X,2);
        n_wins = floor((n_samples-win_samps)/step_samps)+1;

        sample_wins = randperm(n_wins, min(n_per_file, n_wins));

        % =================================================
        %  LOOP WINDOWS
        % =================================================
        for w = sample_wins
            s   = (w-1)*step_samps + 1; 
            seg = X(:, s : s+win_samps-1);
            seg = demean(seg);

            % ---------- VAR ----------

            [A, SIG] = tsdata_to_var(seg, morder, 'LWR');
            if isbad(A), continue; end

            info = var_info(A, SIG);
            if info.error, continue; end


            % ---------- GC Features ----------
            fres_high = 2^nextpow2(info.acdec);

            feat_high = compute_feat(A, SIG, fs, fres_high);
            feat_low  = compute_feat(A, SIG, fs, low_fres);

            if isempty(feat_high) || isempty(feat_low)
                continue;
            end

            feat_high = feat_high(:);
            feat_low  = feat_low(:);

            % ---------- comparison ----------
            C = corrcoef(feat_high, feat_low);
            r = C(1,2);
            mae = mean(abs(feat_high - feat_low));

            k = k + 1;
            corr_all(k) = r;
            mae_all(k)  = mae;
        end
    end
end

%%
corr_all = corr_all(1:k);
mae_all  = mae_all(1:k);
    

%% =========================================================
%  FINAL RESULTS
% =========================================================
fprintf('\n========== FINAL RESULTS ==========\n');
fprintf('Mean correlation: %.6f\n', mean(corr_all));
fprintf('Std correlation : %.6f\n', std(corr_all));
fprintf('Mean MAE        : %.6e\n', mean(mae_all));



%% SAVE TO CSV
results_table = table(corr_all, mae_all, ...
    'VariableNames', {'correlation','mae'});

writetable(results_table, 'fres_comparison_results.csv');


%% =========================================================
%  FEATURE FUNCTION
% =========================================================
function feat = compute_feat(A, SIG, fs, fres)


    f = var_to_spwcgc(A, SIG, fres);

    nvars = size(A,1);
    nfreqs = size(f,3);
    freq_axis = linspace(0, fs/2, nfreqs);

    bands = [0.5 4;
             4 8;
             8 13;
             13 30;
             30 45];

    F = zeros(nvars,nvars,5);

    for b = 1:5
        bins = (freq_axis >= bands(b,1) & freq_axis <= bands(b,2));
        F(:,:,b) = mean(f(:,:,bins),3);
    end

    mask = ~eye(nvars);
    feat = zeros(5*nvars*(nvars-1),1);

    for b = 1:5
        mat = F(:,:,b);
        feat((b-1)*nvars*(nvars-1)+1 : b*nvars*(nvars-1)) = mat(mask);
    end
    fprintf('Feature vector length: %d\n', length(feat));
    fprintf('Min: %.4f\n', min(feat));
    fprintf('Max: %.4f\n', max(feat));
    fprintf('NaNs: %d\n', sum(isnan(feat)));

end


