%% Setup
fs         = 256;
win_sec    = 10;
win_samps  = win_sec * fs;
step_samps = round(win_samps * 0.25);
momax      = 15;
n_per_file = 30;
n_files_max = 10;

% Read seizure-containing files
fid = fopen('RECORDS-WITH-SEIZURES-H5.txt','r');
seizure_files = textscan(fid,'%s'); fclose(fid);
seizure_files = seizure_files{1};

rng(42);   % reproducibility

summary        = {};   % patient, median, q1, q3, n
all_order_vals = [];   % every individual order
all_order_pats = {};   % patient label per order

%% Loop over all 24 patients
for pid = 17
    patient = sprintf('chb%02d*', pid);
    files = dir(fullfile('Data', [patient '_*.h5']));
    if isempty(files)
        fprintf('No files for %s, skipping\n', patient); continue
    end

    
    % Remove seizure files
    files = files(~ismember({files.name}, seizure_files));
    if isempty(files)
        fprintf('No non-seizure files for %s, skipping\n', patient); continue
    end

    % Randomly choose up to n_files_max files
    files = files(randperm(numel(files), min(n_files_max, numel(files))));

    bic_orders = [];   % RESET per patient

    for f = 1:numel(files)
        filePath = fullfile(files(f).folder, files(f).name);
        X = h5read(filePath,'/signals');
        X=X';
        X = double(X); 

        n_samples = size(X,2);
        n_wins = floor((n_samples-win_samps)/step_samps)+1;
        if n_wins < 1, continue; end

        sample_wins = randperm(n_wins, min(n_per_file, n_wins));
        for w = sample_wins
            s = (w-1)*step_samps + 1;
            seg = demean(X(:, s : s+win_samps-1));
            [~, BIC] = tsdata_to_infocrit(seg, momax, 'LWR');
            [mval, p] = min(BIC);
            if ~isnan(mval) && p > 0
                bic_orders(end+1) = p;
            end
        end
    end

    if isempty(bic_orders)
        fprintf('%s: no valid orders\n', patient); continue
    end

    med = round(median(bic_orders));
    q1  = round(prctile(bic_orders,25));
    q3  = round(prctile(bic_orders,75));
    n   = numel(bic_orders);

    summary(end+1,:)  = {patient, med, q1, q3, n};
    all_order_vals    = [all_order_vals; bic_orders(:)];
    all_order_pats    = [all_order_pats; repmat({patient}, n, 1)];

    fprintf('%s: p = %d (IQR %d-%d, n=%d)\n', patient, med, q1, q3, n);
end

%% Save summary CSV (one row per patient)
T = cell2table(summary, ...
    'VariableNames', {'patient','median_order','q1','q3','n_windows'});
writetable(T, 'var_order_summary_17.csv');

%% Save long-format CSV (every order) for boxplots / distributions
T_all = table(all_order_pats, all_order_vals, ...
    'VariableNames', {'patient','order'});
writetable(T_all, 'var_order_all_23.csv');

fprintf('\nSaved var_order_summary.csv and var_order_all.csv\n');


  