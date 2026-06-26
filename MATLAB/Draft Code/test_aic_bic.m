filePath = fullfile('Data', 'chb01_03.h5');
X = h5read(filePath, '/signals');
%%
X=X';
fs = 256;

win_sec = 10;

win_samps  = win_sec * fs;
step_samps = round(win_samps * 0.25);  % 75% overlap

% Number of windows
n_samples = size(X, 2);

% +1 porque remove o ultimo intervalo
n_wins = floor((n_samples - win_samps) / step_samps) + 1;


% Sliding window
windows = cell(n_wins, 1);

for w = 1:n_wins

    start_idx = (w-1)*step_samps + 1;
    end_idx   = start_idx + win_samps - 1;

    seg = X(:, start_idx:end_idx);

    % basic preprocessing
    seg = seg - mean(seg, 2);

    windows{w} = seg;

end
%%
data = double(windows{5});
data = data * 1e6;
data(18, :) = [];          % remove channel 19 (P7-T7)
size(data)
%data = double(windows{5});
data = demean(data);
size(data)
%%
% [AIC, BIC] = tsdata_to_infocrit(data, 20, 'LWR');
%%
% plot(1:20, AIC);
% title('Gof Plot');
% 


%%
data = double(windows{5});
data = data * 1e6;
data = data - mean(data,2);

% Correlation between channels
C = corrcoef(data');

threshold = 0.99;
[row,col] = find(triu(abs(C) > threshold,1));

if isempty(row)
    disp('No near-duplicate channels found')
else
    disp('Near-duplicate channel pairs:')
    for k = 1:length(row)
        fprintf('Ch%d vs Ch%d: r = %.8f\n', ...
            row(k), col(k), C(row(k),col(k)));
    end
end

disp('Absolute correlation matrix:')
disp(round(abs(C)*1000)/1000)

%%
fp = fullfile('Data','chb04_19.h5');

info = h5info(fp);

disp('Datasets in file:')
for k = 1:length(info.Datasets)
    fprintf('%d : %s\n', k, info.Datasets(k).Name);
end

ds = info.Datasets(1).Name;

raw = h5read(fp,['/' ds]);

fprintf('Dataset "%s"\n',ds);
fprintf('Size: [%s]\n',num2str(size(raw)));

X = double(raw);

% Put channels in rows
if size(X,1) > size(X,2)
    X = X.';
end

X2 = X;

nCh = size(X2,1);

fprintf('\nChannels: %d\n',nCh);
fprintf('Samples : %d\n',size(X2,2));


%% Dead channels

v = var(X2,0,2);

dead = find(v < eps(max(v)));

if isempty(dead)
    fprintf('\nNo dead channels found.\n');
else
    fprintf('\nDead channels:\n');
    disp(dead')
end

%% Duplicate channels

R = corrcoef(X2.');

R(1:nCh+1:end) = 0;

[ii,jj] = find(triu(abs(R) > 0.9999));

if isempty(ii)

    fprintf('\nNo duplicate channels found.\n');

else

    fprintf('\nDuplicate channels:\n');

    for k = 1:numel(ii)

        fprintf('Ch %d <-> Ch %d   r = %.8f\n', ...
            ii(k), jj(k), R(ii(k),jj(k)));

    end

end

%% Covariance diagnostics

C = cov(X2.');

fprintf('\nCovariance rank      : %d / %d\n',rank(C),nCh);
fprintf('Covariance condition : %.3e\n',cond(C));



e = sort(eig(C));

fprintf('\nSmallest eigenvalue : %.3e\n',e(1));
fprintf('Largest eigenvalue  : %.3e\n',e(end));

%% Labels (if available)

try
    labels = h5read(fp,'/channel_labels');
    disp('Channel labels:')
    disp(labels)
catch
    fprintf('\n(no channel_labels dataset found)\n');
end

