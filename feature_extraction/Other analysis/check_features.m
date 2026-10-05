S = load('Features/chb01_03_gc_features.mat');

S.all_feats;
S.all_labels;
S.morder;


nan_count = sum(isnan(S.all_feats(:)));
fprintf('Total NaNs: %d\n', nan_count);

valid_cols = sum(~any(isnan(S.all_feats), 1));

fprintf('Valid columns (no NaNs): %d\n', valid_cols);


%%
old = load('Features/before_features.mat');
new = load('Features/chb01_03_gc_features.mat');

valid_old = ~all(isnan(old.all_feats),1);
valid_new = ~all(isnan(new.all_feats),1);

valid = valid_old & valid_new;

fprintf('Common valid windows: %d\n', sum(valid));

Xold = old.all_feats(:,valid);
Xnew = new.all_feats(:,valid);

v1 = Xold(:);
v2 = Xnew(:);

C = corrcoef(v1,v2);
r = C(1,2);

fprintf('Global correlation = %.8f\n', r);

mae = mean(abs(v1-v2));

fprintf('Global MAE = %.8e\n', mae);

n = size(Xold,2);

corr_win = zeros(n,1);
mae_win  = zeros(n,1);

for w = 1:n
    C = corrcoef(Xold(:,w), Xnew(:,w));
    corr_win(w) = C(1,2);

    mae_win(w) = mean(abs(Xold(:,w)-Xnew(:,w)));
end

fprintf('Mean correlation = %.6f\n', mean(corr_win));
fprintf('Min correlation  = %.6f\n', min(corr_win));

fprintf('Mean MAE = %.6e\n', mean(mae_win));
fprintf('Max MAE  = %.6e\n', max(mae_win));