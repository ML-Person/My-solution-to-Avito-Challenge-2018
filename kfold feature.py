import numpy as np
import gc

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def RMSE(L,L1):
    return np.sqrt(mean_squared_error(L,L1))

folds = KFold(n_splits=5, shuffle=True, random_state=44)
oof_preds = np.zeros(X_train.shape[0])
sub_preds = np.zeros(X_test.shape[0])


for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train)):
    dtrain = lgb.Dataset(X_train[train_idx], label=y_train.iloc[train_idx],
                         feature_name=list(feature_names),
                         categorical_feature=categorical)
    dvalid = lgb.Dataset(X_train[valid_idx], label=y_train.iloc[valid_idx],
                         feature_name=list(feature_names),
                         categorical_feature=categorical)

    model = lgb.train(params, dtrain,
                      valid_sets=[dtrain, dvalid],
                      valid_names=['train', 'valid'],
                      num_boost_round=rounds,
                      early_stopping_rounds=early_stop_rounds,
                      verbose_eval=100)

    oof_preds[valid_idx] = model.predict(X_train[valid_idx])
    sub_preds += model.predict(X_test) / folds.n_splits
    print('Fold %2d rmse : %.6f' % (n_fold + 1, RMSE(y_train.iloc[valid_idx], oof_preds[valid_idx])))
    del dtrain, dvalid
    gc.collect()

print('Full RMSE score %.6f' % RMSE(y_train, oof_preds))