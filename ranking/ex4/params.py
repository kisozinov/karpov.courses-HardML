from hyperopt import hp
import numpy as np

PATH_TO_MODEL = 'model2.bin'

PARAM_GRID = {
    'ndcg_top_k': 10,
    'max_depth': 1 + hp.randint('max_depth', 15),
    'n_estimators': 100,
    'lr': hp.loguniform('lr', np.log(0.1), np.log(0.5)),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 0.9),
    'subsample': hp.uniform('subsample', 0.5, 0.9),
    'min_samples_leaf': 5 + hp.randint('min_samples_leaf', 100),
}

PARAMS = {'colsample_bytree': 0.6, 'lr': 0.4, 'max_depth': 8, 'min_samples_leaf': 85, 'subsample': 0.8}
