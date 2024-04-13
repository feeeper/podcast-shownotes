# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Optimize XGBoost with Optuna

# %%
import sys
sys.path.append('..')

import typing as t

import optuna
from xgboost import XGBClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import polars as pl
from nltk.metrics import windowdiff
import numpy as np
import sklearn

from src.utils import get_segmentation

# %%
raw_df = pl.read_csv('../data/400-415-with-target.csv')
en_labse_df = pl.read_csv('../data/en_labse.csv')
ru_labse_df = pl.read_csv('../data/ru_labse.csv')

# %%
non_embeddings_columns = ['ru_sentence', 'en_sentence', 'episode', 'is_first_sentence_in_topic', 'target']
embedding_columns = lambda df: [col for col in df.columns if col not in non_embeddings_columns]


# %%
def score(gt: str, pred: str) -> float:
    boundary = '|'
    k = int(round(len(gt) / (gt.count(boundary) * 2.)))
    return windowdiff(gt, pred, k, boundary=boundary)


# %%
def split_train_test(df: pl.DataFrame) -> t.Tuple[pl.DataFrame, pl.DataFrame]:
    train_episodes = [400, 401, 402, 403, 404, 405, 406, 407, 408, 409]
    test_episodes = [410, 411, 412, 413, 414, 415]
    
    train = df.filter(pl.col('episode').is_in(train_episodes))
    test = df.filter(pl.col('episode').is_in(test_episodes))
    
    return train, test


# %%
def objective(trial: optuna.Trial) -> float:
    data, target = ru_labse_df[embedding_columns(ru_labse_df)], ru_labse_df['target']
    # train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)
    train, test = split_train_test(ru_labse_df)
    train_x = train[embedding_columns(ru_labse_df)]
    test_x = test[embedding_columns(ru_labse_df)]
    
    train_y = train['target']
    test_y = test['target']
    
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)

    param = {
        "random_state": 42,
        # "silent": 1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
        param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
    bst = xgb.train(param, dtrain, evals=[(dtest, "validation")], callbacks=[pruning_callback])
    preds = bst.predict(dtest)
    pred_labels = np.rint(preds)

    roc_auc = sklearn.metrics.roc_auc_score(test_y, pred_labels)
    return roc_auc

#     test = test.hstack([pl.Series('prediction', preds)])
    
#     gt = get_segmentation(test, 'target')
#     pred = get_segmentation(test, 'prediction')
    
#     windiffs = []
#     for ep in gt:        
#         wdiff = score(gt[ep], pred[ep])
#         windiffs.append(wdiff)

#     return sum(windiffs)/len(windiffs)


# %%
study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    direction='maximize'
)
study.optimize(objective, n_trials=5)
print(study.best_trial)

# %%
train, test = split_train_test(ru_labse_df)
train_x = train[embedding_columns(ru_labse_df)]
test_x = test[embedding_columns(ru_labse_df)]

train_y = train['target']
test_y = test['target']

# xgbc = XGBClassifier(**study.best_params)
xgbc = XGBClassifier(random_state=42)
xgbc.fit(train_x, train_y)

# %%
pl.DataFrame(xgbc.predict_proba(test_x)).with_columns([
    pl.col('column_1').map_elements(lambda x: int(x > i/10)).alias(f'threshold_{i/10}') for i in range(3, 10)
]).sum()

# %%
en_labse_df.filter(pl.col('episode').is_in([410, 411, 412, 413, 414, 415])).hstack([pl.Series('pred', xgbc.predict(test_x))])[[
    'episode',
    'ru_sentence',
    'en_sentence',
    'target',
    'pred',
]].filter((pl.col('pred') == 1))

# %%
optuna.Trial.suggest_int(
