# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LazyPredict

# %%
import sys
sys.path.append('..')


import typing as t

import optuna
import lazypredict
from xgboost import XGBClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import polars as pl
from nltk.metrics import windowdiff
import numpy as np
import sklearn

from src.utils import get_segmentation

# %%
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# %%
raw_df = pl.read_csv('../data/400-415-with-target.csv')
en_labse_df = pl.read_csv('../data/en_labse.csv')
ru_labse_df = pl.read_csv('../data/ru_labse.csv')

# %%
non_embeddings_columns = ['ru_sentence', 'en_sentence', 'episode', 'is_first_sentence_in_topic', 'target']
embedding_columns = lambda df: [col for col in df.columns if col not in non_embeddings_columns]


# %%
def split_train_test(df: pl.DataFrame) -> t.Tuple[pl.DataFrame, pl.DataFrame]:
    train_episodes = [400, 401, 402, 403, 404, 405, 406, 407, 408, 409]
    test_episodes = [410, 411, 412, 413, 414, 415]
    
    train = df.filter(pl.col('episode').is_in(train_episodes))
    test = df.filter(pl.col('episode').is_in(test_episodes))
    
    return train, test


# %%
data, target = ru_labse_df[embedding_columns(ru_labse_df)], ru_labse_df['target']

train, test = split_train_test(ru_labse_df)

train_x = train[embedding_columns(ru_labse_df)]
test_x = test[embedding_columns(ru_labse_df)]

train_y = train['target']
test_y = test['target']

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(train_x.to_pandas(), test_x.to_pandas(), train_y.to_pandas(), test_y.to_pandas())

print(models)
