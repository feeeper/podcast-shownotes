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
# ```
# +++
# title = "Text Segmentation: Supervised approach"
# date = 2023-10-08
# slug = "Text-Segmentation-Supervised-Approach"
# draft = false
# type = "post"
# +++
# ```

# %% [markdown]
# статья про сегментацию при помощи supervised ml: первое предложение топика vs "внутреннее" предложение топика
#
# 1. интро: пробовал всякие разные варианты, а что если просто взять логрег и на эмбеддингах обучить бинарный классификатор?
# 2. делаем разные эмбединги
# 3. проверяем классификатор
# 4. результат 1
# 5. а может xgb?
# 6. результат 2
# 7. выборка несбалансированная - давайте добавим сгенерированные примеры
# 8. результат 3
# 9. итого

# %% [markdown]
# # Text Segmentation: Supervised approach

# %% [markdown]
# Here is probably the last article about text segmentation. [Previously]() I've explored three approaches: GraphSeg, TextTiling, and an improved TextTiling with an embeddings-based similarity metric. The results were only okay — not great. Additionally, I mentioned a supervised approach: building a classifier that could determine if a sentence is a boundary sentence or not. The text and code below detail my attempt to construct a binary classifier for this task.
#
# > **Friendly Alert**: This post will include really huge amount of Python code. Be ready.
#
# This articles contains two large sections:
# 1. **Dataset** section related to dataset that I'll later,
# 2. **Analysis** section will be about testing different classifers and how they perform on this task with my data.

# %% [markdown]
# But firstly I need to import all necessary libraries and modules.

# %%
import sys
sys.path.append('..')

import typing as t
from dataclasses import dataclass
import re
import warnings
warnings.filterwarnings("ignore")
import os

from joblib import Memory

import pandas as pd
import polars as pl
import numpy as np

from stop_words import get_stop_words
from sentence_transformers import SentenceTransformer
import lazypredict
import sklearn
from nltk.metrics import windowdiff

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn import (
    neighbors,
    naive_bayes,
    discriminant_analysis,
    ensemble,
    calibration,
    semi_supervised,
)

from src.utils import get_segmentation

memory = Memory('../.cache')

# %% [markdown]
# ## Dataset

# %% [markdown]
# I'll use the same dataset I've used earlier: an automatic transcript of some DevZen podcast episodes. The same dataset I used in the previous article in this series.

# %%
csv_path = '../data/400-415-with-target.csv'
df = pl.read_csv(csv_path)
df.head()


# %% [markdown]
# The only valuable columns here are:
# 1. `episode` &mdash; episode number,
# 2. `ru_sentence` &mdash; segment text in Russion (original),
# 3. `en_sentence` &mdash; segment text in English (machine translated `ru_sentence`),
# 4. `target` &mdash; if the segment is boundary sentcence.

# %% [markdown]
# ### Build embeddings

# %% [markdown]
# > A friendly reminder: an embedding is a vector representation of a text. Various methods exist for building embeddings, with **FastText** and **Word2Vec** being among the most well-known and classic. A good embedding algorithm should yield similar embeddings for similar texts.
#
# Just as an experiment I'll use two model for embeddings from HuggingFace:

# %%
@dataclass
class Models:
    paraphrase = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    labse = 'sentence-transformers/labse'


# %% [markdown]
# `memory.cache` is a great way to cache function results. It could be really helpful if you have to do long-running calculation. For instance, building embeddings for a bunch of texts.

# %%
@memory.cache(verbose=False)
def get_embeddings(sentences: list[str], model_name: str) -> list[float]:
    model = SentenceTransformer(model_name)
    return model.encode(list(sentences))


# %%
ru_paraphrase_embeddings = get_embeddings(df['ru_sentence'], Models.paraphrase)
en_paraphrase_embeddings = get_embeddings(df['en_sentence'], Models.paraphrase)

ru_labse_embeddings = get_embeddings(df['ru_sentence'], Models.labse)
en_labse_embeddings = get_embeddings(df['en_sentence'], Models.labse)

# %%
print(f'{ru_paraphrase_embeddings.shape=}')
print(f'{en_paraphrase_embeddings.shape=}')

print(f'{ru_labse_embeddings.shape=}')
print(f'{en_labse_embeddings.shape=}')

# %% [markdown]
# `paraphrase-multilingual-MiniLM-L12-v2` provides vectors with 384 dimensions and `labse` &mdash; 768.

# %%
embeddings = {
    'russian': {
        'paraphrase': ru_paraphrase_embeddings,
        'labse': ru_labse_embeddings,
    },
    'english': {
        'paraphrase': en_paraphrase_embeddings,
        'labse': en_labse_embeddings,
    }
}


# %% [markdown]
# ### Build the dataset

# %% [markdown]
# Below you can see a short function to build a dataset from embeddings. For binary classification I need only embeddings and targets for each of them. I'll add `ru_sentence`, `en_sentence`, and `episode` columns only for further research.

# %%
def build_dataset(
    df: t.Union[pd.DataFrame, pl.DataFrame],
    embeddings: np.ndarray,
    prefix: str = '',
    columns_to_add: list[str] = ['ru_sentence', 'en_sentence', 'episode', 'target']
) -> pl.DataFrame:
    res = pl.from_dataframe(df[[col for col in columns_to_add if col in df.columns]])
    emb_df = pl.DataFrame(embeddings,  schema=[f'{prefix}{x+1}' for x in range(len(embeddings[0]))])
    res = pl.concat([res, emb_df], how='horizontal')
    return res


# %% [markdown]
# Let's use the function and see what I get. The only reason why I printed only four first values from embeddings is that the whole table wouldn't fit into any screen.

# %%
ru_labse_ds = build_dataset(df, ru_labse_embeddings)
ru_paraphrase_ds = build_dataset(df, ru_paraphrase_embeddings)

en_labse_ds = build_dataset(df, en_labse_embeddings)
en_paraphrase_ds = build_dataset(df, en_paraphrase_embeddings)

with pl.Config(fmt_str_lengths=100, tbl_width_chars=120):
    print(ru_labse_ds[['ru_sentence', 'en_sentence', 'episode', 'target'] + [f'{x}' for x in range(1, 5)]].head())


# %% [markdown]
# I use the same function to split dataset into train and test parts as I used earlier. All sentences related to episodes from 400 to 409 included were gone to the train part, others &mdash; to the test part.

# %%
def split_train_test(df: pl.DataFrame) -> t.Tuple[pl.DataFrame, pl.DataFrame]:
    # hardcoded episode numbers to split into train and test parts
    train_episodes = [400, 401, 402, 403, 404, 405, 406, 407, 408, 409]
    test_episodes = [410, 411, 412, 413, 414, 415]
    
    train = df.filter(pl.col('episode').is_in(train_episodes))
    test = df.filter(pl.col('episode').is_in(test_episodes))
    
    return train, test


# %% [markdown]
# To measure a classifier's performace I `windowdiff` metric that I already used in the article about unsupervised approaches.

# %%
def window_diff(gt: str, pred: str, boundary='|') -> float:
    k = int(round(len(gt) / (gt.count(boundary) * 2.)))
    return windowdiff(gt, pred, k, boundary=boundary)


# %% [markdown]
# ### [Maybe should be deleted] One line EDA

# %% [markdown]
# I have to admit that dataset is dramatically disbalanced:

# %%
df['target'].value_counts().with_columns(pl.col('count').apply(lambda x: f'{x/df.shape[0] * 100:.2f}%'))


# %% [markdown]
# As you can see only 0.67% sentences in the dataset are boundary sentences. So we keep it in mind for future work.

# %% [markdown]
# ## Analysis

# %% [markdown]
# ### Train and score classifiers

# %% [markdown]
# I'm not ready to spend hours to check all algorithms in sklearn so I'll write a simple function that train and test passed classifer. After that I'll get the best perfroming classifiers.

# %%
def data_columns(data_frame: pl.DataFrame, exclude_columns: list[str]) -> list[str]:
    return [col for col in data_frame.columns if col not in exclude_columns]


# %% [markdown]
# The `clore_clf` function gets a classifer and dataframe, fits the classifer, and collects metrics for it. In case if the classifer could return probability than the function tries different thresholds. As a result the function returns an array with metrics for each threshold.

# %%
def score_clf(
    clf: t.Any,
    df: pl.DataFrame,
    verbose: bool = False
) -> None:
    train, test = split_train_test(df)
    cols = data_columns(df, ['ru_sentence', 'en_sentence', 'episode', 'target'])
    
    clf.fit(
        train[cols],
        train['target'],
    )
    
    if hasattr(clf, 'predict_proba'):
        predictions = clf.predict_proba(test[cols])
        test = test.hstack([pl.Series('prediction', predictions[:, 1])])
    else:
        predictions = clf.predict(test[cols])
        test = test.hstack([pl.Series('prediction', predictions)])

    windows_diffs = []

    metrics = []
    for g in sorted(test[['episode', 'target', 'prediction']].groupby('episode')):
        ground_truth_seg = ''.join([str(x) for x in g[1]['target']])
        if hasattr(clf, 'predict_proba'):
            best_ts = None
            min_wd = 1
            topics_count = 0
            # score classifier for different thresholds
            for threshold in [x/10 for x in range(1, 10)]:
                predicred_seg = ''.join([str(int(x > threshold)) for x in g[1]['prediction']])
                wd = window_diff(ground_truth_seg, predicred_seg, boundary='1')
                min_wd = min(min_wd, wd)
                best_ts = threshold if min_wd == wd else best_ts
                topics_count = predicred_seg.count('1') + 1 if min_wd == wd else topics_count
                windows_diffs.append(wd)
                metrics.append({
                    'clf': clf.__class__.__name__,
                    'episode': g[0],
                    'threshold': threshold,
                    'window_diff': wd,
                    'predicted_topics_count': predicred_seg.count('1') + 1,
                    'ground_truth_topics_count': ground_truth_seg.count("1") + 1
                })
            if verbose: print(f'episode={g[0]}\t{best_ts=}\tbest windowdiff={min_wd}\ttopics count for best windowdiff={topics_count}\treal topic count={ground_truth_seg.count("1")+1}')
        else:
            predicred_seg = ''.join([str(x) for x in g[1]['prediction']])
            wd = window_diff(ground_truth_seg, predicred_seg, boundary='1')
            windows_diffs.append(wd)
            metrics.append({
                'clf': clf.__class__.__name__,
                'episode': g[0],
                'threshold': None,
                'window_diff': wd,
                'predicted_topics_count': predicred_seg.count('1') + 1,
                'ground_truth_topics_count': ground_truth_seg.count("1") + 1
            })
            if verbose: print(f'episode={g[0]}\twindowdiff={wd:.4f}\ttopics count={predicred_seg.count("1")+1}\treal topic count={ground_truth_seg.count("1")+1}')
    if verbose: print(f'avg windowdiff through episodes = {np.mean(windows_diffs):.4f}')
    return metrics


# %% [markdown]
# Below the code that helped me to train 25 different classifers on LaBSE-based dataset. Classifers are from default LogisticRegression to XGB and LGB boostings.

# %%
random_state = 42

nc_clf = neighbors.NearestCentroid()
gauss_clf = sklearn.naive_bayes.GaussianNB()
bernoulli_clf = sklearn.naive_bayes.BernoulliNB()
logreg_clf = sklearn.linear_model.LogisticRegression()
lindis_clf = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
linear_svc_clf = sklearn.svm.LinearSVC(random_state=random_state)
knn_clf = sklearn.neighbors.KNeighborsClassifier()
perc_clf = sklearn.linear_model.Perceptron(random_state=random_state)
ada_clf = sklearn.ensemble.AdaBoostClassifier(random_state=random_state)
sgd_clf = sklearn.linear_model.SGDClassifier(random_state=random_state)
ext_tree_clf = sklearn.tree.ExtraTreeClassifier(random_state=random_state)
dec_tree_clf = sklearn.tree.DecisionTreeClassifier(random_state=random_state)
pass_agg_clf = sklearn.linear_model.PassiveAggressiveClassifier(random_state=random_state)
svc_clf = sklearn.svm.SVC(random_state=random_state)
ext_trees_clf = sklearn.ensemble.ExtraTreesClassifier(random_state=random_state)
calib_clf = sklearn.calibration.CalibratedClassifierCV()
label_prop_clf = sklearn.semi_supervised.LabelPropagation()
label_spr_clf = sklearn.semi_supervised.LabelSpreading()
bag_clf = sklearn.ensemble.BaggingClassifier(random_state=random_state)
rnd_forest_clf = sklearn.ensemble.RandomForestClassifier(random_state=random_state)
dummy_clf = sklearn.dummy.DummyClassifier(random_state=random_state)
quad_clf = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
ridge_clf = sklearn.linear_model.RidgeClassifierCV()

lgb_clf = LGBMClassifier(random_state=random_state, verbose=-1)
xgb_clf = XGBClassifier(random_state=random_state)

clfs = [
    nc_clf,
    gauss_clf,
    bernoulli_clf,
    logreg_clf,
    lindis_clf,
    linear_svc_clf,
    knn_clf,
    perc_clf,
    ada_clf,
    sgd_clf,
    ext_tree_clf,
    dec_tree_clf,
    pass_agg_clf,
    svc_clf,
    ext_trees_clf,
    calib_clf,
    label_prop_clf,
    label_spr_clf,
    bag_clf,
    rnd_forest_clf,
    dummy_clf,
    quad_clf,
    ridge_clf,
    lgb_clf,
    xgb_clf,
]

labse_metrics = []
for clf in clfs:
    try:
        labse_metrics.extend(score_clf(clf,ru_labse_ds))
    except Exception as e:
        print(e)

# %%
metrics_df = pl.DataFrame(labse_metrics)
metrics_df.shape

# %% [markdown]
# There are 1014 rows contains the dataframe with result metrics.
#
# It would be hard to dive into all 1014 rows so lets take a look at the results for 410 episode &mdash; the first test episode:

# %%
episode_metrics = metrics_df.filter(pl.col('episode') == 410).sort(['episode', 'clf', 'threshold'])
print(episode_metrics)

# %% [markdown]
# Interesting but not informative. Too noisy results.
#
# The worst classifers couldn't find any boundary sentence. It means that `predicted_topics_count` for these classifers will be equal to 1:

# %%
worst_classifers = metrics_df.group_by([
    pl.col('clf')
]).agg(
    pl.col('predicted_topics_count').max().alias('max_predicted_topics_count')
).filter(pl.col('max_predicted_topics_count') == 1).sort('clf')

print(set(worst_classifers['clf']))

# %% [markdown]
# So I'll get rid off all classifers from future research which can't even find single boundary sentence: 
# - DummyClassifier,
# - RidgeClassifierCV,
# - SGDClassifier,
# - QuadraticDiscriminantAnalysis.
#
# The reason is that all provided episodes have at least 7 topics.

# %%
metrics_df = metrics_df.filter(~pl.col('clf').is_in(set(worst_classifers['clf'])))
metrics_df

# %% [markdown]
# Even if I remove worst classifers I have too many data. So the next step is to try to find the best classifers. The best classifer is the one that performs best (in terms of `windowdiff` score) for at least one test episode:

# %%
metrics_df = metrics_df.with_columns(
    pl.struct(
        ['predicted_topics_count', 'ground_truth_topics_count']
    ).map_elements(
        lambda x: x['predicted_topics_count'] / x['ground_truth_topics_count']
    ).alias('ratio')
)

best_performing_clfs = metrics_df.group_by(
    ['episode']
).agg([
    pl.col('window_diff').min(),
    pl.col('threshold').min(),
]).join(
    metrics_df, on='window_diff', how='left'
).sort(['episode', 'clf', 'threshold'])

best_performing_clfs

# %% [markdown]
# A table above represents the best performing classifer for each episode. For some episodes there are more than one best performing classifier &mdash; that's fine.
#
# Also I added `ratio` column as a ratio found topics count to the real topics count. This field will be used later.
#
# The list below contains the best performed classifers. Let's deep dive into those models.

# %%
best_performing_clfs_names = sorted(list(best_performing_clfs['clf'].unique()))
best_performing_clfs_names


# %% [markdown]
# ### Best performing classifiers analysis

# %% [markdown]
# The best of the best classifers:
# 1. should have small `windowdiff` score,
# 2. should find topics for each test episode,
# 3. should have ratio ~1.
#
# The functions below will show me these classifiers:

# %%
def get_clf_best_metrics(clf_name: str) -> pl.DataFrame:
    classifier_metrics = metrics_df.filter(pl.col('clf') == clf_name)

    classifier_best_scores = classifier_metrics.groupby([
        pl.col('episode')
    ]).agg([
        pl.col('window_diff').min().alias('best_win_diff')
    ]).sort([
        pl.col('episode')
    ]).join(
        classifier_metrics,
        left_on='best_win_diff',
        right_on='window_diff'
    ).with_columns(
        pl.col('best_win_diff').alias('window_diff')
    )

    return classifier_best_scores[['episode', 'window_diff', 'threshold', 'predicted_topics_count', 'ground_truth_topics_count', 'ratio']]


def get_results_for_best_thresholds(
    best_classifier_metrics: pl.DataFrame,
    all_classifier_metrics: pl.DataFrame
) -> list[pl.DataFrame]:
    gr = best_classifier_metrics.groupby(
        pl.col('threshold')
    ).agg(
        pl.col('episode').count().alias('count')
    )
    gr = gr.filter(
        pl.col('count') == gr['count'].max()
    )

    for t in list(gr['threshold']):
        yield all_classifier_metrics.filter(pl.col('threshold') == t)[['episode', 'window_diff', 'threshold', 'predicted_topics_count', 'ground_truth_topics_count', 'ratio']]
    return


# %%
test_episodes = list(split_train_test(ru_labse_ds)[1]['episode'].unique())
best_results = []
for wc in best_performing_clfs_names:
    print(f'{wc:=^100}')
    clf_best_metrics = get_clf_best_metrics(wc)
    
    print()
    for d in get_results_for_best_thresholds(clf_best_metrics, metrics_df.filter(pl.col('clf') == wc)):
        tmp = d if len(d) > 0 else clf_best_metrics
        print(tmp)
        print(f'avg window diff: {tmp["window_diff"].mean():.2f}')
        print(f'avg ratio: {tmp["ratio"].mean():.2f}')
        print(f'episodes found: {len(tmp)} out of {len(test_episodes)}')
        best_results.append({
            'clf': wc,
            'threshold': tmp['threshold'][0],
            'avg_win_diff': tmp["window_diff"].mean(),
            'avg_ratio': tmp["ratio"].mean(),
            'episodes_with_topics': len(tmp),
            'total_test_episodes': len(test_episodes),
        })
        print()
    print()

# %% [markdown]
# Each classifer has at least one threshold value that somehow works. Lets see which of these classifers and thresholds work better than others:

# %%
with pl.Config(fmt_str_lengths=100, tbl_width_chars=120):
    print(pl.DataFrame(best_results).sort([pl.col('avg_win_diff')]))

# %% [markdown]
# JFYI: null `threshold` means that classifer doesn't provide `predict_proba` method.
#
# There are three classifiers that showed `avg_ratio` near 1 but LinearDiscriminantAnalysis performs pretty well with thresholds 0.2 and 0.7:
# 1. LinearDiscriminantAnalysis,
# 2. PassiveAggressiveClassifier,
# 3. LogisticRegression.

# %% [markdown]
# Now I want to see two things:
# 1. detailed metrics,
# 2. sentences that were predicted as boundary sentence.

# %%
final_clfs = {
    'LinearDiscriminantAnalysis': [(lindis_clf, 0.2), (lindis_clf, 0.7)],
    'PassiveAggressiveClassifier': [(pass_agg_clf, None)],
    'LogisticRegression': [(logreg_clf, 0.1)]
}


# %%
def get_metrics_for_clf_and_threshold(clf: str, threshold: t.Optional[float]) -> pl.DataFrame:
    clf_metrics = metrics_df.filter(pl.col('clf') == clf)
    if threshold is not None:
        clf_metrics = clf_metrics.filter(pl.col('threshold') == threshold)
    return clf_metrics


# %%
_, test_labse_df = split_train_test(ru_labse_ds)
data = test_labse_df[data_columns(test_labse_df, ['ru_sentence', 'en_sentence', 'episode', 'target'])]

for clf_name, vals in final_clfs.items():
    for val in vals:
        with pl.Config(fmt_str_lengths=200, tbl_width_chars=200, tbl_rows=-1):
            print(f'{clf_name:=^130}')
            print(get_metrics_for_clf_and_threshold(clf_name, val[1]))
            
            if hasattr(val[0], 'predict_proba'):
                preds = val[0].predict_proba(data)
                result_df = test_labse_df.hstack([pl.Series('prediction', [int(x > val[1]) for x in preds[:,1]])])
            else:
                result_df = test_labse_df.hstack([pl.Series('prediction', val[0].predict(data))])
                
            for ep in test_episodes:
                episode_result_df = result_df.filter(((pl.col('prediction') == 1) | (pl.col('target') == 1)) & (pl.col('episode')== ep))[['episode', 'ru_sentence', 'en_sentence', 'target', 'prediction']]
                title = f'{clf_name}.{ep}'
                print(f'{title:-^130}')
                print(episode_result_df)
