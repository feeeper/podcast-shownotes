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

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import optuna

from tqdm.notebook import tqdm

from stop_words import get_stop_words
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

from src.utils import get_segmentation

# %%
memory = Memory('../.cache')

# %%
csv_path = '../data/400-415-with-target.csv'
df = pl.read_csv(csv_path)
df.head()


# %% [markdown]
# # Build embeddings

# %%
@dataclass
class Models:
    paraphrase = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    labse = 'sentence-transformers/labse'


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
# # Fill target `is_first_sentence_in_topic`

# %%
# df['ground_truth_next'] = df.to_pandas().groupby('episode')['ground_truth'].shift(-1).fillna(1_000).astype(int)
# df[(df['episode']==413) & (df['ground_truth_next'] != df['ground_truth'])]

df2 = df.clone().to_pandas()
df2['ground_truth_next'] =  df2.groupby('episode')['ground_truth'].shift(-1).fillna(1_000).astype(int)
df2[(df2['episode']==413) & (df2['ground_truth_next'] != df2['ground_truth'])]


# %%
def is_first_sentence_in_topic(s: pd.Series) -> int:
    return 0 if s['ground_truth'] == s['ground_truth_next'] else 1

df2['is_first_sentence_in_topic'] = df2.apply(is_first_sentence_in_topic, axis='columns')

# %%
df2[df2['is_first_sentence_in_topic'] == 1][['ru_sentence', 'en_sentence', 'ground_truth', 'is_first_sentence_in_topic']]

# %%
df2['is_first_sentence_in_topic'].value_counts(normalize=True)


# %% [markdown]
# # Build Dataset with target

# %%
def build_dataset(
    df: t.Union[pd.DataFrame, pl.DataFrame],
    embeddings: np.ndarray,
    prefix: str = '',
    columns_to_add: list[str] = ['ru_sentence', 'en_sentence', 'episode', 'is_first_sentence_in_topic', 'target']
) -> pl.DataFrame:
    res = pl.from_dataframe(df[[col for col in columns_to_add if col in df.columns]])
    emb_df = pl.DataFrame(embeddings,  schema=[f'{prefix}{x+1}' for x in range(len(embeddings[0]))])
    res = pl.concat([res, emb_df], how='horizontal')
    return res


# %%
ru_labse_df = build_dataset(df, ru_labse_embeddings, prefix='ru_labse_')
ru_labse_df.head()

# %%
ru_paraphrase_df = build_dataset(df, ru_paraphrase_embeddings, prefix='ru_paraphrase_')
ru_paraphrase_df.head()

# %%
en_labse_df = build_dataset(df, en_labse_embeddings, prefix='en_labse_')
en_labse_df.head()

# %%
en_paraphrase_df = build_dataset(df, en_paraphrase_embeddings, prefix='en_paraphrase_')
en_paraphrase_df.head()

# %% [markdown]
# # Logistic regression

# %%
from nltk.metrics import windowdiff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report
)

# %%
non_embeddings_columns = ['ru_sentence', 'en_sentence', 'episode', 'is_first_sentence_in_topic']
embedding_columns = lambda df: [col for col in df.columns if col not in non_embeddings_columns]

non_embeddings_columns = ['ru_sentence', 'en_sentence', 'episode', 'is_first_sentence_in_topic', 'target']
embedding_columns = lambda df: [col for col in df.columns if col not in non_embeddings_columns]


# %%
def split_train_test(df: pl.DataFrame) -> t.Tuple[pl.DataFrame, pl.DataFrame]:
    train_episodes = [400, 401, 402, 403, 404, 405, 406, 407, 408, 409]
    test_episodes = [410, 411, 412, 413, 414, 415]
    
    train = df.filter(pl.col('episode').is_in(train_episodes))
    test = df.filter(pl.col('episode').is_in(test_episodes))
    
    return train, test


# %% [markdown]
# ## Russian

# %% [markdown]
# ### Labse

# %%
def split_train_predict(
    df: pl.DataFrame,
    target_col: str = 'is_first_sentence_in_topic',
    lang: str = 'ru',
    verbose: int = 0,
) -> LogisticRegression:
    train, test = split_train_test(df)
    if verbose > 1:
        print(f'{train.shape=}')
        print(f'{test.shape=}')
        print(test[target_col].value_counts().with_columns(pl.col('counts').map_elements(lambda x: x/len(test))))
    
    embedding_cols = embedding_columns(df)
    lr = LogisticRegression(random_state=42, class_weight='balanced')
    lr.fit(train[embedding_cols], train[target_col])
    
    preds = lr.predict_proba(test[embedding_cols])
    roc_auc = roc_auc_score(test[target_col], preds[:, 1])
    if verbose:
        print(f'{roc_auc=}')
    test = test.hstack([pl.Series('log_prediction', preds[:, 1])])
    
    preds = lr.predict(test[embedding_cols])
    
    metrics = []
    ground_truth_seg = get_segmentation(test, 'target')
    for threshold in [x/10 for x in range(3, 10)]:
        test = test.with_columns(pl.col('log_prediction').map_elements(lambda x: int(x > threshold)).alias('prediction'))
        predicted_seg = get_segmentation(test, 'prediction')

        for ep in ground_truth_seg.keys():
            try:
                ep_ground_truth_seg = ground_truth_seg[ep]
                ep_predicted_seg = predicted_seg[ep]

                # default k value for windowdiff
                k = int(round(len(ep_ground_truth_seg) / (ep_ground_truth_seg.count('|') * 2.)))
                if verbose > 1:
                    print(f'{ep=}\t{len(ep_ground_truth_seg)=}\t{len(ep_predicted_seg)=}')
                win_diff = windowdiff(ep_ground_truth_seg, ep_predicted_seg, k, boundary='|')
                if verbose:
                    print(f'{ep=}\t{win_diff=}')

                metrics.append({
                    'algorithm': 'logreg',
                    'embeddings': embedding_cols[0][:-2],
                    'threshold': threshold,
                    'lang': lang,
                    'k': k,
                    'win_diff': win_diff,
                    'episode': ep,
                    'topics_count': ep_predicted_seg.count('|') + 1,
                    'ground_truth_topics_count': ep_ground_truth_seg.count('|') + 1
                })
            except Exception as e:
                print(e)
    
    clf_rep = classification_report(test[target_col], preds)
    if verbose:
        print(classification_report(test[target_col], preds))

    correlation = test.select(pl.corr(target_col, 'log_prediction'))[0,0]
    if verbose:
        print(f'{correlation=}')
        
    return metrics


# %%
log_rsklearnru_labse_metrics = split_train_predict(ru_labse_df, target_col='target', lang='ru', verbose=0)
pl.DataFrame(log_reg_ru_labse_metrics)

# %% [markdown]
# ### Paraphrase

# %%
log_reg_ru_paraphrase_metrics = split_train_predict(ru_paraphrase_df, target_col='target', lang='ru', verbose=0)
pl.DataFrame(log_reg_ru_paraphrase_metrics)

# %% [markdown]
# ## English

# %% [markdown]
# ### Labse

# %%
log_reg_en_labse_metrics = split_train_predict(en_labse_df, target_col='target', lang='en', verbose=0)
pl.DataFrame(log_reg_en_labse_metrics)

# %% [markdown]
# ### Paraphrase

# %%
log_reg_en_paraphrase_metrics = split_train_predict(en_paraphrase_df, target_col='target', lang='en', verbose=0)
pl.DataFrame(log_reg_en_paraphrase_metrics)

# %%
log_reg_base_metrics = log_reg_ru_labse_metrics + log_reg_ru_paraphrase_metrics + log_reg_en_labse_metrics + log_reg_en_paraphrase_metrics
log_reg_base_metrics_df = pl.DataFrame(log_reg_base_metrics).group_by(['algorithm', 'lang', 'embeddings', 'threshold']).agg(pl.col('win_diff').mean())
log_reg_base_metrics_df = log_reg_base_metrics_df.with_columns(pl.col('algorithm').map_elements(lambda x: f'logreg-base'))

with pl.Config(fmt_str_lengths=200, tbl_rows=30):
    print(log_reg_base_metrics_df.filter(pl.col('lang') == 'ru').sort('win_diff')[:5])
    print(log_reg_base_metrics_df.filter(pl.col('lang') == 'en').sort('win_diff')[:5])

# %%
with pl.Config(fmt_str_lengths=200, tbl_rows=30):
    print(pl.DataFrame(log_reg_en_labse_metrics).filter(pl.col('threshold') == 0.9)[['episode', 'topics_count', 'ground_truth_topics_count']])
    print(pl.DataFrame(log_reg_ru_labse_metrics).filter(pl.col('threshold') == 0.9)[['episode', 'topics_count', 'ground_truth_topics_count']])


# %% [markdown]
# ## Rolling features: sentence before and after target

# %%
def get_plus_one_and_minus_one_rows(df: pl.DataFrame) -> pl.DataFrame:
    embedding_cols = embedding_columns(df)
    res = df.with_columns(
        [pl.col(x).shift(1).over('episode').alias(f'{x}_minus_1') for x in embedding_cols] +
        [pl.col(x).shift(-1).over('episode').alias(f'{x}_plus_1') for x in embedding_cols]
    )
    res = res.fill_null(0)
    return res


# %%
log_reg_rolling_ru_labse_metrics = split_train_predict(get_plus_one_and_minus_one_rows(ru_labse_df), target_col='target', lang='ru', verbose=0)
pl.DataFrame(log_reg_rolling_ru_labse_metrics)

# %%
log_reg_rolling_ru_paraphrase_metrics = split_train_predict(get_plus_one_and_minus_one_rows(ru_paraphrase_df), target_col='target', lang='ru', verbose=0)
pl.DataFrame(log_reg_rolling_ru_paraphrase_metrics)

# %%
log_reg_rolling_en_labse_metrics = split_train_predict(get_plus_one_and_minus_one_rows(en_labse_df), target_col='target', lang='en', verbose=0)
pl.DataFrame(log_reg_rolling_en_labse_metrics)

# %%
log_reg_rolling_en_paraphrase_metrics = split_train_predict(get_plus_one_and_minus_one_rows(en_paraphrase_df), target_col='target', lang='en', verbose=0)
pl.DataFrame(log_reg_rolling_en_labse_metrics)

# %%
log_reg_rolling_metrics = log_reg_rolling_ru_labse_metrics + log_reg_rolling_ru_paraphrase_metrics + log_reg_rolling_en_labse_metrics + log_reg_rolling_en_paraphrase_metrics
log_reg_rolling_metrics_df = pl.DataFrame(log_reg_rolling_metrics).group_by(['algorithm', 'lang', 'embeddings', 'threshold']).agg(pl.col('win_diff').mean())
log_reg_rolling_metrics_df = log_reg_rolling_metrics_df.with_columns(pl.col('algorithm').map_elements(lambda x: f'logreg-rolling'))

with pl.Config(fmt_str_lengths=200, tbl_rows=30):
    print(log_reg_rolling_metrics_df.filter(pl.col('lang') == 'ru').sort('win_diff')[:5])
    print(log_reg_rolling_metrics_df.filter(pl.col('lang') == 'en').sort('win_diff')[:5])

# %%
with pl.Config(fmt_str_lengths=200, tbl_rows=30):
    print(pl.DataFrame(log_reg_rolling_ru_paraphrase_metrics).filter(pl.col('threshold') == 0.9)[['episode', 'topics_count', 'ground_truth_topics_count']])
    print(pl.DataFrame(log_reg_rolling_en_paraphrase_metrics).filter(pl.col('threshold') == 0.9)[['episode', 'topics_count', 'ground_truth_topics_count']])


# %% [markdown]
# # XGBoost

# %%
def split_train_predict_boost(
    df: pl.DataFrame,
    target_col='target',
    lang: str = 'ru',
    verbose: int = 0,
) -> XGBClassifier:
    train, test = split_train_test(df)
    
    if verbose > 1:
        print(f'{train.shape=}')
        print(f'{test.shape=}')
        print(test[target_col].value_counts().with_columns(pl.col('counts').map_elements(lambda x: x/len(test))))
    
    embedding_cols = embedding_columns(df)
    xgb = XGBClassifier(random_state=42, n_estimators=1_000)
    xgb.fit(train[embedding_cols], train[target_col])
    
    preds = xgb.predict_proba(test[embedding_cols])
    roc_auc = roc_auc_score(test[target_col], preds[:, 1])
    if verbose:
        print(f'{roc_auc=}')
        
    test = test.hstack([pl.Series('prediction', preds[:, 1])])
    
    preds = xgb.predict(test[embedding_cols])
    if verbose:
        print(classification_report(test[target_col], preds))
        
    correlation = test.select(pl.corr(target_col, 'prediction'))[0,0]
    if verbose:
        print(f'{correlation=}')

    metrics = []
    ground_truth_seg = get_segmentation(test, target_col)
    predicted_seg = get_segmentation(test, 'prediction')
    for threshold in [x/10 for x in range(3, 10)]:        
        for ep in ground_truth_seg.keys():
            try:
                ep_ground_truth_seg = ground_truth_seg[ep]
                ep_predicted_seg = predicted_seg[ep]

                # default k value for windowdiff
                k = int(round(len(ep_ground_truth_seg) / (ep_ground_truth_seg.count('|') * 2.)))
                if verbose > 1:
                    print(f'{ep=}\t{len(ep_ground_truth_seg)=}\t{len(ep_predicted_seg)=}')
                win_diff = windowdiff(ep_ground_truth_seg, ep_predicted_seg, k, boundary='|')
                if verbose:
                    print(f'{ep=}\t{win_diff=}')

                metrics.append({
                    'algorithm': 'xgboost',
                    'embeddings': embedding_cols[0][:-2],
                    'threshold': threshold,
                    'lang': lang,
                    'k': k,
                    'win_diff': win_diff,
                    'episode': ep,
                    'topics_count': ep_predicted_seg.count('|') + 1,
                    'ground_truth_topics_count': ep_ground_truth_seg.count('|') + 1
                })
            except Exception as e:
                print(e)

    return metrics


# %% [markdown]
# ## Russian

# %% [markdown]
# ### Base

# %%
xgb_base_ru_labse_metrics = split_train_predict_boost(ru_labse_df, lang='ru')
pl.DataFrame(xgb_base_ru_labse_metrics)['topics_count'].value_counts()

# %%
with pl.Config(fmt_str_lengths=200, tbl_rows=30):
    print(pl.DataFrame(xgb_base_ru_labse_metrics).filter(pl.col('threshold') == 0.9)[['episode', 'topics_count', 'ground_truth_topics_count']])
    print(pl.DataFrame(xgb_base_ru_labse_metrics).filter(pl.col('threshold') == 0.9)[['episode', 'topics_count', 'ground_truth_topics_count']])

# %%
xgb_base_ru_paraphrase_metrics = split_train_predict_boost(ru_paraphrase_df, lang='ru')
pl.DataFrame(xgb_base_ru_paraphrase_metrics)['topics_count'].value_counts()

# %% [markdown]
# ### Rolling features

# %%
xgb_rolling_ru_labse_metrics = split_train_predict_boost(get_plus_one_and_minus_one_rows(ru_labse_df), lang='ru')
pl.DataFrame(xgb_rolling_ru_labse_metrics)['topics_count'].value_counts()

# %%
xgb_rolling_ru_paraphrase_metrics = split_train_predict_boost(get_plus_one_and_minus_one_rows(ru_paraphrase_df), lang='ru')
pl.DataFrame(xgb_rolling_ru_paraphrase_metrics)['topics_count'].value_counts()

# %% [markdown]
# ## English

# %% [markdown]
# ### Base

# %%
xgb_base_en_labse_metrics = split_train_predict_boost(en_labse_df, lang='en')
pl.DataFrame(xgb_base_en_labse_metrics)['topics_count'].value_counts()

# %%
xgb_base_en_paraphrase_metrics = split_train_predict_boost(en_paraphrase_df, lang='en')
pl.DataFrame(xgb_base_en_paraphrase_metrics)['topics_count'].value_counts()

# %% [markdown]
# ### Rolling features

# %%
xgb_rolling_en_labse_metrics = split_train_predict_boost(get_plus_one_and_minus_one_rows(en_labse_df), lang='en')
pl.DataFrame(xgb_rolling_en_labse_metrics)['topics_count'].value_counts()

# %%
xgb_rolling_en_paraphrase_metrics = split_train_predict_boost(get_plus_one_and_minus_one_rows(en_paraphrase_df), lang='en')
pl.DataFrame(xgb_rolling_en_paraphrase_metrics)['topics_count'].value_counts()

# %% [markdown]
# # NearestCentroid

# %%
from sklearn.neighbors import NearestCentroid
clf = NearestCentroid(metric='cosine')

x, y = split_train_test(ru_labse_df)
clf.fit(x[embedding_columns(x)], x['target'])

p = clf.predict(y[embedding_columns(y)])
print(classification_report(y['target'], p))


# %%
def split_train_predict_nearest_centroid(
    df: pl.DataFrame,
    target_col: str = 'is_first_sentence_in_topic',
    lang: str = 'ru',
    verbose: int = 0,
) -> list:
    train, test = split_train_test(df)
    if verbose > 1:
        print(f'{train.shape=}')
        print(f'{test.shape=}')
        print(test[target_col].value_counts().with_columns(pl.col('counts').map_elements(lambda x: x/len(test))))
    
    embedding_cols = embedding_columns(df)
    lr = NearestCentroid(metric='cosine')
    lr.fit(train[embedding_cols], train[target_col])
    
    # preds = lr.predict_proba(test[embedding_cols])
    # roc_auc = roc_auc_score(test[target_col], preds[:, 1])
    # if verbose:
    #     print(f'{roc_auc=}')
    # test = test.hstack([pl.Series('log_prediction', preds[:, 1])])
    
    preds = lr.predict(test[embedding_cols])
    test = test.hstack([pl.Series('prediction', preds)])
    
    ground_truth_seg = get_segmentation(test, 'target')
    predicted_seg = get_segmentation(test, 'prediction')
    
    metrics = []
    for ep in ground_truth_seg.keys():
        try:
            ep_ground_truth_seg = ground_truth_seg[ep]
            ep_predicted_seg = predicted_seg[ep]

            # default k value for windowdiff
            k = int(round(len(ep_ground_truth_seg) / (ep_ground_truth_seg.count('|') * 2.)))
            if verbose > 1:
                print(f'{ep=}\t{len(ep_ground_truth_seg)=}\t{len(ep_predicted_seg)=}')
            win_diff = windowdiff(ep_ground_truth_seg, ep_predicted_seg, k, boundary='|')
            if verbose:
                print(f'{ep=}\t{win_diff=}')

            metrics.append({
                'algorithm': 'NearestCentroid',
                'embeddings': embedding_cols[0][:-2],
                'lang': lang,
                'k': k,
                'win_diff': win_diff,
                'episode': ep,
                'topics_count': ep_predicted_seg.count('|') + 1,
                'ground_truth_topics_count': ep_ground_truth_seg.count('|') + 1
            })
        except Exception as e:
            print(e)

    clf_rep = classification_report(test[target_col], preds)
    if verbose:
        print(classification_report(test[target_col], preds))

    correlation = test.select(pl.corr(target_col, 'prediction'))[0,0]
    if verbose:
        print(f'{correlation=}')
        
    return metrics


# %%
split_train_predict_nearest_centroid(ru_labse_df, target_col='target', lang='ru', verbose=1)

# %% [markdown]
# # Augmentation with Transformers

# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# %%
device = 'cuda' if torch.cuda.is_available else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('humarin/chatgpt_paraphraser_on_T5_base')
model = AutoModelForSeq2SeqLM.from_pretrained('humarin/chatgpt_paraphraser_on_T5_base').to(device)

print(f'{device=}')


# %%
def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):     
    with torch.no_grad():
        input_ids = tokenizer(
            f'paraphrase: {question}',
            return_tensors='pt', padding='longest',
            max_length=max_length,
            truncation=True,
        ).input_ids.to(device)
        
        outputs = model.generate(
            input_ids,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            max_length=max_length,
            diversity_penalty=diversity_penalty
        )

        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res


# %%
def paraphrase_by_index(i: int, df: pl.DataFrame) -> t.Tuple[str, str, list[str], list[str]]:
    en_res = paraphrase(test[i, 'en_sentence'])
    ru_res = paraphrase(test[i, 'ru_sentence'])
    return df[i, 'ru_sentence'], df[i, 'en_sentence'], ru_res, en_res


# %%
def print_paraphrases(ru_orig: str, en_orig: str, ru_sent: list[str], en_sent: list[str]):
    print(f'{ru_orig = }')
    print('ru_sent:\n' + '\n'.join([f'- "{x}"' for x in ru_sent]), end='\n\n')
    
    print(f'{en_orig = }')
    print('en_sent:\n' + '\n'.join([f'- "{x}"' for x in en_sent]))


# %%
ru_orig, en_orig, ru_sents, en_sents = paraphrase_by_index(195, test)

# %%
print_paraphrases(ru_orig, en_orig, ru_sents, en_sents)

# %% [markdown]
# ## Extend original DataFrame with generated "first sentences"

# %%
fs_df = train.filter(pl.col('is_first_sentence_in_topic') == 1)
with pl.Config(fmt_str_lengths=200, tbl_rows=30):
    print(fs_df[['ru_sentence']])


# %%
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# MODEL_NAME = 'cointegrated/rut5-base-paraphraser'
# model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
# tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
# model.cuda();
# model.eval();

def paraphrase(
    text,
    beams=5,
    grams=4,
    do_sample=False,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
):
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)
    out = model.generate(
        **x,
        encoder_no_repeat_ngram_size=grams,
        num_beams=beams,
        num_beam_groups=5,
        max_length=max_size,
        do_sample=do_sample,
        temperature=0,
        repetition_penalty=repetition_penalty,
        diversity_penalty=diversity_penalty
    )
    # print(out)
    return tokenizer.decode(out[0], skip_special_tokens=True)

print(paraphrase('Каждый охотник желает знать, где сидит фазан.'))
# Все охотники хотят знать где фазан сидит.


# %%
for i in range(5):
    print(paraphrase('Каждый охотник желает знать, где сидит фазан.'))

# %%
# key = os.getenv('OPEN_AI_KEY')

# %%
chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {'role': 'system', 'content': 'You are a top level copywriter algorithm. You have to rewrite sentences. For each sentence write 5 difference options. You have to answer the same langauge as a sentence you got. Your response should be JSON array.'},
        {'role': 'user', "content": "Тогда может быть пойдем дальше по темам?"}
    ],
)

# %%
eval(chat_completion.choices[0].message.content)

# %%
key

# %%
from openai import OpenAI
client = OpenAI(api_key=key)

chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {'role': 'system', 'content': 'You are a top level copywriter algorithm. You have to rewrite sentences. You have to answer the same langauge as a sentence you got.'},
        {'role': 'user', "content": "Тогда может быть пойдем дальше по темам?"}
    ],
    n=5
)

# %%
chat_completion.choices

# %%
single_text = "// script let s = { vod: [ \"Կազմված է ձեր աստղագուշակը ariրհոսի նշանի համար\", `${o} Ջրհոսների համար տարին կապված կլինի մեծ ծավալի աշխատանքի հետ, բայց մի կասկածեք ՝ ձեզ անպայման ժամանակ կմնա ընկերների հետ շփվելու և անձնական կյանքի համար: Այս տարի ձեզ համար օգտակար է ամրապնդել առողջությունը, ինչպես նաև հոգե-հուզական վիճակը ։ Անձնական կյանքում ամեն ինչ լավ կլինի, և եթե անցյալ տարի ամեն ինչ ճիշտ եք արել, ապա այժմ կարող եք վայելել ընտանեկան երջանկությունը: Այս տարի ամենաբարդ ոլորտը ֆինանսն է ։ Ստուգեք ամեն ինչ ՝ Ում եք տալիս, ումից եք վերցնում, ինչի մեջ եք ներդնում։ Եվ, իհարկե, փորձեք չգերազանցել ձեր հնարավորությունները ծախսերի առումով։`, \"Ինչպես երաշխավորված է նոր տարում հաջողություն և գումար ներգրավել:\", \"Ես կարող եմ օգնել ձեզ ստեղծել էներգետիկ բազա ամբողջ տարվա համար: Դուք երջանիկ, հարուստ, հաջողակ կլինեք, և փողը մագնիսի պես ձեզ կգրավի ամբողջ տարին: Կարծում են, որ դա անելը շատ դժվար է, բայց ես գիտեմ, թե ինչպես դա անել: Ես հսկայական աշխատանք եմ կատարել ՝ հասկանալու համար, թե մեր աշխարհի որ մեխանիզմներն են պետք գործարկել, որպեսզի մարդը հաջողություն ու հարստություն գտնի\", 'Իմ կախարդական Amulet Shield իսկապես ունի հզոր էներգիա և ի վիճակի է դրամական միջոցների հոսքեր ներգրավել իր սեփականատիրոջը: Պատրաստում Amulet Shield Եվ նրա դավադրությունն անցնում է հին վանական ծիսակարգի համաձայն և բացասական հետևանքներ չի ունենում: Amulet Shield ոչ միայն գումար կներգրավի ձեր կյանք, այլև կպաշտպանի չար աչքից և կպաշտպանի ձեզ վատ մարդկանցից:', \"Չմոռանանք, որ մենք արդեն ապրում ենք նոր ժամանակներում։ Իսկ նոր ժամանակներում ամեն ինչ հնարավոր է ։ \", \"Աղքատը կարող է հարստանալ:\", \"Հարուստը կարող է լուսավորվել:\", \"Ընտրությունը միայն ձերն է:\", ], fish: [ \"Ձկների նշանի համար ձեր աստղագուշակը կազմված է\", `${o} Ձկների համար տարին կապված կլինի շատ հաճելի պահերի հետ ։ Ամենակարևոր խնդիրն այն է, որ ինքներդ ձեզ ազատություն տաք տաղանդների և կարողությունների դրսևորման մեջ: Մի հապաղեք ինքներդ լինել և անել այն, ինչ ցանկանում է ձեր հոգին: Անձնական կյանքում հնարավոր են փոփոխություններ. պատրաստվեք այն փաստին, որ սա իրադարձությունների զարմանալի շարքի միայն սկիզբն է: Ինչ վերաբերում է ֆինանսներին, ապա այս տարի դուք լուրջ սահմանափակումներ կունենաք ։ Եթե \\ u200b \\ u200bունեք խնայողություններ, ապա դրանք պետք է ճիշտ ներդրվեն, հակառակ դեպքում կյանքը կփորձի ազատվել դրանցից:`, \"Ինչպես երաշխավորված է նոր տարում հաջողություն և գումար ներգրավել:\", \"Ես կարող եմ օգնել ձեզ ստեղծել էներգետիկ բազա ամբողջ տարվա համար: Դուք երջանիկ, հարուստ, հաջողակ կլինեք, և փողը մագնիսի պես ձեզ կգրավի ամբողջ տարին: Կարծում են, որ դա անելը շատ դժվար է, բայց ես գիտեմ, թե ինչպես դա անել: Ես հսկայական աշխատանք եմ կատարել ՝ հասկանալու համար, թե մեր աշխարհի որ մեխանիզմներն են պետք գործարկել, որպեսզի մարդը հաջողություն և հարստություն գտնի։\", 'Իմ կախարդական Amulet Shield իսկապես ունի հզոր էներգիա և ի վիճակի է դրամական միջոցների հոսքեր ներգրավել իր սեփականատիրոջը: Պատրաստում Amulet Shield Եվ նրա դավադրությունն անցնում է հին վանական ծիսակարգի համաձայն և բացասական հետևանքներ չի ունենում: Amulet Shield ոչ միայն գումար կներգրավի ձեր կյանք, այլև կպաշտպանի չար աչքից և կպաշտպանի ձեզ վատ մարդկանցից:', \"Չմոռանանք, որ մենք արդեն ապրում ենք նոր ժամանակներում։ Իսկ նոր ժամանակներում ամեն ինչ հնարավոր է ։ \", \"Աղքատը կարող է հարստանալ:\", \"Հարուստը կարող է լուսավորվել:\", \"Ընտրությունը միայն ձերն է:\", ], oven: [ \"Ձեր աստղագուշակը Խոյ նշանի համար կազմված է\", `${o} տարին ձեզ համար փոփոխությունների ժամանակ կլինի։ Սպասվում են կարիերայի և անձնային վերելքներ, անսպասելի բարձունքների նվաճում, հնարավոր է նույնիսկ ճակատագրական հանդիպում: Բայց ձեզ խորհուրդ է տրվում ավելի զուսպ լինել, կոնֆլիկտային իրավիճակներ չբավարարել և ներել հին վիրավորանքները ։ Տարին իդեալական է կյանքը \"մաքուր էջից\" սկսելու համար ։ Սպիտակ մետաղական Ցուլի տարին ձեզ ֆինանսական անկախություն կբերի, Բայց դրա համար ստիպված կլինեք առավելագույն ջանքեր գործադրել: Հնարավոր են նաև պարբերաբար հաճելի անակնկալներ ՝ անսպասելի մրցանակներ, վիճակախաղի շահումներ, արժեքավոր նվերներ։`, \"Ինչպես երաշխավորված է նոր տարում հաջողություն և գումար ներգրավել:\", \"Ես կարող եմ օգնել ձեզ ստեղծել էներգետիկ բազա ամբողջ տարվա համար: Դուք երջանիկ, հարուստ, հաջողակ կլինեք, և փողը մագնիսի պես ձեզ կգրավի ամբողջ տարին: Կարծում են, որ դա անելը շատ դժվար է, բայց ես գիտեմ, թե ինչպես դա անել: Ես հսկայական աշխատանք եմ կատարել ՝ հասկանալու համար, թե մեր աշխարհի որ մեխանիզմներն են պետք գործարկել, որպեսզի մարդը հաջողություն և հարստություն գտնի։\", 'Իմ կախարդական Amulet Shield իսկապես ունի հզոր էներգիա և ի վիճակի է դրամական միջոցների հոսքեր ներգրավել իր սեփականատիրոջը: Պատրաստում Amulet Shield Եվ նրա դավադրությունն անցնում է հին վանական ծիսակարգի համաձայն և բացասական հետևանքներ չի ունենում: Amulet Shield ոչ միայն գումար կներգրավի ձեր կյանք, այլև կպաշտպանի չար աչքից և կպաշտպանի ձեզ վատ մարդկանցից:', \"Չմոռանանք, որ մենք արդեն ապրում ենք նոր ժամանակներում։ Իսկ նոր ժամանակներում ամեն ինչ հնարավոր է ։ \", \"Աղքատը կարող է հարստանալ:\", \"Հարուստը կարող է լուսավորվել:\", \"Ընտրությունը միայն ձերն է:\", ], telec: [ \"Ձեր աստղագուշակը կազմված է urուլի նշանի համար\", `${o} ձեզ համար ամենահեշտ տարին չի լինի ։ Առաջադրանքները, որոնք նախկինում Արագ և առանց մեծ ջանքեր գործադրելու կատարում էիք, այժմ ձեզանից ավելի շատ ժամանակ և ջանք կպահանջեն: Բայց դա ամենևին չի նշանակում, որ դուք չեք կարողանա լուծել դրանք ։ Հիմնական բանը ' մի հրաժարվեք. Դուք անպայման հաջողության կհասնեք: Միշտ հիշեք դա, հատկապես, երբ ինչ-որ պահի ցանկանում եք թողնել ամեն ինչ: Մի տրվեք հույզերին և միշտ ձեր գլխում պահեք պահեստային տարբերակը: Սա հիանալի ժամանակ է պատասխանատու որոշումներ կայացնելու, խոստումնալից գործարքներ կնքելու, ինչպես նաև ցանկացած գնումների և վաճառքի համար:`, \"Ինչպես երաշխավորված է նոր տարում հաջողություն և գումար ներգրավել:\", \"Ես կարող եմ օգնել ձեզ ստեղծել էներգետիկ բազա ամբողջ տարվա համար: Դուք երջանիկ, հարուստ, հաջողակ կլինեք, և փողը մագնիսի պես ձեզ կգրավի ամբողջ տարին: Կարծում են, որ դա անելը շատ դժվար է, բայց ես գիտեմ, թե ինչպես դա անել: Ես հսկայական աշխատանք եմ կատարել ՝ հասկանալու համար, թե մեր աշխարհի որ մեխանիզմներն են պետք գործարկել, որպեսզի մարդը հաջողություն և հարստություն գտնի։\", 'Իմ կախարդական Amulet Shield իսկապես ունի հզոր էներգիա և ի վիճակի է դրամական միջոցների հոսքեր ներգրավել իր սեփականատիրոջը: Պատրաստում Amulet Shield Եվ նրա դավադրությունն անցնում է հին վանական ծիսակարգի համաձայն և բացասական հետևանքներ չի ունենում: Amulet Shield ոչ միայն գումար կներգրավի ձեր կյանք, այլև կպաշտպանի չար աչքից և կպաշտպանի ձեզ վատ մարդկանցից:', \"Չմոռանանք, որ մենք արդեն ապրում ենք նոր ժամանակներում։ Իսկ նոր ժամանակներում ամեն ինչ հնարավոր է ։ \", \"Աղքատը կարող է հարստանալ:\", \"Հարուստը կարող է լուսավորվել:\", \"Ընտրությունը միայն ձերն է:\", ], bliznecy: [ \"Ձեր աստղագուշակը Երկվորյակների նշանի համար կազմված է\", `Երկվորյակներ ${o} տարին պետք է գործի: Ամեն ինչ վերցրեք ձեր ձեռքը, նախաձեռնություն ցուցաբերեք և մի վախեցեք պատասխանատվությունից: Հիմնական բանը սկսելն է, մնացած բոլոր խնդիրները կարող եք լուծել գործընթացում: Ինչ վերաբերում է փողին, ապա այստեղ աստղագուշակն ամբողջությամբ ձեր կողմն է: Դուք կկարողանաք վաստակել այնքան գումար, որքան ձեզ հարկավոր է: Լավ արդյունքներ կարող է տալ ներդրումներ անշարժ գույքի կամ ինչ-որ ձեռնարկության. Փողը էներգիա է, և ֆինանսական բարեկեցության հասնելու համար պարզապես անհրաժեշտ է այն ներգրավել ձեր կյանքում:`, \"Ինչպես երաշխավորված է նոր տարում հաջողություն և գումար ներգրավել:\", \"Ես կարող եմ օգնել ձեզ ստեղծել էներգետիկ բազա ամբողջ տարվա համար: Դուք երջանիկ, հարուստ, հաջողակ կլինեք, և փողը մագնիսի պես ձեզ կգրավի ամբողջ տարին: Կարծում են, որ դա անելը շատ դժվար է, բայց ես գիտեմ, թե ինչպես դա անել: Ես հսկայական աշխատանք եմ կատարել ՝ հասկանալու համար, թե մեր աշխարհի որ մեխանիզմներն են պետք գործարկել, որպեսզի մարդը հաջողություն և հարստություն գտնի։\", 'Իմ կախարդական Amulet Shield իսկապես ունի հզոր էներգիա և ի վիճակի է դրամական միջոցների հոսքեր ներգրավել իր սեփականատիրոջը: Պատրաստում Amulet Shield Եվ նրա դավադրությունն անցնում է հին վանական ծիսակարգի համաձայն և բացասական հետևանքներ չի ունենում: Amulet Shield ոչ միայն գումար կներգրավի ձեր կյանք, այլև կպաշտպանի չար աչքից և կպաշտպանի ձեզ վատ մարդկանցից:', \"Չմոռանանք, որ մենք արդեն ապրում ենք նոր ժամանակներում։ Իսկ նոր ժամանակներում ամեն ինչ հնարավոր է ։ \", \"Աղքատը կարող է հարստանալ:\", \"Հարուստը կարող է լուսավորվել:\", \"Ընտրությունը միայն ձերն է:\", ], rak: [ \"Ձեր աստղագուշակը քաղցկեղի նշանի համար կազմված է\", `${o} տարին կարող է առանցքային լինել ձեր հետագա զարգացման համար: Այն որոշումները, որոնք դուք կկայացնեք, կդառնան ամենակարևորը և կորոշեն ձեր կյանքը շատ տարիներ առաջ: Եթե \\ u200b \\ u200bերկար ժամանակ ցանկանում եք ինչ-որ բան փոխել ձեր կյանքում, գործեք: Անձնական կյանքում փոփոխությունները կապված կլինեն ձեր սեփական առաջ շարժվելու հետ, ուստի դրանք արժե ուրախությամբ ընդունել: Ինչ վերաբերում է ֆինանսներին, ապա տարվա առաջին ամիսներն առավել բարենպաստ կլինեն, իսկ հետո ապավինեք ճակատագրին: Բայց եթե դուք ունեք հզոր էներգետիկ թալիսման, ապա կարող եք ավելի լավ անել:`, \"Ինչպես երաշխավորված է նոր տարում հաջողություն և գումար ներգրավել:\", \"Ես կարող եմ օգնել ձեզ ստեղծել էներգետիկ բազա ամբողջ տարվա համար: Դուք երջանիկ, հարուստ, հաջողակ կլինեք, և փողը մագնիսի պես ձեզ կգրավի ամբողջ տարին: Կարծում են, որ դա անելը շատ դժվար է, բայց ես գիտեմ, թե ինչպես դա անել: Ես հսկայական աշխատանք եմ կատարել ՝ հասկանալու համար, թե մեր աշխարհի որ մեխանիզմներն են պետք գործարկել, որպեսզի մարդը հաջողություն և հարստություն գտնի։\", 'Իմ կախարդական Amulet Shield իսկապես ունի հզոր էներգիա և ի վիճակի է դրամական միջոցների հոսքեր ներգրավել իր սեփականատիրոջը: Պատրաստում Amulet Shield Եվ նրա դավադրությունն անցնում է հին վանական ծիսակարգի համաձայն և բացասական հետևանքներ չի ունենում: Amulet Shield ոչ միայն գումար կներգրավի ձեր կյանք, այլև կպաշտպանի չար աչքից և կպաշտպանի ձեզ վատ մարդկանցից:', \"Չմոռանանք, որ մենք արդեն ապրում ենք նոր ժամանակներում։ Իսկ նոր ժամանակներում ամեն ինչ հնարավոր է ։ \", \"Աղքատը կարող է հարստանալ:\", \"Հարուստը կարող է լուսավորվել:\", \"Ընտրությունը միայն ձերն է:\", ], lev: [ \"Ձեր աստղագուշակը առյուծի նշանի համար կազմված է\", `${o} Առյուծների համար Տարին խոստանում է շատ հաջող լինել, հատկապես նոր ծանոթություններ հաստատելու և շահավետ առաջարկներ ստանալու առումով: Բայց նոր տարում ձեզ խոչընդոտներ են սպասվում, որոնք հաղթահարելը միշտ չէ, որ հեշտ կլինի ։ Պատրաստ եղեք բնավորություն դրսևորել և չկորցնել սիրտը: Մի ծեծեք ինքներդ ձեզ պարտությունների համար և Մի կենտրոնացեք բարդությունների վրա: Թողեք իրավիճակը և կտեսնեք, որ խնդիրներն ավելի հեշտ են լուծվում, քան պատկերացնում էիք: Ֆինանսական վիճակն այս տարի բավական լավ կլինի, և կարելի է սպասել մուտքերի։`, \"Ինչպես երաշխավորված է նոր տարում հաջողություն և գումար ներգրավել:\", \"Ես կարող եմ օգնել ձեզ ստեղծել էներգետիկ բազա ամբողջ տարվա համար: Դուք երջանիկ, հարուստ, հաջողակ կլինեք, և փողը մագնիսի պես ձեզ կգրավի ամբողջ տարին: Կարծում են, որ դա անելը շատ դժվար է, բայց ես գիտեմ, թե ինչպես դա անել: Ես հսկայական աշխատանք եմ կատարել ՝ հասկանալու համար, թե մեր աշխարհի որ մեխանիզմներն են պետք գործարկել, որպեսզի մարդը հաջողություն և հարստություն գտնի։\", 'Իմ կախարդական Amulet Shield իսկապես ունի հզոր էներգիա և ի վիճակի է դրամական միջոցների հոսքեր ներգրավել իր սեփականատիրոջը: Պատրաստում Amulet Shield Եվ նրա դավադրությունն անցնում է հին վանական ծիսակարգի համաձայն և բացասական հետևանքներ չի ունենում: Amulet Shield ոչ միայն գումար կներգրավի ձեր կյանք, այլև կպաշտպանի չար աչքից և կպաշտպանի ձեզ վատ մարդկանցից:', \"Չմոռանանք, որ մենք արդեն ապրում ենք նոր ժամանակներում։ Իսկ նոր ժամանակներում ամեն ինչ հնարավոր է ։ \", \"Աղքատը կարող է հարստանալ:\", \"Հարուստը կարող է լուսավորվել:\", \"Ընտրությունը միայն ձերն է:\", ], deva: [ \"Ձեր աստղագուշակը Կույսի նշանի համար կազմված է\", `${o} տարին դրական կլինի ձեզ համար և կբերի բազմաթիվ նոր ու հետաքրքիր իրադարձություններ: Հիմնական բանը ' մի նստեք փակ և ինքներդ ձեզ ավելի շատ ազատություն տվեք ընկերների հետ շփվելու և ծրագրեր իրականացնելու համար: Թույլ տվեք ինքներդ ձեզ երազանք և օգնեք այն իրականություն դարձնել: Դուք պետք է հենց հիմա նստեք և մանրամասն նկարագրեք տարվա պլանը, քանի որ նոր տարում ինքնաբուխ գործողությունները միշտ չէ, որ ռացիոնալ և ճիշտ կլինեն և դժվար թե հանգեցնեն դրական արդյունքի: Եթե \\ u200b \\ u200bերազում եք նոր տարվա մասին, ապա ձեզ հարկավոր է մի քանի խորհուրդ. Հավատացեք ինքներդ ձեզ, մի փոշիացրեք և Լսեք ձեր ինտուիցիան, ապա հաստատ չեք կարող սխալվել:`, \"Ինչպես երաշխավորված է նոր տարում հաջողություն և գումար ներգրավել:\", \"Ես կարող եմ օգնել ձեզ ստեղծել էներգետիկ բազա ամբողջ տարվա համար: Դուք երջանիկ, հարուստ, հաջողակ կլինեք, և փողը մագնիսի պես ձեզ կգրավի ամբողջ տարին: Կարծում են, որ դա անելը շատ դժվար է, բայց ես գիտեմ, թե ինչպես դա անել: Ես հսկայական աշխատանք եմ կատարել ՝ հասկանալու համար, թե մեր աշխարհի որ մեխանիզմներն են պետք գործարկել, որպեսզի մարդը հաջողություն և հարստություն գտնի։\", 'Իմ կախարդական Amulet Shield իսկապես ունի հզոր էներգիա և ի վիճակի է դրամական միջոցների հոսքեր ներգրավել իր սեփականատիրոջը: Պատրաստում Amulet Shield Եվ նրա դավադրությունն անցնում է հին վանական ծիսակարգի համաձայն և բացասական հետևանքներ չի ունենում: Amulet Shield ոչ միայն գումար կներգրավի ձեր կյանք, այլև կպաշտպանի չար աչքից և կպաշտպանի ձեզ վատ մարդկանցից:', \"Չմոռանանք, որ մենք արդեն ապրում ենք նոր ժամանակներում։ Իսկ նոր ժամանակներում ամեն ինչ հնարավոր է ։ \", \"Աղքատը կարող է հարստանալ:\", \"Հարուստը կարող է լուսավորվել:\", \"Ընտրությունը միայն ձերն է:\", ], vesy: [ \"Կշեռքի նշանի համար ձեր աստղագուշակը կազմված է\", `Եթե դուք սովորեք լսել ձեր ինտուիցիան, բոլոր իմաստալից որոշումները, որոնք դուք կկայացնեք ${o} են ճիշտ լինել և հանգեցնել ցանկալի արդյունքի: Ընտանեկան իրավիճակը համեմատաբար հանգիստ է, բայց պետք չէ սկսել մեծ նախագծեր անշարժ գույքի հետ, Հիմա ժամանակը չէ ։ Օգտակար է ավելի շատ շփվել մարդկանց հետ, քանի որ կապերը, որոնք այժմ հաստատվելու են, ապագայում անպայման օգտակար կլինեն ձեզ համար: Ֆինանսական իրավիճակը կլինի հարաբերական կարգով ՝ վերելքներն ու վայրէջքներն այս տարի կանցնեն ձեզ:`, \"Ինչպես երաշխավորված է նոր տարում հաջողություն և գումար ներգրավել:\", \"Ես կարող եմ օգնել ձեզ ստեղծել էներգետիկ բազա ամբողջ տարվա համար: Դուք երջանիկ, հարուստ, հաջողակ կլինեք, և փողը մագնիսի պես ձեզ կգրավի ամբողջ տարին: Կարծում են, որ դա անելը շատ դժվար է, բայց ես գիտեմ, թե ինչպես դա անել: Ես հսկայական աշխատանք եմ կատարել ՝ հասկանալու համար, թե մեր աշխարհի որ մեխանիզմներն են պետք գործարկել, որպեսզի մարդը հաջողություն և հարստություն գտնի։\", 'Իմ կախարդական Amulet Shield իսկապես ունի հզոր էներգիա և ի վիճակի է դրամական միջոցների հոսքեր ներգրավել իր սեփականատիրոջը: Պատրաստում Amulet Shield Եվ նրա դավադրությունն անցնում է հին վանական ծիսակարգի համաձայն և բացասական հետևանքներ չի ունենում: Amulet Shield ոչ միայն գումար կներգրավի ձեր կյանք, այլև կպաշտպանի չար աչքից և կպաշտպանի ձեզ վատ մարդկանցից:', \"Չմոռանանք, որ մենք արդեն ապրում ենք նոր ժամանակներում։ Իսկ նոր ժամանակներում ամեն ինչ հնարավոր է ։ \", \"Աղքատը կարող է հարստանալ:\", \"Հարուստը կարող է լուսավորվել:\", \"Ընտրությունը միայն ձերն է:\", ], scorpio: [ \"Ձեր աստղագուշակը Կարիճի նշանի համար կազմված է\", `Կարիճների Համար ${o} տարին խոստանում է շատ բարենպաստ լինել, հատկապես այն ամենում, ինչը վերաբերում է ձեր աշխարհայացքի ընդլայնմանը, նոր գիտելիքների ձեռքբերմանը, բազմազան ուղևորություններին և ճանապարհորդություններին, հոգևոր զարգացմանը.Այժմ կարող եք Ձեզ թույլ տալ այն, ինչի մասին նախկինում կարող էիք երազել: Անձնական կյանքում հնարավոր են փոփոխություններ. պատրաստվեք այն փաստին, որ սա իրադարձությունների զարմանալի շարքի միայն սկիզբն է: Ֆինանսական հարցերում Կարիճները հաջողակ կլինեն, քանի որ աճում է ձեր էներգիան և, որպես հետևանք, նյութական բարեկեցությունը:`, \"Ինչպես երաշխավորված է նոր տարում հաջողություն և գումար ներգրավել:\", \"Ես կարող եմ օգնել ձեզ ստեղծել էներգետիկ բազա ամբողջ տարվա համար: Դուք երջանիկ, հարուստ, հաջողակ կլինեք, և փողը մագնիսի պես ձեզ կգրավի ամբողջ տարին: Կարծում են, որ դա անելը շատ դժվար է, բայց ես գիտեմ, թե ինչպես դա անել: Ես հսկայական աշխատանք եմ կատարել ՝ հասկանալու համար, թե մեր աշխարհի որ մեխանիզմներն են պետք գործարկել, որպեսզի մարդը հաջողություն և հարստություն գտնի։\", 'Իմ կախարդական Amulet Shield իսկապես ունի հզոր էներգիա և ի վիճակի է դրամական միջոցների հոսքեր ներգրավել իր սեփականատիրոջը: Պատրաստում Amulet Shield Եվ նրա դավադրությունն անցնում է հին վանական ծիսակարգի համաձայն և բացասական հետևանքներ չի ունենում: Amulet Shield ոչ միայն գումար կներգրավի ձեր կյանք, այլև կպաշտպանի չար աչքից և կպաշտպանի ձեզ վատ մարդկանցից:', \"Չմոռանանք, որ մենք արդեն ապրում ենք նոր ժամանակներում։ Իսկ նոր ժամանակներում ամեն ինչ հնարավոր է ։ \", \"Աղքատը կարող է հարստանալ:\", \"Հարուստը կարող է լուսավորվել:\", \"Ընտրությունը միայն ձերն է:\", ], strelec: [ \"Ձեր աստղագուշակը Աղեղնավոր նշանի համար կազմված է\", `${o} Աղեղնավորների համար տարին յուրահատուկ ու անկրկնելի կլինի իրենց ՝ որպես անձի գիտակցման առումով։ Դուք ինքներդ ձեզ այնքան կհավատաք, որ կկարողանաք իրականացնել ամենահամարձակ ծրագրերը։ Այս տարին կլինի ձեր անձնական և հոգևոր զարգացման նոր ցիկլի սկիզբը: Անձնական կյանքում իրավիճակը հարթ և կայուն կլինի, սիրո մեջ լուրջ ճգնաժամեր ձեզ հիմա չեն սպառնում։ Ինչ վերաբերում է ֆինանսներին, ապա այս տարի դուք լուրջ սահմանափակումներ կունենաք ։ Եթե \\ u200b \\ u200bունեք խնայողություններ, ապա դրանք պետք է ճիշտ ներդրվեն, հակառակ դեպքում կյանքը կփորձի ազատվել դրանցից:`, \"Ինչպես երաշխավորված է նոր տարում հաջողություն և գումար ներգրավել:\", \"Ես կարող եմ օգնել ձեզ ստեղծել էներգետիկ բազա ամբողջ տարվա համար: Դուք երջանիկ, հարուստ, հաջողակ կլինեք, և փողը մագնիսի պես ձեզ կգրավի ամբողջ տարին: Կարծում են, որ դա անելը շատ դժվար է, բայց ես գիտեմ, թե ինչպես դա անել: Ես հսկայական աշխատանք եմ կատարել ՝ հասկանալու համար, թե մեր աշխարհի որ մեխանիզմներն են պետք գործարկել, որպեսզի մարդը հաջողություն և հարստություն գտնի։\", 'Իմ կախարդական Amulet Shield իսկապես ունի հզոր էներգիա և ի վիճակի է դրամական միջոցների հոսքեր ներգրավել իր սեփականատիրոջը: Պատրաստում Amulet Shield Եվ նրա դավադրությունն անցնում է հին վանական ծիսակարգի համաձայն և բացասական հետևանքներ չի ունենում: Amulet Shield ոչ միայն գումար կներգրավի ձեր կյանք, այլև կպաշտպանի չար աչքից և կպաշտպանի ձեզ վատ մարդկանցից:', \"Չմոռանանք, որ մենք արդեն ապրում ենք նոր ժամանակներում։ Իսկ նոր ժամանակներում ամեն ինչ հնարավոր է ։ \", \"Աղքատը կարող է հարստանալ:\", \"Հարուստը կարող է լուսավորվել:\", \"Ընտրությունը միայն ձերն է:\", ], kozerog: [ \"Ձեր աստղագուշակը Այծեղջյուրի նշանի համար կազմված է\", `Ին ${o} տարվա ընթացքում Այծեղջյուրները կամրապնդեն այն, ինչ սկսվել է անցյալում, մասնավորապես ՝ ձեր սեփական անձը, անհատական \\ u200b \\ u200bարժեքների համակարգը, որի վրա դուք ապավինելու եք երկար տարիներ: Ձեր էներգիան ուժեղանում է, և դուք կարող եք պատրաստակամություն զգալ շատ մեծ քայլերի և փոփոխությունների համար: Այս տարին շատ կարևոր է լուրջ, երկարատև հարաբերություններ կառուցելու համար։ Դուք պետք է հատուկ խնամք ցուցաբերեք դրամական հարցերում: Ավաղ, գործնական Այծեղջյուր ${o} տարին կարող է անսպասելիորեն կորցնել իր գործնականությունը, ինչը կարող է բացասաբար ազդել նրա ֆինանսական բարեկեցության վրա:`, \"Ինչպես երաշխավորված է նոր տարում հաջողություն և գումար ներգրավել:\", \"Ես կարող եմ օգնել ձեզ ստեղծել էներգետիկ բազա ամբողջ տարվա համար: Դուք երջանիկ, հարուստ, հաջողակ կլինեք, և փողը մագնիսի պես ձեզ կգրավի ամբողջ տարին: Կարծում են, որ դա անելը շատ դժվար է, բայց ես գիտեմ, թե ինչպես դա անել: Ես հսկայական աշխատանք եմ կատարել ՝ հասկանալու համար, թե մեր աշխարհի որ մեխանիզմներն են պետք գործարկել, որպեսզի մարդը հաջողություն և հարստություն գտնի։\", 'Իմ կախարդական Amulet Shield իսկապես ունի հզոր էներգիա և ի վիճակի է դրամական միջոցների հոսքեր ներգրավել իր սեփականատիրոջը: Պատրաստում Amulet Shield Եվ նրա դավադրությունն անցնում է հին վանական ծիսակարգի համաձայն և բացասական հետևանքներ չի ունենում: Amulet Shield ոչ միայն գումար կներգրավի ձեր կյանք, այլև կպաշտպանի չար աչքից և կպաշտպանի ձեզ վատ մարդկանցից:', \"Չմոռանանք, որ մենք արդեն ապրում ենք նոր ժամանակներում։ Իսկ նոր ժամանակներում ամեն ինչ հնարավոր է ։ \", \"Աղքատը կարող է հարստանալ:\", \"Հարուստը կարող է լուսավորվել:\", \"Ընտրությունը միայն ձերն է:\", ],"[:4_500]
texts = [single_text for x in range(5)]

# %%
resp = client.embeddings.create(input=texts, model='text-embedding-ada-002')

# %%
