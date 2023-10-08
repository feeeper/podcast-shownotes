# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Segmentation. Part 2

# %% [markdown]
# In this section I'm going to 

# %%
import typing as t

import math
import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import plotly.express as px

from stop_words import get_stop_words

import nltk
from nltk.metrics import windowdiff
from nltk.tokenize import texttiling
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.texttiling import TokenSequence, TokenTableField, smooth
nltk.download('stopwords')

from sentence_transformers import SentenceTransformer

from pyconverse import SemanticTextSegmention

# %%
df = pd.read_csv('../data/400-415-episodes-ground-truth.csv')
df = df[df['episode'] == 412]
df = df.drop('Unnamed: 9', axis='columns')
df

# %%
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cuda')

# %%
ru_embeddings = model.encode(df['ru_sentence'].values)

# %%
en_embeddings = model.encode(df['en_sentence'].values)

# %% [markdown]
# ## Dimension reduction

# %% [markdown]
# ### tSNE

# %%
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine


def tsne_scatter(embeddings: list[float], metric: t.Union[str, t.Callable], title: str, color_column: str = 'ground_truth') -> None:
    tsne = TSNE(n_components=2, metric=metric, random_state=42, early_exaggeration=20, n_jobs=-1)
    tsne_embeddings = tsne.fit_transform(embeddings)

    tsne_df = pd.DataFrame(tsne_embeddings, columns=('x', 'y'))
    tsne_df = df.join(tsne_df.set_index(df.index))
    
    tsne_df[color_column] = tsne_df[color_column].astype(str)
    metric_name = metric if type(metric) == str else metric.__name__
    return px.scatter(
        title=f'TSNE - {metric_name}: {title}',
        data_frame=tsne_df,
        x='x',
        y='y',
        color=color_column,
        hover_data=['ru_sentence', 'en_sentence'],
        width=1000,
        height=1000)


# %%
tsne_scatter(ru_embeddings, 'euclidean', 'Russian')

# %%
tsne_scatter(en_embeddings, 'euclidean', 'English')

# %%
tsne_scatter(ru_embeddings, cosine, 'Russian')

# %%
tsne_scatter(en_embeddings, cosine, 'English')

# %% [markdown]
# ### PCA

# %%
import numpy as np
from sklearn.decomposition import PCA


def pca_scatter(embeddings: list[float], title: str, color_column: str = 'ground_truth') -> None:
    pca = PCA(n_components=2, random_state=42)
    pca_embeddings = pca.fit_transform(embeddings)

    pca_df = pd.DataFrame(pca_embeddings, columns=('x', 'y'))
    pca_df = df.join(pca_df.set_index(df.index))
    
    pca_df[color_column] = pca_df[color_column].astype(str)
    return px.scatter(
        title=f'PCA: {title}',
        data_frame=pca_df,
        x='x',
        y='y',
        color=color_column,
        hover_data=['ru_sentence', 'en_sentence'],
        width=1000,
        height=1000)


# %%
pca_scatter(ru_embeddings, 'Russian')

# %%
pca_scatter(en_embeddings, 'Russian')

# %% [markdown]
# ### UMAP

# %%
import umap


def umap_scatter(embeddings: list[float], metric: t.Union[str, t.Callable], title: str, color_column: str = 'ground_truth') -> None:
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric=metric, n_jobs=-1)
    umap_embeddings = umap_reducer.fit_transform(embeddings)

    umap_df = pd.DataFrame(umap_embeddings, columns=('x', 'y'))
    umap_df = df.join(umap_df.set_index(df.index))
    
    umap_df[color_column] = umap_df[color_column].astype(str)
    metric_name = metric if type(metric) == str else metric.__name__
    return px.scatter(
        title=f'UMAP - {metric_name} - {title}',
        data_frame=umap_df,
        x='x',
        y='y',
        color=color_column,
        hover_data=['ru_sentence', 'en_sentence'],
        width=1000,
        height=1000)


# %%
umap_scatter(ru_embeddings, 'euclidean', 'Russian')

# %%
umap_scatter(en_embeddings, 'euclidean', 'English')

# %%
umap_scatter(ru_embeddings, cosine, 'Russian')

# %%
umap_scatter(en_embeddings, cosine, 'English')

# %% [markdown]
# ## Labse embeddings

# %%
model = SentenceTransformer('sentence-transformers/labse', device='cuda')
model

# %%
labse_ru_embeddings = model.encode(df['ru_sentence'].values)

# %%
labse_en_embeddings = model.encode(df['en_sentence'].values)

# %%
tsne_scatter(labse_ru_embeddings, 'euclidean', 'Russian')

# %%
tsne_scatter(labse_en_embeddings, 'euclidean', 'English')

# %%
tsne_scatter(labse_ru_embeddings, cosine, 'Russian')

# %%
tsne_scatter(labse_en_embeddings, cosine, 'English')

# %%
pca_scatter(labse_ru_embeddings, 'Russian')

# %%
pca_scatter(labse_en_embeddings, 'English')

# %%
umap_scatter(labse_ru_embeddings, 'euclidean', 'Russian')

# %%
umap_scatter(labse_en_embeddings, 'euclidean', 'English')

# %%
umap_scatter(labse_ru_embeddings, cosine, 'Russian')

# %%
umap_scatter(labse_en_embeddings, cosine, 'English')

# %% [markdown]
# # Supervised segmentation

# %%
ru_embeddings_df = pd.DataFrame(labse_ru_embeddings, columns=[f'ru_{x}' for x in range(len(labse_ru_embeddings[0]))])
en_embeddings_df = pd.DataFrame(labse_en_embeddings, columns=[f'en_{x}' for x in range(len(labse_en_embeddings[0]))])
embeddings_df = ru_embeddings_df.join(en_embeddings_df)
embeddings_df = df[['size', 'episode', 'start', 'end', 'ru_sentence', 'en_sentence', 'topic_num', 'ground_truth']].join(embeddings_df.set_index(df.index))
embeddings_df.head()

# %%
import re

def get_en_embedding(df: pd.DataFrame, idx: int) -> list[float]:
    return df.loc[idx][[x for x in list(df) if re.match('en\_\d+', x)]].values

def get_ru_embedding(df: pd.DataFrame, idx: int) -> list[float]:
    return df.loc[idx][[x for x in list(df) if re.match('ru\_\d+', x)]].values


# %%
idx = 14492

ru_vector = get_ru_embedding(embeddings_df, idx)
en_vector = get_en_embedding(embeddings_df, idx)

cosine(ru_vector, en_vector)


# %%
def apply_cosine(s: pd.Series, df: pd.DataFrame) -> float:
    ru_vec = s[[x for x in list(df) if re.match('ru\_\d+', x)]]
    en_vec = s[[x for x in list(df) if re.match('en\_\d+', x)]]
    res = cosine(ru_vec, en_vec)
    return res


# %%
embeddings_df['cosine'] = embeddings_df.apply(lambda x: apply_cosine(x, embeddings_df), axis='columns')
embeddings_df.head()

# %%
embeddings_df[embeddings_df['cosine']==embeddings_df['cosine'].max()]

# %%
embeddings_df[embeddings_df['cosine']==embeddings_df['cosine'].min()]

# %%
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForNextSentencePrediction.from_pretrained("bert-base-cased")


# %%
def is_next_sentence(model: BertForNextSentencePrediction, tokenizer: BertTokenizer, sent_1: str, sent_2: str) -> float:
    try:
        encoding = tokenizer(sent_1, sent_2, return_tensors='pt')
        outputs = model(**encoding, labels=torch.LongTensor([1]))
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits)[0]
        return probs[0].item()
    except Exception as e:
        print(e)
        print(f'{sent_1=}')
        print(f'{sent_2=}')
        raise


# %%
sent_1 = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
sent_2 = "The sky is blue due to the shorter wavelength of blue light."

is_next_sentence(model, tokenizer, sent_1, sent_2)

# %%
sent_1 = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
sent_2 = "Here you can order any pasta or pizza."

is_next_sentence(model, tokenizer, sent_1, sent_2)

# %%
sent_1 = df.loc[13289, 'en_sentence']
sent_2 = df.loc[13290, 'en_sentence']

is_next_sentence(model, tokenizer, sent_1, sent_2)

# %%
from IPython.display import display_markdown

md_table =  '| sent_1 | sent_2 | prob |\n'
md_table += '|:-------|:-------|-----:|\n'
start_from = 13289
for i in range(10):
    sent_1 = df.loc[start_from + i, 'en_sentence']
    sent_2 = df.loc[start_from + i + 1, 'en_sentence']

    is_sent_2_continuation = is_next_sentence(model, tokenizer, sent_1, sent_2)

    md_table += f'| {sent_1} | {sent_2} | {is_sent_2_continuation } |\n'
    
display_markdown(md_table, raw=True)

# %%
md_table =  '| sent_1 | sent_2 | prob |\n'
md_table += '|:-------|:-------|-----:|\n'
start_from = 13289
for i in range(10):
    sent_1 = df.loc[start_from + i, 'en_sentence']
    sent_2 = df.loc[start_from + i + 10, 'en_sentence']

    is_sent_2_continuation = is_next_sentence(model, tokenizer, sent_1, sent_2)

    md_table += f'| {sent_1} | {sent_2} | {is_sent_2_continuation } |\n'
    
display_markdown(md_table, raw=True)

# %%
df[df['ru_sentence'].str.contains('тема')].head(10)

# %%
first_and_last_sentences_df = pd.concat([df.drop_duplicates('ground_truth', keep='first'), df.drop_duplicates('ground_truth', keep='last')]).sort_values('start')
first_and_last_sentences_df

# %%
idxs = first_and_last_sentences_df.index

md_table =  '| idx_1 | idx_2 | sent_1 | sent_2 | prob | ground_truth |\n'
md_table += '|------:|------:|:-------|:-------|-----:|-------------:|\n'

results = []
for i in range(0, len(idxs)-1):
    sent_1 = df.loc[idxs[i], 'en_sentence']
    sent_2 = df.loc[idxs[i + 1], 'en_sentence']
    is_sent_2_continuation = is_next_sentence(model, tokenizer, sent_1, sent_2)

    md_table += f'| {idxs[i]} | {idxs[i + 1]} | {sent_1} | {sent_2} | {is_sent_2_continuation } | {int(idxs[i + 1] - idxs[i] != 1)} |\n'
    results.append({
        'idx_1': idxs[i],
        'idx_2': idxs[i + 1],
        'sent_1': sent_1,
        'sent_2': sent_2,
        'is_sent_2_continuation': is_sent_2_continuation,
        'ground_truth': int(idxs[i + 1] - idxs[i] == 1)
    })
    
display_markdown(md_table, raw=True)

# %%
results_df = pd.DataFrame(results)
results_df

# %%
results_df[['is_sent_2_continuation']].corrwith(results_df['ground_truth'])

# %%
df['en_sentence_next'] = df['en_sentence'].shift(-1)
df['ru_sentence_next'] = df['ru_sentence'].shift(-1)
df


# %%
def apply_is_next_sentence(s: pd.Series) -> float:
    sent_1 = s['en_sentence']
    sent_2 = s['en_sentence_next']
    is_sent_2_continuation = is_next_sentence(model, tokenizer, sent_1, sent_2)
    return is_sent_2_continuation


# %%
df['is_continuation'] = df.apply(apply_is_next_sentence, axis='columns')
df

# %%
df['ground_truth_next'] = df['ground_truth'].shift(1).fillna(df['ground_truth'].max()).astype(int)
df[df['ground_truth'] != df['ground_truth_next']][['ru_sentence', 'ru_sentence_next', 'ground_truth', 'ground_truth_next', 'is_continuation']]


# %%
def apply_is_next_sentence_ground_truth(s: pd.Series) -> int:
    return 1 if s['ground_truth'] == s['ground_truth_next'] else 0


# %%
df['is_next_sentence_ground_truth'] = df.apply(apply_is_next_sentence_ground_truth, axis='columns')
df[['ru_sentence', 'ru_sentence_next', 'ground_truth', 'ground_truth_next', 'is_continuation', 'is_next_sentence_ground_truth']]

# %%
df[['is_continuation']].corrwith(df['is_next_sentence_ground_truth'])

# %%
df['en_sentence_next_2'] = df['en_sentence_next'].shift(-1)
df['ru_sentence_next_2'] = df['ru_sentence_next'].shift(-1)
df

# %%
df['ru_sentence_next_merged'] = df['ru_sentence_next'] + ' ' + df['ru_sentence_next_2']
df[['ru_sentence', 'ru_sentence_next_merged', 'ru_sentence_next', 'ru_sentence_next_2']].head()

# %%
df['en_sentence_next_merged'] = df['en_sentence_next'] + ' ' + df['en_sentence_next_2']
df[['en_sentence', 'en_sentence_next_merged', 'en_sentence_next', 'en_sentence_next_2']].head()

# %%
df['en_sentence_next_merged_next'] = df['en_sentence_next_merged'].shift(-1)
df['ru_sentence_next_merged_next'] = df['ru_sentence_next_merged'].shift(-1)
df.head()


# %%
def apply_is_continuation(s: pd.Series, sent_1_header: str, sent_2_header: str) -> float:
    sent_1 = s[sent_1_header] if not pd.isna(s[sent_1_header]) else ''
    sent_2 = s[sent_2_header] if not pd.isna(s[sent_2_header]) else ''

    is_sent_2_continuation = is_next_sentence(model, tokenizer, sent_1, sent_2)

    return is_sent_2_continuation

df['is_continuation_2_sent'] = df.apply(lambda x: apply_is_continuation(x, 'en_sentence_next_merged', 'en_sentence_next_merged_next'), axis='columns')
df.head()

# %%
df[['is_next_sentence_ground_truth', 'is_continuation_2_sent', 'is_continuation']]  # .corrwith(df['is_continuation'])

# %%
df[['is_next_sentence_ground_truth', 'is_continuation_2_sent', 'is_continuation']][df['is_next_sentence_ground_truth'] == 0]

# %%
df[df['is_next_sentence_ground_truth'] == 0]['is_continuation_2_sent'].describe()

# %%
df[df['is_next_sentence_ground_truth'] == 0]['is_continuation'].describe()

# %%
df = df.join(pd.DataFrame(labse_en_embeddings, columns=[f'en_{x}' for x in range(len(labse_en_embeddings[0]))], index=df.index))
df = df.join(pd.DataFrame(labse_ru_embeddings, columns=[f'ru_{x}' for x in range(len(labse_ru_embeddings[0]))], index=df.index))
df


# %%
def umap_scatter(embeddings: list[float], metric: t.Union[str, t.Callable], color_column: str, title: str) -> None:
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric=metric)
    umap_embeddings = umap_reducer.fit_transform(embeddings)

    umap_df = pd.DataFrame(umap_embeddings, columns=('x', 'y'))
    umap_df = df.join(umap_df.set_index(df.index))
    
    umap_df[color_column] = umap_df[color_column].astype(str)
    metric_name = metric if type(metric) == str else metric.__name__
    return px.scatter(
        title=f'UMAP - {metric_name} - {title}',
        data_frame=umap_df,
        x='x',
        y='y',
        color=color_column,
        hover_data=['ru_sentence', 'en_sentence'],
        width=1000,
        height=1000)


# %%
df['is_next_sentence_ground_truth'].value_counts()

# %%
ru_labse_scatter_euc = umap_scatter(labse_ru_embeddings, 'euclidean', 'is_next_sentence_ground_truth', title='Two classes: next sentence is the new topic\'s first sentence')
ru_labse_scatter_euc

# %%
en_labse_scatter_euc = umap_scatter(labse_en_embeddings, 'euclidean', 'is_next_sentence_ground_truth', title='Two classes: next sentence is the new topic\'s first sentence')
en_labse_scatter_euc

# %%
ru_scatter_euc = umap_scatter(ru_embeddings, 'euclidean', 'is_next_sentence_ground_truth', title='Two classes: next sentence is the new topic\'s first sentence (non-labse)')
ru_scatter_euc

# %%
en_scatter_euc = umap_scatter(en_embeddings, 'euclidean', 'is_next_sentence_ground_truth', title='Two classes: next sentence is the new topic\'s first sentence (non-labse)')
en_scatter_euc

# %%
ru_labse_scatter_cos = umap_scatter(labse_ru_embeddings, cosine, 'is_next_sentence_ground_truth', title='Two classes: next sentence is the new topic\'s first sentence (labse, cosine)')
ru_labse_scatter_cos

# %%
en_labse_scatter_cos = umap_scatter(labse_en_embeddings, cosine, 'is_next_sentence_ground_truth', title='Two classes: next sentence is the new topic\'s first sentence (labse, cosine)')
en_labse_scatter_cos

# %%
type(cosine)

# %%
df[df['is_next_sentence_ground_truth']==0][['ru_sentence']].values

# %%
df[df['ru_sentence'] == 'Ну, это не я, это слушатель в Divzen чатике с никнеймом AlexD, он принес ссылку на блокпост, который рассказывает про новый флаг в гите UpdateRefs.']

# %%
df[df['ground_truth'] == 2]

# %%
df[df['ru_sentence'] == 'Ну, это не я, это слушатель в Divzen чатике с никнеймом AlexD, он принес ссылку на блокпост, который рассказывает про новый флаг в гите UpdateRefs.']

# %%
df.loc[13_619]

# %%
umap_scatter(labse_ru_embeddings, 'correlation', 'ground_truth', 'ru labse embeddings, ground_truth')

# %%
umap_scatter(labse_ru_embeddings, 'correlation', 'is_next_sentence_ground_truth', 'ru labse embeddings, is_next_sentence_ground_truth')

# %%
