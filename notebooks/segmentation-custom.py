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
df = pd.read_csv('/mnt/c/Users/andrei/Downloads/400-415-episodes-ground-truth.csv')
df = df[df['episode'] == 412]
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


def tsne_scatter(embeddings: list[float], metric: t.Union[str, t.Callable], title: str) -> None:
    tsne = TSNE(n_components=2, metric=metric, random_state=42, early_exaggeration=20)
    tsne_embeddings = tsne.fit_transform(embeddings)

    tsne_df = pd.DataFrame(tsne_embeddings, columns=('x', 'y'))
    tsne_df = df.join(tsne_df.set_index(df.index))
    
    tsne_df['ground_truth'] = tsne_df['ground_truth'].astype(str)
    return px.scatter(
        title=f'TSNE - {metric}: {title}',
        data_frame=tsne_df,
        x='x',
        y='y',
        color='ground_truth',
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


def pca_scatter(embeddings: list[float], title: str) -> None:
    pca = PCA(n_components=2, random_state=42)
    pca_embeddings = pca.fit_transform(embeddings)

    pca_df = pd.DataFrame(pca_embeddings, columns=('x', 'y'))
    pca_df = df.join(pca_df.set_index(df.index))
    
    pca_df['ground_truth'] = pca_df['ground_truth'].astype(str)
    return px.scatter(
        title=f'PCA: {title}',
        data_frame=pca_df,
        x='x',
        y='y',
        color='ground_truth',
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


def umap_scatter(embeddings: list[float], metric: t.Union[str, t.Callable], title: str) -> None:
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric=metric)
    umap_embeddings = umap_reducer.fit_transform(embeddings)

    umap_df = pd.DataFrame(umap_embeddings, columns=('x', 'y'))
    umap_df = df.join(umap_df.set_index(df.index))
    
    umap_df['ground_truth'] = umap_df['ground_truth'].astype(str)
    return px.scatter(
        title=f'UMAP - {metric}: {title}',
        data_frame=umap_df,
        x='x',
        y='y',
        color='ground_truth',
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
