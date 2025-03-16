# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: jupytext,text_representation
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Failed clustering segmentation

# %% [markdown]
# This is the third installment in a series about text segmentation. The first part covered text segmentation metrics, and the second focused on unsupervised text segmentation methods.

# %% [markdown]
# When I began writing this article, I had a strong belief that employing clustering was an excellent approach to segment text into distinct topics. Regrettably, it turned out that I was mistaken. Perhaps I overlooked something crucial, which is why this approach yielded no meaningful results.

# %% [markdown]
# On the other hand I tried multiple dimention reduction techniques and got a lot of beautiful graphics that show my failure.

# %% [markdown]
# I had two goals to investigate:
# 1. Grouping sentences about a topic in a podcast episode into one cluster.
# 2. Distinguishing between sentences within a topic and sentences that start a new discussion, and creating two clusters accordingly.

# %% [markdown]
# However, when I visualized the embeddings and examined the plots, it was evident that there were no clear boundaries between topics. This indicated that using clustering for text segmentation wouldn't be straightforward. It appears that more research is needed, and I hope to dedicate more time to it later.

# %% [markdown]
# By the way, this little research provided an opportunity to experiment with new visualization techniques and dimension reduction methods. So instead of focusing solely on text segmentation methods, this article has transformed into one about embeddings visualization and dimension reduction techniques.

# %% [markdown]
# In this article, I'll examine three dimension reduction techniques: t-SNE, PCA, and UMAP. For each technique, I'll create various scatter plots based on the objectives I mentioned earlier and specific options related to each method.
#
# > **Disclaimer:** I won't go into detailed algorithmic explanations in this article, but I'll strive to provide simplified explanations of the core concepts behind these methods.

# %% [markdown]
# ## Setup

# %% [markdown]
# As I've previously discussed, I'm using transcripts from the DevZen podcast. Specifically, I've selected the 412th episode randomly. In this article, I'll focus on visualizing and attempting to cluster sentences from this particular episode.

# %% [markdown]
# To start, I'll import all the essential libraries and generate embeddings for the upcoming analysis.

# %%
import os
import typing as t
from dataclasses import dataclass
import re
import warnings
warnings.filterwarnings("ignore")

from joblib import Memory

import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from tqdm.notebook import tqdm

from stop_words import get_stop_words
from sentence_transformers import SentenceTransformer
from nltk import windowdiff

# %%
memory = Memory('../.cache')

# %%
csv_path = '../data/400-415-episodes-ground-truth.csv'
df = pd.read_csv(csv_path)
df = df[df['episode'] == 412]
df.head()


# %% [markdown]
# Just a friendly reminder of the column meanings:
# 1. `size`: Size of the model used to transcribe an episode.
# 2. `episode`: Episode number.
# 3. `start` and `end`: Start and end timings for each sentence respectively. Some sentences may have overlapping timings due to a limitation during transcription.
# 4. `ru_sentence`: Original sentence in Russian.
# 5. `en_sentence`: Machine-translated original sentence in English.
# 6. `sentence_length`: Length of the original sentence.
# 7. `topic_num`: Topic number from the episode's show notes.
# 8. `ground_truth`: Manually set topic number.

# %% [markdown]
# ### Prepare embeddings

# %% [markdown]
# Similar to the [segmentation article](../article.html), I'll utilize the **SentenceTransformers** library to construct embeddings and **Plotly** for visualization.

# %% [markdown]
# For embeddings, I'll utilize two multilanguage models: `paraphrase-multilingual-MiniLM-L12-v2` and `labse`. I'll experiment with various parameters for each dimension reduction algorithm.

# %%
@dataclass
class Models:
    paraphrase = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    labse = 'sentence-transformers/labse'


# %%
@memory.cache
def get_embeddings(sentences: list[str], model_name: str) -> list[float]:
    model = SentenceTransformer(model_name)
    return model.encode(sentences)


# %%
ru_paraphrase_embeddings = get_embeddings(df['ru_sentence'].values, Models.paraphrase)
en_paraphrase_embeddings = get_embeddings(df['en_sentence'].values, Models.paraphrase)

ru_labse_embeddings = get_embeddings(df['ru_sentence'].values, Models.labse)
en_labse_embeddings = get_embeddings(df['en_sentence'].values, Models.labse)

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
# The initial model, `paraphrase-multilingual-MiniLM-L12-v2`, produces embeddings with 384 dimensions, while the second model, `labse`, yields 768-dimensional embeddings.

# %% [markdown]
# ### Prepare dataset

# %% [markdown]
# To accomplish the second objective, I'll use a bit of pandas ✨magic✨ to label each sentence in the dataset as either an "inner topic sentence" or a "first topic sentence."

# %%
df['ground_truth_next'] = df['ground_truth'].shift(1).fillna(df['ground_truth'].max()).astype(int)


# %%
def is_first_sentence_in_topic(s: pd.Series) -> int:
    return 0 if s['ground_truth'] == s['ground_truth_next'] else 1

df['is_first_sentence_in_topic'] = df.apply(is_first_sentence_in_topic, axis='columns')

# %%
df[df['is_first_sentence_in_topic'] == 1][['ru_sentence', 'en_sentence', 'ground_truth', 'is_first_sentence_in_topic']]

# %%
color_columns = ['ground_truth', 'is_first_sentence_in_topic']

# %% [markdown]
# From here I can start my research.

# %% [markdown]
# ## PCA

# %% [markdown]
# > **Principal component analysis (PCA)** is a popular technique for analyzing large datasets containing a high number of dimensions/features per observation, increasing the interpretability of data while preserving the maximum amount of information, and enabling the visualization of multidimensional data. [Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)
#
# This algorithm identifies the directions in the data where it varies the most. It then projects the data onto these directions (principal components), which are orthogonal and ordered by the amount of variance they explain. This transformation allows for a more compact representation of the original data while retaining its essential features.
#
# Here is a video from the StatQuest channel below. The creator provides a more comprehensive explanation of PCA along with examples.
# https://www.youtube.com/watch?v=FgakZw6K1QQ
#
# > **Disclaimer #2:** I have no information about the creator of the StatQuest channel; I'm just providing a link to the video.

# %%
from sklearn.decomposition import PCA


def pca_scatter(embeddings: list[float], title: str, color_column: str) -> None:
    pca = PCA(n_components=2, random_state=42)
    pca_embeddings = pca.fit_transform(embeddings)

    pca_df = pd.DataFrame(pca_embeddings, columns=('x', 'y'))
    pca_df = df.join(pca_df.set_index(df.index))
    
    pca_df[color_column] = pca_df[color_column].astype(str)
    fig = px.scatter(
        title=f'PCA: {title}',
        data_frame=pca_df,
        x='x',
        y='y',
        color=color_column,
        hover_data=['ru_sentence', 'en_sentence'],
        width=1000
    )
    
    fig.update_layout(legend=dict(
        orientation='h',
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    
    return fig

for color_column in color_columns:
    for lang, embedding_pairs in embeddings.items():
        for model_name, embedding in embedding_pairs.items():
            pca_scatter(
                embeddings=embedding,
                title=f'lang={lang}, model_name={model_name}, color={color_column}',
                color_column=color_column).show(renderer='png')
            print(f'[lang={lang}-model_name={model_name}-color={color_column}.html](./pca-lang={lang}-model_name={model_name}-color={color_column}.html)')

# %% [markdown]
# As observed, PCA does not effectively assist in segregating topics into distinct clusters — neither into 11 clusters for each topic nor into 2 clusters for first/inner sentences.

# %% [markdown]
# ## t-SNE

# %% [markdown]
# > **t-distributed stochastic neighbor embedding (t-SNE)** is a statistical method for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map. [Wikipedia](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
#
# I may not be able to explain the entire algorithm in detail, but in essence, it helps transform data into a representation where similar instances are nearby and dissimilar ones are far apart. This is achieved by optimizing the match between pairwise similarities in the original high-dimensional space and the lower-dimensional representation.
#
# To see the algorithm in action with a simple example, you can watch the video below from StatQuest:
# <iframe width="560" height="315" src="https://www.youtube.com/watch?v=NEaUSP4YerM" title="StatQuest: t-SNE" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# %%
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine


@memory.cache
def tsne_transform(embeddings: list[list[float]], metric: t.Union[str, t.Callable], early_exaggeration: int):
    tsne = TSNE(n_components=2, metric=metric, random_state=42, early_exaggeration=early_exaggeration, n_jobs=-1)
    tsne_embeddings = tsne.fit_transform(embeddings)
    return tsne_embeddings

def tsne_scatter(embeddings: list[list[float]], metric: t.Union[str, t.Callable], title: str, color_column: str) -> None:
    tsne_embeddings = tsne_transform(embeddings=embeddings, metric=metric, early_exaggeration=20)

    tsne_df = pd.DataFrame(tsne_embeddings, columns=('x', 'y'))
    tsne_df = df.join(tsne_df.set_index(df.index))    
    tsne_df[color_column] = tsne_df[color_column].astype(str)
    
    fig = px.scatter(
        title=f't-SNE: {title}',
        data_frame=tsne_df,
        x='x',
        y='y',
        color=color_column,
        hover_data=['ru_sentence', 'en_sentence'],
        width=1000)
    
    fig.update_layout(legend=dict(
        orientation='h',
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    
    return fig


# %% [markdown]
# In contrast to PCA, t-SNE supports various similarity metrics. In my case, I tested two of them: `scipy.spatial.distance.cosine` and the default `euclidean`.

# %%
metrics = {
    'cosine': cosine,
    'euclidean': 'euclidean',
}

# %%
for metric_name, metric in metrics.items():
    for color_column in color_columns:
        for lang, embedding_pairs in embeddings.items():
            for model_name, embedding in embedding_pairs.items():
                tsne_scatter(
                    embeddings=embedding,
                    metric=metric_name,
                    title=f'lang={lang}, metric={metric_name}, model_name={model_name}, color={color_column}',
                    color_column=color_column).show(renderer='png')                
                print(f'[lang={lang}-metric={metric_name}-model_name={model_name}-color={color_column}.html](./tsne-lang={lang}-metric={metric_name}-model_name={model_name}-color={color_column}.html)')

# %% [markdown]
# Unfortunately, t-SNE also didn't assist me in visualizing distinct differences between topics or identifying first/inner sentences within a topic.

# %% [markdown]
# ## UMAP

# %% [markdown]
# > **UMAP (Uniform Manifold Approximation and Projection)** is a novel manifold learning technique for dimension reduction. UMAP is constructed from a theoretical framework based in Riemannian geometry and algebraic topology. The result is a practical scalable algorithm that applies to real world data. The UMAP algorithm is competitive with t-SNE for visualization quality, and arguably preserves more of the global structure with superior run time performance. Furthermore, UMAP has no computational restrictions on embedding dimension, making it viable as a general purpose dimension reduction technique for machine learning. [arXiv](https://arxiv.org/abs/1802.03426)
#
#
# UMAP is a modern dimension reduction technique, and you can find a helpful video about the algorithm on StatQuest's YouTube channel using this link: [UMAP Explained](https://www.youtube.com/watch?v=eN0wFzBA4Sc).

# %%
import umap


@memory.cache
def umap_transofrm(embeddings: list[list[float]], metric: t.Union[str, t.Callable]):
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric=metric)
    umap_embeddings = umap_reducer.fit_transform(embeddings)
    return umap_embeddings


def umap_scatter(
    embeddings: list[float],
    metric: t.Union[str, t.Callable],
    color_column: str,
    title: str,
    df: pd.DataFrame
):
    umap_embeddings = umap_transofrm(embeddings, metric)

    umap_df = pd.DataFrame(umap_embeddings, columns=('x', 'y'))
    umap_df = df.join(umap_df.set_index(df.index))    
    umap_df[color_column] = umap_df[color_column].astype(str)
        
    fig = px.scatter(
        title=f'UMAP: {title}',
        data_frame=umap_df,
        x='x',
        y='y',
        color=color_column,
        hover_data=['ru_sentence', 'en_sentence'],
        width=1000
    )

    fig.update_layout(legend=dict(
        orientation='h',
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    
    return fig


# %% [markdown]
# Just like t-SNE, UMAP offers various metrics for similarity computation. In my exploration, I'll be utilizing both `scipy.spatial.distance.cosine` and the default `euclidean`.

# %%
for metric_name, metric in metrics.items():
    for color_column in color_columns:
        for lang, embedding_pairs in embeddings.items():
            for model_name, embedding in embedding_pairs.items():
                umap_scatter(
                    embeddings=embedding,
                    metric=metric_name,
                    title=f'lang={lang}, metric={metric_name}, model_name={model_name}, color={color_column}',
                    color_column=color_column,
                    df=df
                ).show(renderer='png')
                print(f'[interactive plot](./umap-lang={lang}-metric={metric_name}-model_name={model_name}-color={color_column}.html)')

# %% [markdown]
# So, there are no strong indications that we can segment the transcript into topics using just embeddings and clustering.

# %% [markdown]
# ## Conclusion

# %% [markdown]
# When I initially considered the concept of clustering podcast transcripts, I had high hopes for its success. Unfortunately, not all concepts lead to visible results.

# %% [markdown]
# On the flip side, I gained valuable experience with **Plotly** and various dimension reduction algorithms.

# %% [markdown]
# ## PS

# %% [markdown]
# You might wonder why I created each scatter plot individually instead of using subplots. The reason is that each subplot [cannot have its own individual legend](https://community.plotly.com/t/plotly-subplots-with-individual-legends/1754/10). However, the interactive legend was the main factor affecting my choice of **Plotly** over **seaborn** or **matplotlib**:

# %%
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# Create data for the scatter plots
x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y1 = [10, 11, 12, 13, 14, 4, 7, 1, 5, 11]
z1 = [1, 2, 3, 4, 0, 5, 6, 7, 8, 9]

custom_colors = px.colors.qualitative.D3
color_scale = [custom_colors[val] for val in z1]

x2 = [2, 3, 4, 5, 6]
y2 = [8, 9, 10, 11, 12]

# Create scatter plot traces
trace1 = go.Scatter(
    x=x1,
    y=y1,
    name='Plot 1',
    mode='markers',
    marker=dict(
        color=color_scale,  # Color scale based on the Z axis
        colorscale='Viridis',  # Choose the desired colorscale (e.g., 'Viridis', 'Blues', 'reds', etc.)
    ))
trace2 = go.Scatter(x=x2, y=y2, mode='markers', name='Plot 2')

# Create a subplot
fig = make_subplots(rows=1, cols=2)

# Add the traces to the subplot
fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=1, col=2)

# Display the plot
fig.show(renderer='png')

# %%
fig.write_html('./subplots.html')

# %% [markdown]
# # Grouping sentences

# %%
metric_name, metric = 'cosine', metrics['cosine']
color_column = 'ground_truth'
lang = 'russian'
embedding_pairs = embeddings[lang]

for model_name, embedding in embedding_pairs.items():
    fig = umap_scatter(
        embeddings=embedding,
        metric=metric_name,
        title=f'lang={lang}, metric={metric_name}, model_name={model_name}, color={color_column}',
        color_column=color_column,
        df=df)
    fig.show()

# %%
sentences = df[['ru_sentence', 'ground_truth', 'en_sentence']].values
batch_size = 5
overlap = 0
chunks = []
en_chunks = []
ground_truth = []
model_name = Models.paraphrase

# print(sentences)
for start in range(0, len(sentences), batch_size - overlap):
    chunks.append(' '.join([x[0] for x in sentences[start:start+batch_size]]))
    ground_truth.append(stats.mode([x[1] for x in sentences[start:start+batch_size]]).mode)
    en_chunks.append(' '.join([x[2] for x in sentences[start:start+batch_size]]))

ru_chunks_paraphrase_embeddings = get_embeddings(chunks, Models.paraphrase)
print(ru_chunks_paraphrase_embeddings.shape)
# en_paraphrase_embeddings = get_embeddings(df['en_sentence'].values, Models.paraphrase)

# ru_labse_embeddings = get_embeddings(df['ru_sentence'].values, Models.labse)

# %%
df2 = pd.DataFrame({
    'ground_truth': ground_truth,
    'ru_sentence': chunks,
    'en_sentence': en_chunks,
})
print(color_column)
umap_scatter(
    embeddings=ru_chunks_paraphrase_embeddings,
    metric=metric_name,
    title=f'lang={lang}, metric={metric_name}, model_name={model_name}, color={color_column}',
    color_column=color_column,
    df=df2
)

# %% [markdown]
# ## DBSCAN

# %%
from sklearn.cluster import DBSCAN

# %%
model = DBSCAN(n_jobs=-1, metric='cosine', eps=0.05)
preds = model.fit_predict(ru_chunks_paraphrase_embeddings)
preds

# %%
[x for x in list(zip(chunks, preds)) if x[1] != 0]

# %% [markdown]
# ## HDBSCAN

# %%
from hdbscan import HDBSCAN

hdbscan = HDBSCAN(
    # metric='cosine',
    min_cluster_size=2,
    # max_cluster_size=40,
    # leaf_size=10
)
preds = hdbscan.fit_predict(ru_chunks_paraphrase_embeddings)
preds

# %%
for x in [x for x in list(zip(chunks, preds))][:10]:
    print(f'{x[1]}: {x[0]}')

# %% [markdown]
# ## Pairwise distance

# %%
emb_plus_one = list(zip(ru_chunks_paraphrase_embeddings, ru_chunks_paraphrase_embeddings[1:]))
emb_plus_one_df = pd.DataFrame({
    'first': [x[0] for x in emb_plus_one],
    'second': [x[1] for x in emb_plus_one],
    'sentence': chunks[:-1],
    'ground_truth': ground_truth[:-1]
})
emb_plus_one_df[:20]

# %%
from sklearn.metrics import pairwise

emb_plus_one_df['dist'] = pairwise.paired_cosine_distances(list(emb_plus_one_df['first'].values), list(emb_plus_one_df['second'].values))

# %%
emb_plus_one_df

# %%
emb_plus_one_df[60:80]

# %% [markdown]
# # LLM

# %%
from openai import OpenAI
import json
import fuzzysearch

# %%
csv_path = '../data/400-415-episodes-ground-truth.csv'
df = pd.read_csv(csv_path)
df.head()


# %%
def is_first_sentence_in_topic(s: pd.Series) -> int:
    return 0 if s['ground_truth'] == s['ground_truth_next'] else 1

df['ground_truth_next'] = df['ground_truth'].shift(1).fillna(df['ground_truth'].max()).astype(int)
df['is_first_sentence_in_topic'] = df.apply(is_first_sentence_in_topic, axis='columns')
df.head()

# %%
api_key = os.getenv('OPEN_ROUTER_API_KEY')

# %%
client = OpenAI(
    api_key=api_key,
    base_url='https://openrouter.ai/api/v1'
)


# %%
def find_nearest(actual: list[int], expected: int, previous: int) -> int:
    if expected in actual:
        return expected
    
    greater_than_previous = [x for x in actual if x >= previous]
    if len(greater_than_previous) == 0:
        # not sure about this decision
        return actual[-1]
    
    candidates = [num for num in actual if num > previous]
    if not candidates:
        return None
    return min(candidates, key=lambda num: abs(num - expected))


# %%
memory.clear()


# %%
def adjust_delimiters(predicted_delimiters: list[int], min_length: int = 5):
    result = [predicted_delimiters[0]]
    
    for i in range(1, len(predicted_delimiters)):
        if predicted_delimiters[i] - result[-1] >= min_length:
            result.append(predicted_delimiters[i])
    
    return result

def get_delimiters(
    response: dict,
    sentences: list[str],
    min_topic_len: int = 5,
    verbose: int = 0
) -> list[int]:
    previous_index = -1
    predicted_delimiters = []
    for item in response:
        expected_index = item['index']
        try:
            actual_index = [i for i, x in enumerate(sentences) if x == item['text']]
            if verbose: print(f'1. {actual_index=}')
            
            if not actual_index:
                actual_index = [i for i, x in enumerate(sentences) if fuzzysearch.find_near_matches(x, item['text'], max_l_dist=1)]
            if verbose: print(f'2. {actual_index=}')
                
            if not actual_index:
                print(f'Warning: \"{item["text"]}\" (index={item["index"]}) not found ')
                continue

            if len(actual_index) > 1:
                if verbose:
                    print(f'{actual_index=}, {expected_index=}, {previous_index=}')
                actual_index = find_nearest(actual_index, expected_index, previous_index)
                if verbose: print(f'3. {actual_index=}')
            else:
                try:
                    actual_index = actual_index[0]
                    if verbose: print(f'4. {actual_index=}')
                except IndexError as ie:
                    print(f'{actual_index=}')
                    print(ie)

            if verbose:
                print(f"{expected_index=}, {actual_index=}, {item['text']=}")

            if actual_index is None:
                print(f'Warning: \"{item["text"]}\": actual_index is None ({previous_index=})')
                continue
                
            if previous_index > -1 and previous_index > actual_index:
                print(f'Warning: \"{item["text"]}\": {previous_index=} is larger than {actual_index=}')
                continue
            previous_index = actual_index
            predicted_delimiters.append(actual_index)
        except ValueError:
            print(f"\"{item}\" not presented")
    
    
    return adjust_delimiters(predicted_delimiters, min_topic_len)


# %%
def call_llm(
    model: str,
    sentences: list[str],
    max_tokens: int,
    temperature: int = 0
) -> dict:
    # prompt = (
    #     'You are a podcast host. You recorded an episode of a podcast and want to split the episode into multiple parts. '
    #     'Each part is related to a single topic. '
    #     'You have an array of sentences. '
    #     'Return the indexes of the sentences which could be considered as a first sentence of the new topic. '
    #     'The first sentence should be considered as the first sentence either. '
    #     'Do not change the sentences. '
    #     'Response should be valid JSON.'
    # )
#     PROMPT = """
#     You are a podcast host, and you have recorded an episode that you want to split into distinct topic-based segments.

#     You are given an array of sentences representing the transcript of the episode. Your task is to identify the starting point of each new topic and return the indexes of the sentences where new topics begin.

#     Guidelines:
#     - The first sentence should always be considered the start of a topic.  
#     - A new topic may begin when there is a noticeable shift in context, such as phrases like "Now let's talk about..." or a clear change in subject matter.  
#     - You cannot modify the sentences — only determine the starting points of new topics.  
#     - The output should be a valid JSON array of objects. Each object should have two keys:
#       - `index` — the integer index of the starting sentence  
#       - `sentence` — the sentence itself  

#     Example:
#     Input:
#     [
#       "Welcome to the show!",
#       "Today we will discuss climate change.",
#       "Global temperatures have risen significantly in the last decade.",
#       "Now let's talk about renewable energy.",
#       "Solar and wind power are becoming more affordable."
#     ]
#     Output:
#     [
#       {"index": 0, "sentence": "Welcome to the show!"},
#       {"index": 3, "sentence": "Now let's talk about renewable energy."}
#     ]

#     Explanation:
#     - The first object is included because the first sentence should always be considered the start of a topic.  
#     - The second object is included because the phrase "Now let's talk about..." signals a topic change.  
#     """

    PROMPT = """
    You are a podcast host, and you have recorded an episode that you want to split into distinct topic-based segments.

    You are given an array of sentences representing the transcript of the episode. Your task is to identify the starting point of each new topic and return the indexes of the sentences where new topics begin.

    Guidelines:
    - The first sentence should always be considered the start of a topic.  
    - A new topic may begin when there is a noticeable shift in context, such as phrases like "Now let's talk about..." or a clear change in subject matter.  
    - You cannot modify the sentences — only determine the starting points of new topics.  

    ### Output Requirements:
    - The output should be a valid JSON array of objects.  
    - Each object should have two keys:
      - `"index"` — the integer index of the starting sentence  
      - `"sentence"` — the sentence itself  
    - **Keys and strings must be enclosed in double quotes** (not single quotes).  
    - The output must not contain trailing commas or invalid characters.  

    ### Example:
    Input:
    [
      "Welcome to the show!",
      "Today we will discuss climate change.",
      "Global temperatures have risen significantly in the last decade.",
      "Now let's talk about renewable energy.",
      "Solar and wind power are becoming more affordable."
    ]
    Output:
    [
      {"index": 0, "sentence": "Welcome to the show!"},
      {"index": 3, "sentence": "Now let's talk about renewable energy."}
    ]

    ### Correct JSON Format Example:
    [
      {"index": 0, "sentence": "Welcome to the show!"},
      {"index": 3, "sentence": "Now let's talk about renewable energy."}
    ]"""


    response = client.chat.completions.create(
        messages=[
            {'role': 'system', 'content': PROMPT},
            {'role': 'user', 'content': json.dumps(sentences, ensure_ascii=False)}
        ],
        # model='google/gemini-2.0-flash-lite-001',
        temperature=1,
        model=model,
        max_tokens=max_tokens,
        response_format={
            'type': 'json_schema',
            'json_schema': {
                'name': 'sentences',
                'strict': True,
                'schema': {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "sentences": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "index": {
                                        "type": "integer",
                                        "description": "Sentence index."
                                    },
                                    "text": {
                                        "type": "string",
                                        "description": "The sentence."
                                    }
                                },
                                "required": ["index", "text"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["sentences"],
                    "additionalProperties": False
                },
            },
        },
    )
    
    try:
        content = response.choices[0].message.content
        jcontent = json.loads(content)
        return jcontent.get('sentences', {})
    except Exception as e:
        print(f'{response=}')
        raise e    


# %%
# expected = 303
# actual = [76, 205, 355, 363, 372, 434, 437, 459, 482, 571, 576, 579, 585, 599, 607, 616, 624, 813, 836, 863, 916, 919, 1080, 1088, 1103, 1113, 1135]
# previous = 200

# assert find_nearest(actual, expected, previous) == 355, f'{find_nearest(actual, expected, previous)} != 355'

# %%
# ground_truth = ''.join(map(str, df['is_first_sentence_in_topic'].values))

# %%
def get_segments(sentences: list[str], delimiters: list[int]) -> list[list[str]]:
    if len(delimiters) > 1:
        borders = [{'start': delimiters[i], 'end': delimiters[i+1]} for i in range(0, len(delimiters)-1)]
        borders.append({'start': borders[-1]['end'], 'end': len(sentences)})
    else:
        borders = [{'start': 0, 'end': len(sentences)}]
    result = [sentences[x['start']:x['end']] for x in borders]
    return result

    
sentences = ['a', 'b', 'c', 'd']

delimiters = [0, 1]
assert get_segments(sentences, delimiters) == [['a'], ['b', 'c', 'd']]

delimiters = [0, 3]
assert get_segments(sentences, delimiters) == [['a','b', 'c'], ['d']]

delimiters = [0]
assert get_segments(sentences, delimiters) == [['a','b', 'c', 'd']]

# %%
result_by_episode = {}

# %%
for i, episode in enumerate(sorted(df['episode'].unique()), start=1):
    try:
        sentences = df[df['episode'] == episode][['ru_sentence', 'ground_truth', 'en_sentence']].values
        episode_result = {}

        if episode in result_by_episode and 'llm_response' in result_by_episode[episode] and result_by_episode[episode]['llm_response'] != {}:
            llm_response = result_by_episode[episode]['llm_response']
        else:
            llm_response = call_llm(
                model='google/gemini-2.0-flash-lite-001',
                sentences=[x[0] for x in sentences],
                max_tokens=8_000,
                temperature=1
            )
        episode_result['llm_response'] = llm_response
        
        predicted_delimiters = get_delimiters(response=llm_response, sentences=[x[0] for x in sentences])
        episode_result['predicted_delimiters'] = predicted_delimiters

        predicted = np.zeros(len(sentences), np.int8)
        predicted[predicted_delimiters] = 1
        predicted = ''.join(map(str, predicted))
        episode_result['predicted'] = predicted
        
        ground_truth = ''.join(map(str, df[df['episode'] == episode]['is_first_sentence_in_topic'].values))
        episode_result['ground_truth'] = ground_truth

        windiff = windowdiff(ground_truth, predicted, 1)
        episode_result['windiff'] = windiff
    except Exception as e:
        print(episode, e)
    finally:
        result_by_episode[episode] = episode_result
        print(f'{i}/{df["episode"].nunique()}: {episode_result.get("windiff", "n/a")}')

# %%
[x for x in [v.get('windiff', -1) for k, v in result_by_episode.items()] if x > 0]

# %%
[v.get('windiff', -1) for k, v in result_by_episode.items()]

# %%
episode = 401
llm_response = result_by_episode[episode]['llm_response']
sentences = df[df['episode'] == episode][['ru_sentence', 'ground_truth', 'en_sentence']].values
predicted_delimiters = get_delimiters(response=llm_response, sentences=[x[0] for x in sentences])
predicted = np.zeros(len(sentences), np.int8)
predicted[predicted_delimiters] = 1
predicted = ''.join(map(str, predicted))
ground_truth = ''.join(map(str, df[df['episode'] == episode]['is_first_sentence_in_topic'].values))
windiff = windowdiff(ground_truth, predicted, 1)

segments = get_segments(sentences[:, 0], predicted_delimiters)

windiff, len(segments)

# %% [markdown]
# ## Information density

# %%
import nltk
from nltk import word_tokenize
import pymorphy3

# %%
nltk.download('punkt')

# %%
morph = pymorphy3.MorphAnalyzer()

def information_density(text):
    words = word_tokenize(text, language="russian")
    content_words = 0
    total_words = len(words)

    for word in words:
        parsed = morph.parse(word)[0]
        pos = parsed.tag.POS
        # Content words: Nouns (NOUN), Verbs (VERB), Adjectives (ADJF, ADJS), Adverbs (ADVB)
        if pos in {"NOUN", "VERB", "ADJF", "ADJS", "ADVB"}:
            content_words += 1
    
    density = content_words / total_words if total_words > 0 else 0
    return density

# Example
text = "Быстрая рыжая лиса перепрыгнула через ленивую собаку."
score = information_density(text)
print(f"Information Density: {score:.2f}")

# %%
for i, segment in enumerate(segments, start=1):
    print(f'{i}. [{information_density(" ".join(segment))}] {" ".join(segment)[:min(1_500, len(" ".join(segment)))]}')

# %% [markdown]
# ## Keywords

# %%
topics = [' '.join(s) for s in segments]
topics[:3]

# %%
stop_words = set(get_stop_words('ru'))

def tokenize(text: str) -> list[str]:
    def _get_normal_form_or_none(word: str, verbose=False) -> bool:
        parsed = morph.parse(word)[0]

        pos = parsed.tag.POS
        if pos is not None and pos != 'NOUN':
            if verbose: print(f'1. {word} -> pos is not None and not NOUN')
            return None

        normal_form = parsed.normal_form
        if not normal_form.isalnum():
            if verbose: print(f'2. {word} -> {normal_form.isalnum()=}')
            return None
        
        if word in stop_words or normal_form in stop_words:
            return None

        if verbose: print(f'3. {word} -> {normal_form=}')
        return normal_form
    
    words = word_tokenize(text, language='russian')
    result = []
    for word in words:
        norm_form = _get_normal_form_or_none(word, word == 'Canonical')
        if norm_form is None:
            continue
        result.append(norm_form)

    return result


# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
topics_tokenized = [' '.join(tokenize(t)) for t in topics]
topics_tokenized = topics
transformer = TfidfVectorizer(
    max_features=200,
    preprocessor=lambda x: ' '.join(tokenize(x)))
topics_tokenized_tranformed = transformer.fit_transform(topics_tokenized)

# %%
transformer.get_feature_names_out()

# %%

# %%
for n in range(10):
    print(f'Topic: "{topics[n]}"\n')
    print(f'Keywords: {transformer.get_feature_names_out()[np.array(topics_tokenized_tranformed[n].todense() > 0)[0]]}')
    print('=' * 100)

# %%
# get_stop_words('ru')
words = word_tokenize(topics[0], language="russian")
words = [x for x in words if x not in set(get_stop_words('ru'))]
content_words = 0
for word in words:
    parsed = morph.parse(word)[0]
    pos = parsed.tag.POS
    normalized_word = parsed.normal_form
    # Content words: Nouns (NOUN), Verbs (VERB), Adjectives (ADJF, ADJS), Adverbs (ADVB)
    # print(pos)
    print(f'{str(pos):<7}{normalized_word:<20}{word}')
    if pos in {"NOUN", "VERB", "ADJF", "ADJS", "ADVB"}:
        content_words += 1
