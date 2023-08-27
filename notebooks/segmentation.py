# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Segmentation

# %% [markdown]
# ## Intro

# %% [markdown]
# In recent years, the internet has seen the emergence of numerous new podcasts. Some, like [The Joe Rogan Experience](https://open.spotify.com/show/4rOoJ6Egrf8K2IrywzwOMk), have gained immense popularity, while many others remain relatively unknown.
#
# As a devoted podcast enthusiast, I am subscribed to more than 40 podcasts. Unfortunately, a considerable number of these podcasts lack proper timestamps for their episodes. A similar situation exists on YouTube, where a significant proportion of videos lack timestamps for their various sections.
#
# This challenge has motivated me to embark on a new project that combines my passion for podcasts with my interest in machine learning. Operating under the working title of "automatic timestamps generator," the primary goal of this project is to automatically segment podcast episodes into distinct topics and generate corresponding titles for each segment.
#
# For this endeavor, I will use the Russian podcast DevZen, which delves into topics related to technology, software engineering, management, and databases.

# %% [markdown]
# ## Approaches

# %% [markdown]
# Various methods exist for text segmentation:
#
# 1. **Supervised Learning:**
#    1. One approach involves determining whether a sentence is a boundary sentence. This can be done by creating an embedding for each sentence and training a binary classification model using these embeddings.
#
# 2. **Unsupervised Learning:**
#    1. [TextTiling](https://aclanthology.org/J97-1003.pdf), introduced by Marti A. Hearst in 1997, presents two scoring techniques for evaluating potential segment divisions: the block-comparison method and the vocabulary introduction method. The algorithm involves dividing the text into equally sized groups of pseudosentences and computing a score for each pair of groups.
#    2. [LCseg](https://aclanthology.org/P03-1071.pdf) is an algorithm that works with both text and speech. Its core algorithm consists of two main components: a method for identifying and weighing strong term repetitions using lexical chains, and a method for hypothesizing topic boundaries based on simultaneous chains of term repetitions. For more information about this algorithm, you can read the paper "Discourse Segmentation of Multi-Party Conversation."
#    3. [TopicTiling](https://aclanthology.org/W12-3307.pdf) is a variant of the TextTiling algorithm that uses the Latent Dirichlet Allocation (LDA) topic model to split documents. It's faster than other LDA-based methods, but it's specific to certain topics and requires separate models for each subject. Since I plan to apply text segmentation to diverse podcasts covering technology, software engineering, movies, and music, this algorithm might not be the best fit for my project.
#    4. [GraphSeg](https://aclanthology.org/S16-2016.pdf) attempts to construct a graph where each node represents a sentence, and edges connect two semantically related sentences. The algorithm then identifies all maximal cliques and merges them into segments.
#    5. Another method is segmentation based on cosine similarity. [Unsupervised Topic Segmentation of Meetings with BERT Embeddings](https://arxiv.org/pdf/2106.12978.pdf) article describes one of the possible options. The paper introduces an approach based on BERT embeddings. This algorithm is a modified version of the original TextTiling and detects topic changes based on a similarity score using BERT embeddings. The authors provide a [reference implementation for the algorithm on their GitHub](https://github.com/gdamaskinos/unsupervised_topic_segmentation).

# %% [markdown]
# In this article, I will be conducting a comparison of several of these approaches using my own dataset: podcast episodes that have been transcribed and segmented into individual sentences.
#
# Unfortunately I couldn't find an existing Python implementation for each algorithm above. So I'll test only few of them:
# 1. **TextTiling**: the algorithm implementation could be found in the [NLTK](https://www.nltk.org/api/nltk.tokenize.texttiling.html#nltk.tokenize.TextTilingTokenizer)
# 2. **LCSeg**: the only one implementation I've found is [written on Elixir](https://github.com/oliverswitzer/lc_seg/tree/main). I've spent almost 4 hours trying to start the app on Ubuntu without any success
# 3. **TopicTiling**: I don't even want to test because of it's subject limitations I've wrote earlier
# 4. **GraphSeg**: there is at least one Python implementation &mdash; [`graphseg-python`](https://github.com/Dobatymo/graphseg-python). 
# 5. For **"Unsupervised Topic Segmentation of Meetings with BERT Embeddings"** there are two Python implementation: from paper's authors and in ConversePy library
#
# As an experiment I'll build segmentation for multiple episodes using **NLTK**'s **TextTiling**, **graphseg-python** and **ConversePy**.

# %% [markdown]
# # Segmentation methods comparison

# %% [markdown]
# ## GraphSeg
#
# As a GraphSeg implementation I've choosen a library named `graphseg-python`. Sadly there is no PyPI module and I've spend couple of hours trying to segment my test texts with this bunch of scripts. Unfortunately the best what I've achieved is that the script ran on my laptop in a separate virtual environment.
#
# So I've randomly selected 412th episode to check the best `min_seg` parameter for GraphSeg algorith. Without diving into details `min_seg` is responisble for the minimal possible topic size. More about GraphSeg parameters you can find in the original paper ["Unsupervised Text Segmentation Using Semantic Relatedness Graphs"](https://aclanthology.org/S16-2016.pdf) by Goran Glavas, Federico Nanni, and Simone Paolo Ponzetto.
#
# I've tried six parameter's values: 3, 6, 12, 24, 48, and 96 and two languages: Russian (original) and English (translated with machine translation without any editing).

# %%
from nltk.metrics import windowdiff, pk
import pandas as pd


def calc_windowdiff(lang: str) -> None:
    df = pd.read_csv('../data/412_ep_reference.csv')
    sentence_column = 'sentence' if lang == 'ru' else 'en_translation'
    df = df[[sentence_column, 'ground_truth']].groupby('ground_truth').agg(topic=(sentence_column, lambda x: ''.join(x)))
    df['topic'] = df['topic'].apply(lambda x: '|' + x[1:])
    ground_truth = ''.join(df['topic']).replace(' ', '')

    k = int(round(len(ground_truth) / (ground_truth.count('|') * 2.)))

    for min_seg in [3, 6, 12, 24, 48, 96]:
        graph_seg = open(f'../data/412_episode_lang={lang}_min_seg={min_seg}_segments.txt', 'r', encoding='utf8').read()
        graph_seg = graph_seg.replace(' ', '')
        # for graphseg ground truth and actual segmentation should have the same lengths
        assert len(ground_truth) == len(graph_seg)

        win_diff = windowdiff(ground_truth, graph_seg, boundary="|", k=k)
        print(f'{lang=}\t{min_seg=}\t{k=}\t{win_diff=:.4f}')

calc_windowdiff('ru')
print()
calc_windowdiff('en')
