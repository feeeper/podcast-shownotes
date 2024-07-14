# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
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
# In this article, I will compare several of these methods using my own dataset: podcast episodes that have been transcribed and segmented into individual sentences.
#
# However, I couldn't find existing Python implementations for all the algorithms mentioned above. Therefore, I will only test a few of them:
#
# 1. **TextTiling**: The algorithm's implementation can be found in the [NLTK library](https://www.nltk.org/api/nltk.tokenize.texttiling.html#nltk.tokenize.TextTilingTokenizer).
# 2. **LCSeg**: I found only one implementation written in Elixir, but I had difficulty running it on Ubuntu.
# 3. **TopicTiling**: Due to its limitations for specific subjects, I won't be testing this algorithm.
# 4. **GraphSeg**: There is at least one Python implementation available, called [`graphseg-python`](https://github.com/Dobatymo/graphseg-python).
# 5. For the **"Unsupervised Topic Segmentation of Meetings with BERT Embeddings"**, there are two Python implementations: one from the paper's authors and another in the [pyconverse](https://github.com/maxent-ai/converse) library.
#
# As an experiment, I will build segmentations for multiple podcast episodes using **NLTK**'s **TextTiling**, **graphseg-python**, and the **pyconverse** library.

# %% [markdown]
# # Segmentation methods comparison

# %% [markdown]
# All algorithms I'll test on randomly selected 412th episode to find the best params (if there are any parameter exist for algorithm).

# %% [markdown]
# ## TextTiling
#
# Let's start from TextTiling from NLTK.
#
# The `TextTilingTokenizer` implements TextTiling segmentation. Sadly the implementation has only block-comparison method.
#
# TextTiling has two main parameters: pseudosentence size `w` and size of the block for the block-comparison method `k`. I'll test multiple combination of these two parameters.

# %%
import math
import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from stop_words import get_stop_words

import nltk
from nltk.metrics import windowdiff
from nltk.tokenize import texttiling
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.texttiling import TokenSequence, TokenTableField, smooth
nltk.download('stopwords')

from pyconverse import SemanticTextSegmention

# %%
# df = pd.read_csv('../data/412_ep_reference.csv')
df = pd.read_csv('/mnt/c/Users/andrei/Downloads/400-415-episodes-ground-truth.csv')
df = df[df['episode'] == 412]
df


# %%
def text_tiling_windowdiff(df: pd.DataFrame, lang: str, text_tiling_tokenizer: TokenizerI, verbose: bool=False, **kwargs) -> None:    
    topic_df = df[[f'{lang}_sentence', 'ground_truth']].groupby('ground_truth').agg(topic=(f'{lang}_sentence', lambda x: '\n\n\t'.join(x)))
    source_text = ' '.join(topic_df['topic'])
    ground_truth = ''.join(topic_df['topic'].apply(lambda x: '|' + x.replace('\n\n', '').replace(' ', '')[1:]))
        
    stop_words = get_stop_words(lang)
    
    k = int(round(len(ground_truth) / (ground_truth.count('|') * 2.)))

    metrics = []
    segmentations = {}
    sent_sizes = [20, 40, 80, 100, 200]
    block_sizes = [10, 20, 40, 80, 100]
    t = tqdm(total=len(sent_sizes) * len(block_sizes), disable=not verbose)

    for sent_size in sent_sizes:
        for block_size in block_sizes:
            ttt = text_tiling_tokenizer(stopwords=stop_words, w=sent_size, k=block_size)
            topics = ttt.tokenize(source_text)
            actual = ''.join(['|' + topic.replace('\n', '').replace(' ', '')[1:] for topic in topics])

            assert len(ground_truth) == len(actual)

            win_diff = windowdiff(ground_truth, actual, boundary="|", k=k)
            metrics.append({
                'algorithm': 'text_tiling',
                'lang': lang,
                'topics_count': len(topics),
                'k': k,
                'sent_size': sent_size,
                'block_size': block_size,
                'win_diff': win_diff,
            })
            segmentations[(sent_size, block_size)] = topics
            t.update(1)                

    return pd.DataFrame(metrics), segmentations


# %%
def find(s: str, ch: str):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def plot_segmentation(ground_truth_topics: list[str], actual_topics: list[str], title: str) -> None:
    source_text = re.sub('\s+', '', ''.join([f'|{x[1:]}' for x in ground_truth_topics]))
    actual_text = re.sub('\s+', '', ''.join([f'|{x[1:]}' for x in actual_topics]))
    
    source_idx = find(source_text, '|')
    actual_idx = find(actual_text, '|')
    
    idx_df = pd.DataFrame({
        'idx': source_idx + actual_idx,
        'source': ['ground_truth'] * len(source_idx) + ['actual'] * len(actual_idx),
        'y': [0.] * len(source_idx) + [0.] * len(actual_idx)
    })
    
    sns.set(rc={'figure.figsize':(13.7,2.5)})
    sns.scatterplot(data=idx_df, x="idx", y="y", hue="source", style='source', sizes=(80, 400), s=120)
    plt.legend(title=title, fontsize='9', title_fontsize='10')


# %% [markdown]
# ### Russian text segmentation

# %%
ru_text_tiling_metrics_df, ru_text_tiling_segments = text_tiling_windowdiff(df, 'ru', texttiling.TextTilingTokenizer, verbose=True)
ru_text_tiling_metrics_df

# %%
ru_ground_truth = df[['ru_sentence', 'ground_truth']].groupby('ground_truth').agg(topic=('ru_sentence', lambda x: ' '.join(x)))['topic'].values
en_ground_truth = df[['en_sentence', 'ground_truth']].groupby('ground_truth').agg(topic=('en_sentence', lambda x: ' '.join(x)))['topic'].values


# %%
def plot_segmentation_for_best_score(
    metrics_df: pd.DataFrame,
    segmentation: dict,
    algorithm_params: list[str],
    ground_truth: list[str],
    title: str)-> None:
    min_win_diff_row = metrics_df[metrics_df['win_diff'] == metrics_df['win_diff'].min()]
    win_diff, topics_count, *params = min_win_diff_row[['win_diff', 'topics_count'] + algorithm_params].values[0]

    plot_segmentation(ground_truth, segmentation[tuple(params)], title=f'{title} (best params: {tuple(params)})')


# %%
plot_segmentation_for_best_score(ru_text_tiling_metrics_df, ru_text_tiling_segments, ['sent_size', 'block_size'], ru_ground_truth, title='TextTiling')

# %% [markdown]
# There are two strange things about the metric when applied to Russian text. Firstly, the WindowDiff score changes only with different sent_size settings and doesn't consider block_size. Secondly, the best WindowDiff score is achieved when the text is not segmented at all. It seems like this implementation doesn't work well for non-English texts.
#
# So, this is the reason to take a closer look at the TextTiling implementation in the NLTK library. The code in the `TextTilingTokenizer` class is quite understandable, and the main area of concern lies in [lines 94-97 of the `tokenize` method](https://github.com/nltk/nltk/blob/582e6e35f0e6c984b44ec49dcb8846d9c011d0a8/nltk/tokenize/texttiling.py#L95-L97):
#
# ``` python
# # Remove punctuation
# nopunct_text = "".join(
#     c for c in lowercase_text if re.match(r"[a-z\-' \n\t]", c)
# )
# ```
#
# As you can see, all non-Latin letters are removed during this preprocessing step. Unfortunately, this fact renders the `TextTilingTokenizer` ineffective for all non-Latin languages, including Cyrillic languages like Russian.
#
# The simplest way to "fix" this issue is to duplicate the class and modify this preprocessing step accordingly.

# %%
BLOCK_COMPARISON, VOCABULARY_INTRODUCTION = 0, 1
LC, HC = 0, 1
DEFAULT_SMOOTHING = [0]


class TextTilingTokenizerExt(TokenizerI):
    def __init__(self,
                 w=20,
                 k=10,
                 similarity_method=BLOCK_COMPARISON,
                 stopwords=None,
                 smoothing_method=DEFAULT_SMOOTHING,
                 smoothing_width=2,
                 smoothing_rounds=1,
                 cutoff_policy=HC,
                 demo_mode=False):


        if stopwords is None:
            from nltk.corpus import stopwords
            stopwords = stopwords.words('english')
        self.__dict__.update(locals())
        del self.__dict__['self']


    def tokenize(self, text):
        """Return a tokenized copy of *text*, where each "token" represents
        a separate topic."""

        lowercase_text = text.lower()
        paragraph_breaks = self._mark_paragraph_breaks(text)
        text_length = len(lowercase_text)

        # Tokenization step starts here
        # Remove punctuation
        nopunct_text = "".join(
            c for c in lowercase_text if re.match(r"[\w\-' \n\t]", c)
        )

        nopunct_par_breaks = self._mark_paragraph_breaks(nopunct_text)

        tokseqs = self._divide_to_tokensequences(nopunct_text)

        # The morphological stemming step mentioned in the TextTile
        # paper is not implemented.  A comment in the original C
        # implementation states that it offers no benefit to the
        # process. It might be interesting to test the existing
        # stemmers though.
        #words = _stem_words(words)

        # Filter stopwords
        for ts in tokseqs:
            ts.wrdindex_list = [wi for wi in ts.wrdindex_list
                                if wi[0] not in self.stopwords]

        token_table = self._create_token_table(tokseqs, nopunct_par_breaks)
        # End of the Tokenization step

        # Lexical score determination
        if self.similarity_method == BLOCK_COMPARISON:
            gap_scores = self._block_comparison(tokseqs, token_table)
        elif self.similarity_method == VOCABULARY_INTRODUCTION:
            raise NotImplementedError("Vocabulary introduction not implemented")

        if self.smoothing_method == DEFAULT_SMOOTHING:
            smooth_scores = self._smooth_scores(gap_scores)
        # End of Lexical score Determination

        # Boundary identification
        depth_scores = self._depth_scores(smooth_scores)
        segment_boundaries = self._identify_boundaries(depth_scores)

        normalized_boundaries = self._normalize_boundaries(text,
                                                           segment_boundaries,
                                                           paragraph_breaks)
        # End of Boundary Identification
        segmented_text = []
        prevb = 0

        for b in normalized_boundaries:
            if b == 0:
                continue
            segmented_text.append(text[prevb:b])
            prevb = b

        if prevb < text_length: # append any text that may be remaining
            segmented_text.append(text[prevb:])

        if not segmented_text:
            segmented_text = [text]

        if self.demo_mode:
            return gap_scores, smooth_scores, depth_scores, segment_boundaries
        return segmented_text


    def _block_comparison(self, tokseqs, token_table):
        "Implements the block comparison method"
        def blk_frq(tok, block):
            ts_occs = filter(lambda o: o[0] in block,
                             token_table[tok].ts_occurences)
            freq = sum([tsocc[1] for tsocc in ts_occs])
            return freq

        gap_scores = []
        numgaps = len(tokseqs)-1

        for curr_gap in range(numgaps):
            score_dividend, score_divisor_b1, score_divisor_b2 = 0.0, 0.0, 0.0
            score = 0.0
            #adjust window size for boundary conditions
            if curr_gap < self.k-1:
                window_size = curr_gap + 1
            elif curr_gap > numgaps-self.k:
                window_size = numgaps - curr_gap
            else:
                window_size = self.k

            b1 = [ts.index
                  for ts in tokseqs[curr_gap-window_size+1 : curr_gap+1]]
            b2 = [ts.index
                  for ts in tokseqs[curr_gap+1 : curr_gap+window_size+1]]

            for t in token_table:
                score_dividend += blk_frq(t, b1)*blk_frq(t, b2)
                score_divisor_b1 += blk_frq(t, b1)**2
                score_divisor_b2 += blk_frq(t, b2)**2
            try:
                score = score_dividend/math.sqrt(score_divisor_b1*
                                                 score_divisor_b2)
            except ZeroDivisionError:
                pass # score += 0.0

            gap_scores.append(score)

        return gap_scores

    def _smooth_scores(self, gap_scores):
        "Wraps the smooth function from the SciPy Cookbook"
        return list(smooth(np.array(gap_scores[:]),
                           window_len = self.smoothing_width+1))

    def _mark_paragraph_breaks(self, text):
        """Identifies indented text or line breaks as the beginning of
        paragraphs"""

        MIN_PARAGRAPH = 100
        pattern = re.compile("[ \t\r\f\v]*\n[ \t\r\f\v]*\n[ \t\r\f\v]*")
        matches = pattern.finditer(text)

        last_break = 0
        pbreaks = [0]
        for pb in matches:
            if pb.start()-last_break < MIN_PARAGRAPH:
                continue
            else:
                pbreaks.append(pb.start())
                last_break = pb.start()

        return pbreaks

    def _divide_to_tokensequences(self, text):
        "Divides the text into pseudosentences of fixed size"
        w = self.w
        wrdindex_list = []
        matches = re.finditer("\w+", text)
        for match in matches:
            wrdindex_list.append((match.group(), match.start()))
        return [TokenSequence(i/w, wrdindex_list[i:i+w])
                for i in range(0, len(wrdindex_list), w)]

    def _create_token_table(self, token_sequences, par_breaks):
        "Creates a table of TokenTableFields"
        token_table = {}
        current_par = 0
        current_tok_seq = 0
        pb_iter = par_breaks.__iter__()
        current_par_break = next(pb_iter)
        if current_par_break == 0:
            try:
                current_par_break = next(pb_iter) #skip break at 0
            except StopIteration:
                raise ValueError(
                    "No paragraph breaks were found(text too short perhaps?)"
                    )
        for ts in token_sequences:
            for word, index in ts.wrdindex_list:
                try:
                    while index > current_par_break:
                        current_par_break = next(pb_iter)
                        current_par += 1
                except StopIteration:
                    #hit bottom
                    pass

                if word in token_table:
                    token_table[word].total_count += 1

                    if token_table[word].last_par != current_par:
                        token_table[word].last_par = current_par
                        token_table[word].par_count += 1

                    if token_table[word].last_tok_seq != current_tok_seq:
                        token_table[word].last_tok_seq = current_tok_seq
                        token_table[word]\
                                .ts_occurences.append([current_tok_seq,1])
                    else:
                        token_table[word].ts_occurences[-1][1] += 1
                else: #new word
                    token_table[word] = TokenTableField(first_pos=index,
                                                        ts_occurences= \
                                                          [[current_tok_seq,1]],
                                                        total_count=1,
                                                        par_count=1,
                                                        last_par=current_par,
                                                        last_tok_seq= \
                                                          current_tok_seq)

            current_tok_seq += 1

        return token_table

    def _identify_boundaries(self, depth_scores):
        """Identifies boundaries at the peaks of similarity score
        differences"""

        boundaries = [0 for x in depth_scores]

        avg = sum(depth_scores)/len(depth_scores)
        stdev = np.std(depth_scores)

        #SB: what is the purpose of this conditional?
        if self.cutoff_policy == LC:
            cutoff = avg-stdev/2.0
        else:
            cutoff = avg-stdev/2.0

        depth_tuples = sorted(zip(depth_scores, range(len(depth_scores))))
        depth_tuples.reverse()
        hp = list(filter(lambda x:x[0]>cutoff, depth_tuples))

        for dt in hp:
            boundaries[dt[1]] = 1
            for dt2 in hp: #undo if there is a boundary close already
                if dt[1] != dt2[1] and abs(dt2[1]-dt[1]) < 4 \
                       and boundaries[dt2[1]] == 1:
                    boundaries[dt[1]] = 0
        return boundaries

    def _depth_scores(self, scores):
        """Calculates the depth of each gap, i.e. the average difference
        between the left and right peaks and the gap's score"""

        depth_scores = [0 for x in scores]
        #clip boundaries: this holds on the rule of thumb(my thumb)
        #that a section shouldn't be smaller than at least 2
        #pseudosentences for small texts and around 5 for larger ones.
        clip = min(max(int(len(scores)/10), 2), 5)
        index = clip

        for gapscore in scores[clip:-clip]:
            lpeak = gapscore
            for score in scores[index::-1]:
                if score >= lpeak:
                    lpeak = score
                else:
                    break
            rpeak = gapscore
            for score in scores[index:]:
                if score >= rpeak:
                    rpeak = score
                else:
                    break
            depth_scores[index] = lpeak + rpeak - 2 * gapscore
            index += 1

        return depth_scores

    def _normalize_boundaries(self, text, boundaries, paragraph_breaks):
        """Normalize the boundaries identified to the original text's
        paragraph breaks"""

        norm_boundaries = []
        char_count, word_count, gaps_seen = 0, 0, 0
        seen_word = False

        for char in text:
            char_count += 1
            if char in " \t\n" and seen_word:
                seen_word = False
                word_count += 1
            if char not in " \t\n" and not seen_word:
                seen_word=True
            if gaps_seen < len(boundaries) and word_count > \
                                               (max(gaps_seen*self.w, self.w)):
                if boundaries[gaps_seen] == 1:
                    #find closest paragraph break
                    best_fit = len(text)
                    for br in paragraph_breaks:
                        if best_fit > abs(br-char_count):
                            best_fit = abs(br-char_count)
                            bestbr = br
                        else:
                            break
                    if bestbr not in norm_boundaries: #avoid duplicates
                        norm_boundaries.append(bestbr)
                gaps_seen += 1

        return norm_boundaries

# %%
ru_text_tiling_ext_metrics_df, ru_text_tiling_ext_segments = text_tiling_windowdiff(df, 'ru', TextTilingTokenizerExt, verbose=True)
ru_text_tiling_ext_metrics_df

# %%
plot_segmentation_for_best_score(ru_text_tiling_ext_metrics_df, ru_text_tiling_ext_segments, ['sent_size', 'block_size'], ru_ground_truth, 'TextTiling Updated')


# %%
def print_segmentation(segmentation: list[str], limit: int=100) -> None:
    def _get_words(segment_text: str) -> list[str]:
        return segment_text.replace('\n\n', ' ').split()
    
    def _limit_by_words(segment_words: list[str]) -> str:
        words = segment_words[:limit//2] + [' [...] '] + segment_words[-limit//2:]
        return ' '.join(words)

    for segment in segmentation:
        print(_limit_by_words(_get_words(segment)), end='\n\n')


# %%
print_segmentation(ru_text_tiling_ext_segments[(200, 20)], limit=130)

# %% [markdown]
# Alright, now it appears much improved. The algorithm has successfully divided the text into more than one segment. Furthermore, this segmentation seems quite impressive, as the original segmentation was at a high level, but TextTiling has extracted more detailed segments.

# %% [markdown]
# ### English text segmentation

# %%
en_text_tiling_metrics, en_text_tiling_segments = text_tiling_windowdiff(df, 'en', text_tiling_tokenizer=texttiling.TextTilingTokenizer, verbose=True)
en_text_tiling_metrics_df = pd.DataFrame(en_text_tiling_metrics)
en_text_tiling_metrics_df

# %% [markdown]
# The most disappointing aspect is that the original segmentation had far fewer topics. Nonetheless, it would be interesting to explore the topics we obtain with the best parameters:

# %%
plot_segmentation_for_best_score(en_text_tiling_metrics_df, en_text_tiling_segments, ['sent_size', 'block_size'], en_ground_truth, title='TextTiling Google Translated')

# %%
print_segmentation(en_text_tiling_segments[(200, 40)], limit=130)


# %% [markdown]
# ## GraphSeg
#
# For the implementation of GraphSeg, I chose a library called `graphseg-python`. Unfortunately, there's no PyPI module available for it, and I spent several hours attempting to segment my test texts using the provided scripts. However, I was only able to get the script to run on my laptop within a separate virtual environment.
#
# To proceed, I randomly selected the 412th episode to determine the optimal `min_seg` parameter for the GraphSeg algorithm. Briefly, the `min_seg` parameter is responsible for determining the minimum possible topic size. You can find more details about GraphSeg parameters in the original paper titled ["Unsupervised Text Segmentation Using Semantic Relatedness Graphs"](https://aclanthology.org/S16-2016.pdf) by Goran Glavas, Federico Nanni, and Simone Paolo Ponzetto.
#
# I experimented with six different values for the parameter: 3, 6, 12, 24, 48, and 96. Additionally, I tested the algorithm on two languages: Russian (original) and English (translated using machine translation without any manual editing).
#
# You can find the code and the resulting output below.

# %%
def graph_seg_windowdiff(df: pd.DataFrame, lang: str, verbose: bool = False, **kwargs) -> None:
    episode = kwargs['episode']
    topic_df = df[[f'{lang}_sentence', 'ground_truth']].groupby('ground_truth').agg(topic=(f'{lang}_sentence', lambda x: ''.join(x)))
    topic_df['topic'] = topic_df['topic'].apply(lambda x: '|' + x[1:])
    ground_truth = ''.join(topic_df['topic']).replace(' ', '')

    # default k value for windowdiff
    k = int(round(len(ground_truth) / (ground_truth.count('|') * 2.)))

    metrics = []
    all_topics = {}
    for min_seg in tqdm([3, 6, 12, 24, 48, 96], disable=not verbose):
        # files {episode}_episode_lang={lang}_min_seg={min_seg}_segments were built separately.
        actual = open(f'../data/{episode}_ep_reference_lang={lang}_min_seg={min_seg}_segments.txt', 'r', encoding='utf8').read().strip()
        topics = actual.split('\n\n')
        actual = ''.join(['|' + t.replace(' ', '').replace('\n', '')[1:] for t in topics])
        # for windowdiff ground truth and actual segmentation should have the same lengths        
        assert len(ground_truth) == len(actual), f'{ground_truth[:200]=}\n{actual[:200]}'

        win_diff = windowdiff(ground_truth, actual, boundary="|", k=k)
        metrics.append({
            'algorithm': 'graph_seg',
            'lang': lang,
            'k': k,
            'min_seg': min_seg,
            'win_diff': win_diff,
            'topics_count': len(topics)
        })
        all_topics[(min_seg,)] = topics
    return pd.DataFrame(metrics), all_topics


# %%
ru_graph_seg_metrics_df, ru_segments_graph_seg = graph_seg_windowdiff(df, 'ru', episode=412)
ru_graph_seg_metrics_df

# %%
plot_segmentation_for_best_score(ru_graph_seg_metrics_df, ru_segments_graph_seg, ['min_seg'], ru_ground_truth, title='GraphSeg')

# %%
print_segmentation(ru_segments_graph_seg[(48,)], limit=130)

# %%
en_graph_seg_metrics_df, en_graph_seg_segments = graph_seg_windowdiff(df, 'en', episode=412, verbose=True)
en_graph_seg_metrics_df

# %%
plot_segmentation_for_best_score(en_graph_seg_metrics_df, en_graph_seg_segments, ['min_seg'], en_ground_truth, title='GraphSeg Google Tranaslated')

# %%
print_segmentation(en_graph_seg_segments[(48,)])


# %% [markdown] toc-hr-collapsed=true
# ## Unsupervised Topic Segmentation of Meetings with BERT Embeddings

# %% [markdown]
# The final method I'm exploring is unsupervised segmentation using BERT embeddings. This approach can encompass various techniques, such as default next sentence prediction-based methods or even K-nearest neighbor approaches built on BERT embeddings.
#
# For this experiment, I'm using an implementation from `pyconverse`. The library's maintainer utilizes NLTK's `TextTilingTokenizer` to implement the algorithm described in the research paper. I'm applying this method to segment my own data to compare it with the vanilla **TextTiling** and **GraphSeg** approaches.

# %%
def sts_windowdiff(df: pd.DataFrame, lang: str, semantic_text_segmentation_impl, verbose: bool = False, **kwargs) -> None:
    topic_df = df[[f'{lang}_sentence', 'ground_truth']].groupby('ground_truth').agg(topic=(f'{lang}_sentence', lambda x: ''.join(x)))
    topic_df['topic'] = topic_df['topic'].apply(lambda x: '|' + x[1:])
    ground_truth = ''.join(topic_df['topic']).replace(' ', '')
      
    # default k value for windowdiff
    k = int(round(len(ground_truth) / (ground_truth.count('|') * 2.)))

    all_topics = {}
    metrics = []
    for threshold in tqdm([x/100 for x in range(0, 100, 5)], disable=not verbose):
        topics = semantic_text_segmentation_impl.get_segments(threshold=threshold)
        actual = ''.join(['|' + topic.replace('\n', '').replace(' ', '')[1:] for topic in topics])
        actual = actual.replace(' ', '')

        assert len(ground_truth) == len(actual), f'{len(ground_truth)=} {len(actual)=} {ground_truth[:100]=} {actual[:100]=}'

        topics_count = len(topics)
        # for windowdiff ground truth and actual segmentation should have the same lengths
        assert len(ground_truth) == len(actual)

        win_diff = windowdiff(ground_truth, actual, boundary="|", k=k)
        metrics.append({
            'algorithm': 'semantic_text_segmentation',
            'lang': lang,
            'threshold': threshold,
            'win_diff': win_diff,
            'topics_count': topics_count
        })
        all_topics[(threshold,)] = topics
    return pd.DataFrame(metrics), all_topics


# %%
ru_sts_metrics_df, ru_sts_segments = sts_windowdiff(df, 'ru', SemanticTextSegmention(df, 'ru_sentence'), verbose=True)
ru_sts_metrics_df

# %% [markdown]
# The best results I obtained were for text that wasn't even split into segments. Let's examine the worst segmentation with a `threshold` set to 0.95:

# %%
plot_segmentation(ru_ground_truth, ru_sts_segments[(0.95,)], title='Semantic Text Segmentation (threshold=0.95)')

# %% [markdown]
# And what about best segmentation?

# %%
plot_segmentation_for_best_score(ru_sts_metrics_df, ru_sts_segments, ['threshold'], ru_ground_truth, title='Semantic Text Segmentation')

# %% [markdown]
# It appears to be quite poor but expected.
#
# As I mentioned previously, `pyconverse` utilizes NLTK's `TextTilingTokenizer` with similarity calculations based on BERT embeddings. Consequently, this implementation suffers from the same issue as the standard `TextTilingTokenizer`—it doesn't support non-Latin languages. The `TextTilingTokenizer.tokenize` method simply removes all non-Latin characters (such as Cyrillic) from the text.

# %% [markdown]
# Let's dive into [the SemanticTextSegmention sources](https://github.com/maxent-ai/converse/blob/35613b979f40942856a106f98b6ae26ffefc6df7/pyconverse/segmentation.py#L8) and see what can we do here:
#
# ``` python
# import attr
# import pandas as pd
# import numpy as np
# from .utils import load_sentence_transformer, load_spacy
# from nltk.tokenize.texttiling import TextTilingTokenizer
# from sklearn.metrics.pairwise import cosine_similarity
#
# model = load_sentence_transformer()
# nlp = load_spacy()
#
#
# @attr.s
# class SemanticTextSegmentation:
#
#     """
#     Segment a call transcript based on topics discussed in the call using
#     TextTilling with Sentence Similarity via sentence transformer.
#
#     Paramters
#     ---------
#     data: pd.Dataframe
#         Pass the trascript in the dataframe format
#
#     utterance: str
#         pass the column name which represent utterance in transcript dataframe
#
#     """
#
#     data = attr.ib()
#     utterance = attr.ib(default='utterance')
#
#     def __attrs_post_init__(self):
#         columns = self.data.columns.tolist()
#
#     def get_segments(self, threshold=0.7):
#         """
#         returns the transcript segments computed with texttiling and sentence-transformer.
#
#         Paramters
#         ---------
#         threshold: float
#             sentence similarity threshold. (used to merge the sentences into coherant segments)
#
#         Return
#         ------
#         new_segments: list
#             list of segments        
#         """
#         segments = self._text_tilling()
#         merge_index = self._merge_segments(segments, threshold)
#         new_segments = []
#         for i in merge_index:
#             seg = ' '.join([segments[_] for _ in i])
#             new_segments.append(seg)
#         return new_segments
#
#     def _merge_segments(self, segments, threshold):
#         segment_map = [0]
#         for index, (text1, text2) in enumerate(zip(segments[:-1], segments[1:])):
#             sim = self._get_similarity(text1, text2)
#             if sim >= threshold:
#                 segment_map.append(0)
#             else:
#                 segment_map.append(1)
#         return self._index_mapping(segment_map)
#
#     def _index_mapping(self, segment_map):
#         index_list = []
#         temp = []
#         for index, i in enumerate(segment_map):
#             if i == 1:
#                 index_list.append(temp)
#                 temp = [index]
#             else:
#                 temp.append(index)
#         index_list.append(temp)
#         return index_list
#
#     def _get_similarity(self, text1, text2):
#         sentence_1 = [i.text.strip()
#                       for i in nlp(text1).sents if len(i.text.split(' ')) > 1]
#         sentence_2 = [i.text.strip()
#                       for i in nlp(text2).sents if len(i.text.split(' ')) > 2]
#         embeding_1 = model.encode(sentence_1)
#         embeding_2 = model.encode(sentence_2)
#         embeding_1 = np.mean(embeding_1, axis=0).reshape(1, -1)
#         embeding_2 = np.mean(embeding_2, axis=0).reshape(1, -1)
#         
#         if np.any(np.isnan(embeding_1)) or np.any(np.isnan(embeding_2)):
#             return 1
#         
#         sim = cosine_similarity(embeding_1, embeding_2)
#         return sim
#
#     def _text_tilling(self):
#         tt = TextTilingTokenizer(w=15, k=10)
#         text = '\n\n\t'.join(self.data[self.utterance].tolist())
#         segment = tt.tokenize(text)
#         segment = [i.replace("\n\n\t", ' ') for i in segment]
#         return segment
# ```
#
# To extract this class from the library, only two modifications are required: overriding the `load_sentence_transformer` and `load_spacy` methods. Fortunately, these methods are quite straightforward:
#
# ``` python
# import spacy
# from sentence_transformers import SentenceTransformer
#
# def load_sentence_transformer(model_name='all-MiniLM-L6-v2'):
#     model = SentenceTransformer(model_name)
#     return model
#
#
# def load_spacy():
#     return spacy.load('en_core_web_sm')
# ```
#
# The primary issue for non-English texts is that the maintainer utilizes an English-only model, [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), for embeddings. My solution is to switch to a multilingual model, such as [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
#
# I've copied and pasted this class below and made some improvements. First of all I've inlined `load_sentence_transformer` and `load_spacy` methods. Secondly I've extended `__init__` method with add additional param &mdash; model name. And the last and I think one of the most valuable improvement is that I've replaced NLTK's `TextTilingTokenizer` to the one I've changed earlier &mdash; `TextTilingTokenizerExt`:

# %%
import attr
import pandas as pd
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class SemanticTextSegmentationMultilingual:
    """
    Segment a call transcript based on topics discussed in the call using
    TextTilling with Sentence Similarity via sentence transformer.

    Paramters
    ---------
    data: pd.Dataframe
        Pass the trascript in the dataframe format

    utterance: str
        pass the column name which represent utterance in transcript dataframe

    """

    def __init__(self, data, utterance, model):
        self.data = data
        self.utterance = utterance
        self.model = SentenceTransformer(model)
        self.nlp = spacy.load('en_core_web_sm')
    
    def __attrs_post_init__(self):
        columns = self.data.columns.tolist()

    def get_segments(self, threshold=0.7):
        """
        returns the transcript segments computed with texttiling and sentence-transformer.

        Paramters
        ---------
        threshold: float
            sentence similarity threshold. (used to merge the sentences into coherant segments)

        Return
        ------
        new_segments: list
            list of segments        
        """
        segments = self._text_tilling()
        merge_index = self._merge_segments(segments, threshold)
        new_segments = []
        for i in merge_index:
            seg = ' '.join([segments[_] for _ in i])
            new_segments.append(seg)
        return new_segments

    def _merge_segments(self, segments, threshold):
        segment_map = [0]
        for index, (text1, text2) in enumerate(zip(segments[:-1], segments[1:])):
            sim = self._get_similarity(text1, text2)
            if sim >= threshold:
                segment_map.append(0)
            else:
                segment_map.append(1)
        return self._index_mapping(segment_map)

    def _index_mapping(self, segment_map):
        index_list = []
        temp = []
        for index, i in enumerate(segment_map):
            if i == 1:
                index_list.append(temp)
                temp = [index]
            else:
                temp.append(index)
        index_list.append(temp)
        return index_list

    def _get_similarity(self, text1, text2):
        sentence_1 = [i.text.strip() for i in self.nlp(text1).sents if len(i.text.split(' ')) > 1]
        sentence_2 = [i.text.strip() for i in self.nlp(text2).sents if len(i.text.split(' ')) > 2]
        embeding_1 = self.model.encode(sentence_1)
        embeding_2 = self.model.encode(sentence_2)
        embeding_1 = np.mean(embeding_1, axis=0).reshape(1, -1)
        embeding_2 = np.mean(embeding_2, axis=0).reshape(1, -1)
        
        if np.any(np.isnan(embeding_1)) or np.any(np.isnan(embeding_2)):
            return 1
        
        sim = cosine_similarity(embeding_1, embeding_2)
        return sim

    def _text_tilling(self):
        tt = TextTilingTokenizerExt(w=200, k=40)
        text = '\n\n\t'.join(self.data[self.utterance].tolist())
        segment = tt.tokenize(text)
        segment = [i.replace("\n\n\t", ' ') for i in segment]
        return segment


# %% [markdown]
# As an initial attempt, I used the same model that the `pyconverse` author used, which is `all-MiniLM-L6-v2`:

# %%
ru_sts_ext_all_mini_lm_metrics_df, ru_sts_ext_all_mini_lm_segments = sts_windowdiff(df, 'ru', SemanticTextSegmentationMultilingual(df, 'ru_sentence', 'all-MiniLM-L6-v2'), verbose=True)
ru_sts_ext_all_mini_lm_metrics_df

# %% [markdown]
# As expected, the results were quite poor. For all threshold values, the WindowDiff score was equal to 0.397099, and the segmenter did not split the text at all:

# %%
plot_segmentation_for_best_score(ru_sts_ext_all_mini_lm_metrics_df, ru_sts_ext_all_mini_lm_segments, ['threshold'], ru_ground_truth, title='Semantic Text Segmentation Multilang')

# %%
print_segmentation(ru_sts_ext_all_mini_lm_segments[(.0,)], limit=130)

# %% [markdown]
# The next option is to replace `all-MiniLM-L6-v2` with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. This model is more up-to-date and provides support for multiple languages:

# %%
ru_sts_ext_multilang_mini_lm_metrics_df, ru_sts_ext_multilang_mini_lm_segments = sts_windowdiff(df, 'ru', SemanticTextSegmentationMultilingual(df, 'ru_sentence', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'), verbose=True)
ru_sts_ext_multilang_mini_lm_metrics_df

# %%
ru_sts_ext_multilang_mini_lm_metrics_df, ru_sts_ext_multilang_mini_lm_segments = sts_windowdiff(df, 'ru', SemanticTextSegmentationMultilingual(df, 'ru_sentence', 'sentence-transformers/labse'), verbose=True)
ru_sts_ext_multilang_mini_lm_metrics_df

# %% [markdown]
# Seems like an improvement: at least WindowDiff changes, and I obtain more than one segment for four different `threshold` values. The best WindowDiff score occurs at 0.85, resulting in four segments:

# %%
plot_segmentation_for_best_score(ru_sts_ext_multilang_mini_lm_metrics_df, ru_sts_ext_multilang_mini_lm_segments, ['threshold'], ru_ground_truth, title='Semantic Text Segmentation Multilang')

# %%
print_segmentation(ru_sts_ext_multilang_mini_lm_segments[(.85,)], limit=130)

# %% [markdown]
# Let's test this algorithm with English text. For the initial attempt, I'll use the default `SemanticTextSegmentation`, and then I'll try `SemanticTextSegmentationMultilingual` with two models: `all-MiniLM-L6-v2` and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`:

# %%
en_sts_metrics_df, en_sts_segments = sts_windowdiff(df, 'en', SemanticTextSegmention(df, 'en_sentence'), verbose=True)
en_sts_metrics_df

# %%
plot_segmentation_for_best_score(en_sts_metrics_df, en_sts_segments, ['threshold'], en_ground_truth, title='Semantic Text Segmentation: Google translated')

# %%
print_segmentation(en_sts_segments[(.1,)], limit=130)

# %% [markdown]
# Now, let's see what results I can achieve with the multilingual TextTiling algorithm and the English-only embedding model `all-MiniLM-L6-v2`, which I used in `SemanticTextSegmentationMultilingual`:

# %%
en_sts_ext_all_mini_lm_metrics_df, en_sts_ext_all_mini_lm_segments = sts_windowdiff(df, 'en', SemanticTextSegmentationMultilingual(df, 'en_sentence', 'all-MiniLM-L6-v2'), verbose=True)
en_sts_ext_all_mini_lm_metrics_df

# %%
plot_segmentation_for_best_score(en_sts_ext_all_mini_lm_metrics_df, en_sts_ext_all_mini_lm_segments, ['threshold'], en_ground_truth, title='Semantic Text Segmentation Multilang: Google translated')

# %% [markdown]
# The last combination is the same as the previous one, but with the new multilingual model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`:

# %%
en_sts_ext_multilang_mini_lm_metrics_df, en_sts_ext_multilang_mini_lm_segments = sts_windowdiff(df, 'en', SemanticTextSegmentationMultilingual(df, 'en_sentence', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'), verbose=True)
en_sts_ext_multilang_mini_lm_metrics_df

# %%
plot_segmentation_for_best_score(en_sts_ext_multilang_mini_lm_metrics_df, en_sts_ext_multilang_mini_lm_segments, ['threshold'], en_ground_truth, title='Semantic Text Segmentation Multilang: Google translated')

# %% [markdown]
# As I expect with a more modern model, the results were improved &mdash; the best score is 0.290955 for 0.65 threshold.

# %% [markdown]
# # Concnulsion

# %% [markdown]
# In this article, I attempted to find the best existing text segmentation implementation. While I didn't conduct an exhaustive study of all available algorithms, I did select some of the most popular ones: TextTiling, GraphSeg, and Semantic Segmentation.
#
# Unfortunately, I must conclude that none of these implementations work well enough for my task. Below, I will present a table with all the WindowDiff metrics and showcase the segments that correspond to the best score:

# %%
ru_text_tiling_ext_metrics_df['algorithm'] = ru_text_tiling_ext_metrics_df['algorithm'] + '_ext' 
ru_sts_ext_all_mini_lm_metrics_df['algorithm'] = ru_sts_ext_all_mini_lm_metrics_df['algorithm'] + '_ext_all_mini_lm'
ru_sts_ext_multilang_mini_lm_metrics_df['algorithm'] = ru_sts_ext_multilang_mini_lm_metrics_df['algorithm'] + '_ext_multilang_mini_lm'

ru_metrics = pd.concat([
    ru_text_tiling_metrics_df,
    ru_text_tiling_ext_metrics_df,
    ru_graph_seg_metrics_df,
    ru_sts_metrics_df,
    ru_sts_ext_all_mini_lm_metrics_df,
    ru_sts_ext_multilang_mini_lm_metrics_df,
])

min_windiff = ru_metrics['win_diff'].min()
ru_metrics[ru_metrics['win_diff'] == min_windiff][['algorithm', 'win_diff', 'topics_count']]
# ru_metrics[ru_metrics['topics_count'] > 1].sort_values('win_diff').head(40)

# %% [markdown]
# Semantic text segmentation with the modified Sentence Transformers model yielded the best overall result for Russian text. Let's take another look at what this segmentation looks like:

# %%
plot_segmentation(ru_ground_truth, ru_sts_ext_multilang_mini_lm_segments[(.85,)], title='Semantic Text Segmentation: Winner')

# %%
en_sts_ext_all_mini_lm_metrics_df['algorithm'] = en_sts_ext_all_mini_lm_metrics_df['algorithm'] + '_ext_all_mini_lm'
en_sts_ext_multilang_mini_lm_metrics_df['algorithm'] = en_sts_ext_multilang_mini_lm_metrics_df['algorithm'] + '_ext_multilang_mini_lm'

en_metrics = pd.concat([
    en_text_tiling_metrics_df,
    en_graph_seg_metrics_df,
    en_sts_metrics_df,
    en_sts_ext_all_mini_lm_metrics_df,
    en_sts_ext_multilang_mini_lm_metrics_df,
])

min_windiff = en_metrics['win_diff'].min()
en_metrics[en_metrics['win_diff'] == min_windiff][['algorithm', 'win_diff', 'topics_count']]

# %% [markdown]
# Only one algorithm achieved the best score: semantic text segmentation with the default model. Below is the segmentation plot for the winning algorithm:

# %%
plot_segmentation(en_ground_truth, en_sts_ext_multilang_mini_lm_segments[(.65,)], title='Semantic Text Segmentation: Google translated. Winner')

# %% [markdown]
# PS: You might be wondering why I segmented Google-translated texts. Here's the reason: I wanted to test whether Google-translated text could be segmented more accurately than simply matching boundaries from the translation to the corresponding sentences in the original Russian text.
#
# As you can see, even for Russian text with some modifications, I've achieved a basic level of segmentation. The next challenge is to find a way to improve upon this baseline segmentation.

# %% [markdown]
# # PPS: Analyzing WindowDiff Metrics for 12 Episodes

# %% [markdown]
# To be honest, comparing different algorithms on a single episode is not ideal. So, in this section, I'm going to gather WindowDiff metrics for 12 episodes including 412th.

# %% [markdown]
# First, let's compute WindowDiff for each episode and construct a dataframe containing all the metrics.

# %%
results = {}

# %%
algorithms = {
    'graph_seg': lambda df, lang, ep: graph_seg_windowdiff(df, lang, episode=ep),
    'text_tiling_seg': lambda df, lang, _: text_tiling_windowdiff(df, lang, texttiling.TextTilingTokenizer if lang == 'en' else TextTilingTokenizerExt),
    'semantic_text_seg': lambda df, lang, _: sts_windowdiff(df, lang, SemanticTextSegmentationMultilingual(df, f'{lang}_sentence', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'))
}

episodes = list(range(400, 416))
langs = ['ru', 'en']

t = tqdm(total=(len(episodes) * len(langs) * len(algorithms)))
all_df = pd.read_csv('/mnt/c/Users/andrei/Downloads/400-415-episodes-ground-truth.csv')
for ep in episodes:
    df = all_df[all_df['episode'] == ep] # pd.read_csv(f'../data/{ep}_ep_reference.csv')
    for lang in langs:
        for name, func in algorithms.items():
            try:
                if f'{name}_{ep}_{lang}' in results:
                    continue

                metrics_df, segments = func(df, lang, ep)
                results[f'{name}_{ep}_{lang}'] = {
                    'episode': ep,
                    'metrics': metrics_df,
                    'segments': segments
                }
            except Exception as e:
                print(e)
                print(name, ep, lang)
            finally:
                t.update(1)

# %%
for k, v in results.items():
    v['metrics']['episode'] = v['episode']

# %%
ru_result_keys = {x for x in results.keys() if x.endswith('ru')}
ru_results = {k: v for k, v in results.items() if k in ru_result_keys}

ru_results_df = pd.concat([v['metrics'] for k, v in ru_results.items()])
ru_results_df

# %%
ru_results_df = pd.concat([v['metrics'] for k, v in results.items()])
ru_results_df = ru_results_df.fillna('_')
ru_results_df['grouping_key'] = ru_results_df.apply(lambda x: f'{x["lang"]}__{x["algorithm"]}__{x["min_seg"]}__{x["sent_size"]}__{x["block_size"]}__{x["threshold"]}', axis='columns')

agg_df = ru_results_df.groupby(['grouping_key']).agg(
    win_diff_median=('win_diff', np.median),
    topic_count_median=('topics_count', np.median),
    win_diff_mean=('win_diff', np.mean),
    topic_count_mean=('topics_count', np.mean),
    win_diff_std=('win_diff', np.std),
    single_topic_count=('topics_count', lambda x: np.count_nonzero(x == 1))).reset_index().sort_values('win_diff_median')

agg_df

# %% [markdown]
# I'd like to clarify the meaning of the `single_topic_count` column.
#
# As observed earlier, some segmentation algorithms might treat the entire text as a single segment. I want to identify these algorithms that do not perform any segmentation. Therefore, the column `single_topic_count` indicates how many times the corresponding algorithm did not segment the text.

# %%
agg_df[agg_df['single_topic_count'] != 0].apply(lambda x: x['grouping_key'][:2], axis='columns').value_counts()

# %% [markdown]
# So, 18 algorithms for Russian and 15 algorithms for English (each "algorithm" representing specific hyperparameters) did not perform well.

# %%
agg_df[(agg_df['single_topic_count'] != 0) & (agg_df['grouping_key'].str[:2] == 'ru')].sort_index()

# %% [markdown]
# Out of the 20 semantic text segmentation variations, 18 failed to split a Russian text into segments.

# %%
agg_df[(agg_df['single_topic_count'] != 0) & (agg_df['grouping_key'].str[:2] == 'en')].sort_index()

# %% [markdown]
# Similarly, in the case of English, 15 out of 20 semantic segmentations were unable to split the text into segments.

# %% [markdown]
# What's particularly intriguing is that only the semantic segmentation algorithms proved unsuccessful in segmenting both English and Russian texts.

# %% [markdown]
# ## And the winners are...

# %%
agg_df[(agg_df['single_topic_count'] == 0) & (agg_df['grouping_key'].str[:2] == 'ru')].sort_values('win_diff_median').head(5)

# %%
agg_df[(agg_df['single_topic_count'] == 0) & (agg_df['grouping_key'].str[:2] == 'en')].sort_values('win_diff_median').head(5)

# %% [markdown]
# GraphSeg with `min_seg` values of 96 and 48, along with Semantic Text Segmentation using a 0.9 threshold, exhibited the best performance. While TextTiling showed some effectiveness, it didn't meet my desired expectations. 
#
# On the other hand, for English texts, the WindowDiff metric is slightly lower, but GraphSeg with the same `min_seg` values stands out among the winners. Surprisingly, none of the "versions" of the TextTiling algorithm are present in the list of top performers.
