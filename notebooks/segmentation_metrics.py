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
# # Text segmentation. Metrics

# %% [markdown]
# ## Task

# %% [markdown]
# I'm currently working on a new pet project centered around podcasts. The initial objective of this project is to identify timestamps when podcast hosts transition to discussing new topics. This task is commonly referred to as Text Segmentation:
# > Text segmentation involves breaking down written text into meaningful units, such as words, sentences, or topics. \[[Wikipedia](https://en.wikipedia.org/wiki/Text_segmentation)\]
#
# To test this concept, I'll use the Russian podcast DevZen as an example. Over the past 9 years, the hosts have been discussing various subjects related to technology, software engineering, and databases on a weekly basis.

# %% [markdown]
# ## Metrics

# %% [markdown]
# ### Precision and Recall

# %% [markdown]
# **Precision** &mdash; the percentage of predicted boundaries that match actual boundaries.
#
# **Recall** &mdash; the percentage of actual boundaries that are correctly predicted by the model.
#
# However, this metric might be too coarse in scenarios where predictions are close but not exact matches. Such cases are called "near misses," where the model predicts boundaries that are in close proximity to the actual boundaries. An example is provided below.
#
# Consider a reference segmentation: AAA|BBBB|CC, where AAA, BBBB, CC are "segments," and vertical lines represent boundaries. An algorithm's task is to predict these boundaries.
#
# Now, let's compare Model-A and Model-B using the following table:
#
# ```
# |   Name  | Segmentation |
# |:--------|:-------------|
# | Ref     | AAA|BBBB|CC  |
# | Model-A | AA|BBBBBB|C  |
# | Model-B | A|BBB|CCC|D  |
# ```
#
# Both models have a precision and recall equal to 0 based on traditional definitions. However, it is clear that Model-A performs much better than Model-B, as the difference between its predictions and the reference segmentation is significantly smaller.
#
# Hence, the traditional precision-recall metric might not be the best choice for evaluating this task due to its inability to capture nuanced performance differences in near misses.

# %% [markdown]
# ### Pk

# %% [markdown]
# **Pk**
#
# The **Pk** metric is calculated using a sliding window-based approach. The window size is typically set to half of the average true segment number.
#
# As the window slides through both the reference segmentation and predicted segmentation, at each step, we determine whether the ends of the window are within the same segment or not. If there is a mismatch between the ground truth and predicted segmentation, the error counter is incremented. Finally, the metric value is normalized by the count of measurements. In an ideal scenario, the best segmentation would result in a **Pk** value equal to 0.
#
# Below, you can find the code implementation (sourced from the [NLTK library](https://tedboy.github.io/nlps/_modules/nltk/metrics/segmentation.html#pk)) and its usage for the example provided in the **Precision and Recall** section.

# %%
def pk(ref, hyp, k=None, boundary='1'):
    """
    Compute the Pk metric for a pair of segmentations A segmentation
    is any sequence over a vocabulary of two items (e.g. "0", "1"),
    where the specified boundary value is used to mark the edge of a
    segmentation.

    >>> '%.2f' % pk('0100'*100, '1'*400, 2)
    '0.50'
    >>> '%.2f' % pk('0100'*100, '0'*400, 2)
    '0.50'
    >>> '%.2f' % pk('0100'*100, '0100'*100, 2)
    '0.00'

    :param ref: the reference segmentation
    :type ref: str or list
    :param hyp: the segmentation to evaluate
    :type hyp: str or list
    :param k: window size, if None, set to half of the average reference segment length
    :type boundary: str or int or bool
    :param boundary: boundary value
    :type boundary: str or int or bool
    :rtype: float
    """

    if k is None:
        k = int(round(len(ref) / (ref.count(boundary) * 2.)))

    err = 0
    for i in range(len(ref)-k +1):
        r = ref[i:i+k].count(boundary) > 0
        h = hyp[i:i+k].count(boundary) > 0
        if r != h:
            err += 1
    return err / (len(ref)-k +1.)


# %%
for k in range(1, 6):
    print(f"Model-A | k={k} | {pk('AAA|BBBB|CC', 'AA|BBBBBB|C', boundary='|', k=k):.2f}")
    print(f"Model-B | k={k} | {pk('AAA|BBBB|CC', 'A|BBB|CCC|D', boundary='|', k=k):.2f}")

# %% [markdown]
# Certainly, the metric displays several drawbacks:
#
# 1. False negatives are penalized more than false positives:
#    - When the right end of the probe crosses the reference boundary, it begins recording non-matches. This occurs when the algorithm assigns two sentences to the same segment, contrary to the reference. In such cases of false negatives, the resulting **Pk** would increase to `k`.
#    - The frequency at which this false positive is accounted for by **Pk** depends on its position. If it emerges in the middle of a segment, it is noted `k` times. If it takes place `j` < `k` sentences from the segment's beginning or end, a penalty of `j` times is applied.
# 2. Neglecting the number of boundaries:
# 3. Sensitivity to variations in segment size, as evident in the provided example.
# 4. Overemphasizing near-miss errors. In the example below, algorithm `a_0` achieves a better **Pk** than the more obviously precise algorithm `a_3`.

# %%
samples = {
    'ref': 'AAAAAAAA|BBBBBBBBBBBBB|CCC',
    'a_0':  'AAAAAAAAAAAAAAAAAAAAAA|CCC',
    'a_1':  'AAAAAAAA|DD|BBBBBBBBBB|CCC',
    'a_2':  'AAAAAAAA|DDD|BBBBBBBBB|CCC',
    'a_3':  'AAAAAAAAAA|BBBBBBBBBBB|CCC',
    'a_4':  'AAAAAAAAAAAAAAA|BBBBBB|CCC'
}

k = 4  # half of the average segment length
for key, val in samples.items():
    print(f"{key} | k={k} | {pk(samples['ref'], val, boundary='|', k=k):.2f}")


# %% [markdown]
# ### WindowDiff

# %% [markdown]
# The **WindowDiff** metric aims to address the shortcomings observed in **Pk**. The metric operates in the following manner: at each probe position, it compares the number of reference segmentation boundaries falling within that interval ($r_i$) with the number of boundaries assigned by the algorithm ($a_i$). The algorithm incurs a penalty if $r_i \neq a_i$ (which translates to $|r_i - a_i| > 0$).
#
# More formally:
#
# $$
# WindowDiff = \frac{1}{N - k} \sum_{i = 1}^{N-k} (| b(ref_i, ref_{i+k}) - b(hyp_{i}, hyp_{i+k}) | > 0)
# $$
#
# Here, $b(i, j)$ denotes the count of boundaries between positions $i$ and $j$ in the text, and $N$ represents the number of segments in the text.
#
# Below is the implementation of the algorithm from [NLTK](https://tedboy.github.io/nlps/_modules/nltk/metrics/segmentation.html#windowdiff), along with an additional `weighted` parameter:

# %%
def windowdiff(seg1, seg2, k, boundary="1", weighted=False):
    """
    Compute the windowdiff score for a pair of segmentations.  A
    segmentation is any sequence over a vocabulary of two items
    (e.g. "0", "1"), where the specified boundary value is used to
    mark the edge of a segmentation.

        >>> s1 = "000100000010"
        >>> s2 = "000010000100"
        >>> s3 = "100000010000"
        >>> '%.2f' % windowdiff(s1, s1, 3)
        '0.00'
        >>> '%.2f' % windowdiff(s1, s2, 3)
        '0.30'
        >>> '%.2f' % windowdiff(s2, s3, 3)
        '0.80'

    :param seg1: a segmentation
    :type seg1: str or list
    :param seg2: a segmentation
    :type seg2: str or list
    :param k: window width
    :type k: int
    :param boundary: boundary value
    :type boundary: str or int or bool
    :param weighted: use the weighted variant of windowdiff
    :type weighted: boolean
    :rtype: float
    """

    if len(seg1) != len(seg2):
        raise ValueError("Segmentations have unequal length")
    if k > len(seg1):
        raise ValueError("Window width k should be smaller or equal than segmentation lengths")
    wd = 0
    for i in range(len(seg1) - k + 1):
        ndiff = abs(seg1[i:i+k].count(boundary) - seg2[i:i+k].count(boundary))
        if weighted:
            wd += ndiff
        else:
            wd += min(1, ndiff)
    return wd / (len(seg1) - k + 1.)


# %%
samples = {
    'ref': 'AAAAAAAA|BBBBBBBBBBBBB|CCC',
    'a_0': 'AAAAAAAAAAAAAAAAAAAAAA|CCC',
    'a_1': 'AAAAAAAA|DD|BBBBBBBBBB|CCC',
    'a_2': 'AAAAAAAA|DDD|BBBBBBBBB|CCC',
    'a_3': 'AAAAAAAAAA|BBBBBBBBBBB|CCC',
    'a_4': 'AAAAAAAAAAAAAAA|BBBBBB|CCC'
}

k = 4
for key, val in samples.items():
    print(f"{key} | k={k} | {windowdiff(samples['ref'], val, boundary='|', k=k):.2f}")

# %% [markdown]
# ### Generalized Hamming Distance

# %% [markdown]
# The Generalized Hamming Distance (GHD) is more intricate compared to metrics such as **WindowDiff**, **Pk**, or the **Precision** and **Recall** pair.
#
# This metric involves transforming the generated segmentation into the reference segmentation by performing boundary insertion, deletion, and shift operations. The optimal segmentation would yield a **GHD** value of 0.
#
# Below is the code I obtained from the [NLTK library](https://tedboy.github.io/nlps/_modules/nltk/metrics/segmentation.html#ghd) for calculating the Generalized Hamming Distance:

# %%
import numpy as np

def _init_mat(nrows, ncols, ins_cost, del_cost):
    mat = np.empty((nrows, ncols))
    mat[0, :] = ins_cost * np.arange(ncols)
    mat[:, 0] = del_cost * np.arange(nrows)
    return mat


def _ghd_aux(mat, hyp_bound, ref_bound, ins_cost, del_cost, shift_cost_coeff):
    for i, rowi in enumerate(hyp_bound):
        for j, colj in enumerate(ref_bound):
            shift_cost = shift_cost_coeff * abs(rowi - colj) + mat[i, j]
            if rowi == colj:
                # boundaries are at the same location, no transformation required
                tcost = mat[i, j]
            elif rowi > colj:
                # boundary match through a deletion
                tcost = del_cost + mat[i, j + 1]
            else:
                # boundary match through an insertion
                tcost = ins_cost + mat[i + 1, j]
            mat[i + 1, j + 1] = min(tcost, shift_cost)


def ghd(ref, hyp, ins_cost=2.0, del_cost=2.0, shift_cost_coeff=1.0, boundary="1"):
    """
    Compute the Generalized Hamming Distance for a reference and a hypothetical
    segmentation, corresponding to the cost related to the transformation
    of the hypothetical segmentation into the reference segmentation
    through boundary insertion, deletion and shift operations.

    A segmentation is any sequence over a vocabulary of two items
    (e.g. "0", "1"), where the specified boundary value is used to
    mark the edge of a segmentation.

    Recommended parameter values are a shift_cost_coeff of 2.
    Associated with a ins_cost, and del_cost equal to the mean segment
    length in the reference segmentation.

        >>> # Same examples as Kulyukin C++ implementation
        >>> ghd('1100100000', '1100010000', 1.0, 1.0, 0.5)
        0.5
        >>> ghd('1100100000', '1100000001', 1.0, 1.0, 0.5)
        2.0
        >>> ghd('011', '110', 1.0, 1.0, 0.5)
        1.0
        >>> ghd('1', '0', 1.0, 1.0, 0.5)
        1.0
        >>> ghd('111', '000', 1.0, 1.0, 0.5)
        3.0
        >>> ghd('000', '111', 1.0, 2.0, 0.5)
        6.0

    :param ref: the reference segmentation
    :type ref: str or list
    :param hyp: the hypothetical segmentation
    :type hyp: str or list
    :param ins_cost: insertion cost
    :type ins_cost: float
    :param del_cost: deletion cost
    :type del_cost: float
    :param shift_cost_coeff: constant used to compute the cost of a shift.
        ``shift cost = shift_cost_coeff * |i - j|`` where ``i`` and ``j``
        are the positions indicating the shift
    :type shift_cost_coeff: float
    :param boundary: boundary value
    :type boundary: str or int or bool
    :rtype: float
    """

    ref_idx = [i for (i, val) in enumerate(ref) if val == boundary]
    hyp_idx = [i for (i, val) in enumerate(hyp) if val == boundary]

    nref_bound = len(ref_idx)
    nhyp_bound = len(hyp_idx)

    if nref_bound == 0 and nhyp_bound == 0:
        return 0.0
    elif nref_bound > 0 and nhyp_bound == 0:
        return nref_bound * ins_cost
    elif nref_bound == 0 and nhyp_bound > 0:
        return nhyp_bound * del_cost

    mat = _init_mat(nhyp_bound + 1, nref_bound + 1, ins_cost, del_cost)
    _ghd_aux(mat, hyp_idx, ref_idx, ins_cost, del_cost, shift_cost_coeff)
    return mat[-1, -1]


# %%
samples = {
    'ref': 'AAAAAAAA|BBBBBBBBBBBBB|CCC',
    'a_0': 'AAAAAAAAAAAAAAAAAAAAAA|CCC',
    'a_1': 'AAAAAAAA|DD|BBBBBBBBBB|CCC',
    'a_2': 'AAAAAAAA|DDD|BBBBBBBBB|CCC',
    'a_3': 'AAAAAAAAAA|BBBBBBBBBBB|CCC',
    'a_4': 'AAAAAAAAAAAAAAA|BBBBBB|CCC'
}

for key, val in samples.items():
    print(f"ref vs {key} | {ghd(samples['ref'], val, boundary='|'):.2f}")

# %% [markdown]
# ## Summary

# %% [markdown]
# ## Summary
#
# From my understanding, the most suitable metric for this task is **WindowDiff**. This metric is straightforward to comprehend and presents fewer issues when compared to **Pk**. Although **Precision** and **Recall** metrics are easy to interpret, they often yield values of 0, even for near-miss errors.
#
# For my task, I'll be utilizing the **WindowDiff** metric. As an illustrative example, I'll calculate all metrics for a couple of DevZen episodes.

# %%
import pandas as pd
import numpy as np

ep_415 = pd.read_csv('../data/415_ep_reference.csv')
ep_415.head()


# %% [markdown]
# **topic_num** refers to the topic number that has been calculated from podcast shownotes and the related transcript. **ground_truth** represents the manually filled topic number, serving as the reference segmentation.

# %%
def build_segmented_str(arr: list[int]) -> str:
    current_topic_idx = 0
    for i, topic_num in enumerate(arr):
        if topic_num == current_topic_idx:
            arr[i] = 0    
        else:
            current_topic_idx = arr[i]
            arr[i] = 1
    return ''.join(map(str, arr))


k = int(ep_415.groupby('ground_truth').agg(topic_len=('sentence', 'count'))['topic_len'].mean() // 2)
reference = list(ep_415['ground_truth'].values)
hypothesis = list(ep_415['topic_num'].values)

ref_seg = build_segmented_str(reference)
hyp_seg = build_segmented_str(hypothesis)

start, end = 240, 340
print(''.join(map(str, ref_seg[start:end])))
print(''.join(map(str, hyp_seg[start:end])))
print(''.join(map(str, ep_415['ground_truth'].values[start:end])))

# %% [markdown]
# Now we are prepared to compute metrics:

# %%
print(f'Pk: {pk(ref_seg, hyp_seg, k=k):.3f}')
print(f'WindowDiff: {windowdiff(ref_seg, hyp_seg, k=k):.3f}')
print(f'GHD: {ghd(ref_seg, hyp_seg):.3f}')
