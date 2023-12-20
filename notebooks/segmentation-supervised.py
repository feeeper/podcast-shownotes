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
import typing as t
from dataclasses import dataclass
import re
import warnings
warnings.filterwarnings("ignore")

from joblib import Memory

import pandas as pd
import polars as pl
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from tqdm.notebook import tqdm

from stop_words import get_stop_words
from sentence_transformers import SentenceTransformer

# %%
memory = Memory('../.cache')

# %%
csv_path = '../data/400-415-episodes-ground-truth.csv'
df = pd.read_csv(csv_path)
df.head()


# %% [markdown]
# # Build embeddings

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
# # Fill target `is_first_sentence_in_topic`

# %%
df['ground_truth_next'] = df.groupby('episode')['ground_truth'].shift(-1).fillna(1_000).astype(int)
df[(df['episode']==413) & (df['ground_truth_next'] != df['ground_truth'])]


# %%
def is_first_sentence_in_topic(s: pd.Series) -> int:
    return 0 if s['ground_truth'] == s['ground_truth_next'] else 1

df['is_first_sentence_in_topic'] = df.apply(is_first_sentence_in_topic, axis='columns')

# %%
df[df['is_first_sentence_in_topic'] == 1][['ru_sentence', 'en_sentence', 'ground_truth', 'is_first_sentence_in_topic']]

# %%
df['is_first_sentence_in_topic'].value_counts(normalize=True)


# %% [markdown]
# # Build Dataset with target

# %%
def build_dataset(
    df: t.Union[pd.DataFrame, pl.DataFrame],
    embeddings: np.ndarray,
    prefix: str = '',
    columns_to_add: list[str] = ['ru_sentence', 'en_sentence', 'episode', 'is_first_sentence_in_topic']
) -> pl.DataFrame:
    res = pl.from_dataframe(df[columns_to_add])
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report
)

# %%
non_embeddings_columns = ['ru_sentence', 'en_sentence', 'episode', 'is_first_sentence_in_topic']
embedding_columns = [col for col in train.columns if col not in non_embeddings_columns]


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
train, test = split_train_test(ru_labse_df)
print(f'{train.shape=}')
print(f'{test.shape=}')

# %%
lr = LogisticRegression(random_state=42, class_weight='balanced')
lr.fit(train[embedding_columns], train['is_first_sentence_in_topic'])

# %%
preds = lr.predict_proba(test[embedding_columns])
preds

# %%
roc_auc_score(test['is_first_sentence_in_topic'], preds[:, 1])

# %%
preds = lr.predict(test[embedding_columns])
print(classification_report(test['is_first_sentence_in_topic'], preds))

# %%
preds = lr.predict_proba(test[embedding_columns])

# %%
test.filter(pl.col('is_first_sentence_in_topic')==1)[non_embeddings_columns]

# %%
a = np.argwhere(test['is_first_sentence_in_topic'])[:, 0]
a

# %%
preds[:, 1]

# %%
test[['is_first_sentence_in_topic']].to_pandas()[['is_first_sentence_in_topic']].corrwith(pd.Series(preds[:, 1]))

# %%
df1 = pl.DataFrame(
    {
        "foo": [1, 2, 3],
        "bar": [6, 7, 8],
        "ham": ["a", "b", "c"],
    }
)
x = pl.Series("apple", [10, 20, 30])
df1.hstack([x])

# %%
test = test.hstack([pl.Series('ru_labse_preds', preds[:, 1])])
print(test.select(pl.corr('is_first_sentence_in_topic', 'ru_labse_preds')))

# %%
corr_df = test.hstack([pl.Series('preds', preds[:, 1])])[['preds'] + [x for x in non_embeddings_columns + embedding_columns if not x.endswith('sentence') and x != 'episode']].corr()
corr_df

# %%
test.filter(pl.col('is_first_sentence_in_topic') == 1)[['ru_sentence', 'en_sentence', 'ru_labse_preds', 'is_first_sentence_in_topic']][:20]

# %%
with pl.Config(fmt_str_lengths=500, tbl_rows=-1):
    print(test[['ru_sentence', 'en_sentence', 'ru_labse_preds', 'is_first_sentence_in_topic']][190:200])

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
res = paraphrase(test[1, 'ru_sentence'])
res


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

# %%
