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
# # Sematnic search

# %%
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from pymilvus.exceptions import SchemaNotReadyException

import torch
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset
from math import ceil, floor

import polars as pl

# %%
connections.connect(host='localhost', port='19530')


# %%
def create_or_get_collection(
    collection_name: str = 'devzen_transcript',
    verbose=False
) -> Collection:
    try:
        collection = Collection(collection_name)
        print('Collection exists')
    except SchemaNotReadyException as e:
        print('Collection doesn\'t exist. Create new one.')  
        default_fields = [
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True, description='id'),
            FieldSchema(name='episode', dtype=DataType.INT16, description='episode number'),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=768, description='segment\' embedding'),
            FieldSchema(name='start_at', dtype=DataType.INT16, description='timestamp when the segment starts'),
            FieldSchema(name='segment', dtype=DataType.VARCHAR, description='segment\'s text', max_length=2_048),
        ]
        default_schema = CollectionSchema(fields=default_fields, description='devzen transcriptions')
        collection = Collection(name=collection_name, schema=default_schema)
        # create indexes
        collection.create_index(
            field_name='embedding', 
            index_params= {
                'metric_type':'COSINE',
                'index_type':'HNSW',
                'params':{
                    'M': 128,
                    'efConstruction': 42
                }
            }
        )
    
    return collection


# %%
def insert_segment_to_collection(
    collection: Collection,
    segment: dict,
    verbose: bool = False,
    flush: bool = False
) -> int:
    segment_field = next((x for x in collection.schema.fields if x.name == 'segment'), None)
    max_length = segment_field.params['max_length'] if 'max_length' in segment_field.params else 2048
    
    mr = collection.insert([
        [segment['episode']],
        [segment['embedding']],
        [segment['start_at']],
        [segment['segment'][:max_length]]
    ])
    if flush: collection.flush()
    return mr


# %%
def insert_segments_to_collection(
    collection: Collection,
    segments: list[dict],
    verbose: bool = False
) -> int:
    mrs = []
    for s in segments:
        mrs.append(insert_segment_to_collection(collection, s))
    collection.flush()

    return mrs


# %%
def find_in_collection(
    collection: Collection,
    embedding: list[float],
    top_k: int = 5
) -> tuple:
    collection.load()
    
    search_params = {'metric_type': 'COSINE'}
    results = collection.search(
        [embedding],
        anns_field='embedding',
        param=search_params,
        limit=top_k
    )
    return (results[0].ids, results[0].distances)


# %%
def get_by_ids_in_collection(
    collection: Collection,
    ids: list[int],
    output_fields: list[str] = None
):
    ids_expr = ','.join(map(str, ids))
    expr = f'id in [{ids_expr}]'
    output_fields = ['id', 'embedding', 'episode', 'start_at', 'segment'] if output_fields is None else output_fields
    res = collection.query(expr, output_fields)
    sorted_res = sorted(res, key=lambda k: k['id'])
    return sorted_res


# %%
collection = create_or_get_collection()

# %%
collection_concat = create_or_get_collection('devzen_transcript_concat')

# %%
collection.num_entities

# %%
collection_concat.num_entities

# %%
# collection.drop()

# %%
collection.load()

# %%
collection.num_entities

# %%
res = find_in_collection(collection, xb[0], top_k=1)
res

# %%
entries = get_by_ids_in_collection(collection, res[0], output_fields=['id', 'segment'])
entries

# %% [markdown]
# # Build embeddings and insert to the database

# %%
checkpoint = 'intfloat/multilingual-e5-base'
model = AutoModel.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# %%
csv_path = '../data/400-415-with-target.csv'
df = pl.read_csv(csv_path)
df = df.with_columns(pl.col('start').map_elements(lambda x: floor(x)).alias('start_at'))
print(df.shape)
df.head()

# %%
ds = Dataset.from_pandas(df[['episode', 'start_at', 'ru_sentence']].to_pandas())
ds


# %%
def get_embeddings(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    inputs = { k: v.to(device) for k, v in  encoded_input.items() }
    embeddings = model(**inputs)
    return embeddings.pooler_output


# %%
ds = ds.map(lambda x: { 'embedding': get_embeddings(x['ru_sentence']).detach().cpu().numpy() }, batched=True, batch_size=100)
ds

# %%
results = []
batch_size=100
for row in ds.iter(batch_size=batch_size):
    entities = [{
        'episode': row['episode'][i],
        'start_at': row['start_at'][i],
        'segment': row['ru_sentence'][i],
        'embedding': row['embedding'][i]
    } for i in range(batch_size)]
    res = insert_segments_to_collection(
        collection=collection,
        segments=entities)
    results.extend(res)
    print(f'{len(results)}/{ds.num_rows} ({len(results)/ds.num_rows:.2f}%)', end='\r')

# %% [markdown]
# text = 'websql'
# text = df[2004]['ru_sentence'][0]
# text = df[2104]['ru_sentence'][0]
# text = 'книга про менеджмент'
#
# print(f'{text = }')
#
# text_emb = get_embeddings([text])
# text_emb_numpy = text_emb[0].detach().cpu().numpy()
# print(f'embedding = {text_emb_numpy[:10]}')
#
# result = find_in_collection(collection_concat, text_emb_numpy, top_k=10)
# print(f'find result = {result}')
#
# entities = get_by_ids_in_collection(collection_concat, ids=result[0], output_fields=['id', 'segment', 'episode'])
# for entity in entities:
#     print(f'{entity["id"]}: {entity["segment"]} ({entity["episode"]})')
# # print(f'{entities = }')

# %% [markdown]
# ## concatenate sentences

# %%
csv_path = '../data/400-415-with-target.csv'
df = pl.read_csv(csv_path)
df = df.with_columns(pl.col('start').map_elements(lambda x: floor(x)).alias('start_at'))
with pl.Config(fmt_str_lengths=100, tbl_width_chars=120):
    print(df)

# %%
data = []
MAX_LEN = 1_024
for idx, gr in df.sort(['episode', 'start_at']).group_by(['episode']):
    texts = []
    current_len = 0
    start_at = 0
    for row in gr.iter_rows(named=True):
        text = row['ru_sentence']
        if current_len + len(text) + 1 < MAX_LEN:
            texts.append(text)
            current_len += len(text)
        else:
            data.append({
                'segment': ' '.join(texts),
                'episode': row['episode'],
                'start_at': start_at
            })
            texts = [text]
            start_at = row['start_at']
            current_len = len(text)

# %%
df = pl.from_dicts(data)
df

# %%
ds = Dataset.from_pandas(df[['episode', 'start_at', 'segment']].to_pandas())
ds = ds.map(lambda x: { 'embedding': get_embeddings(x['segment']).detach().cpu().numpy() }, batched=True, batch_size=100)
ds

# %%
results = []
batch_size=100
for row in ds.iter(batch_size=batch_size):
    entities = [{
        'episode': row['episode'][i],
        'start_at': row['start_at'][i],
        'segment': row['segment'][i],
        'embedding': row['embedding'][i]
    } for i in range(len(row['episode']))]
    res = insert_segments_to_collection(
        collection=collection_concat,
        segments=entities)
    results.extend(res)
    print(f'{len(results)}/{ds.num_rows} ({len(results)/ds.num_rows:.2f*100}%)', end='\r')

# %%
text = 'websql'
text = df[1000]['segment'][0]
text = df[1000]['segment'][0]

print(f'{text = }')

text_emb = get_embeddings([text])
text_emb_numpy = text_emb[0].detach().cpu().numpy()
print(f'embedding = {text_emb_numpy[:10]}')

result = find_in_collection(collection, text_emb_numpy, top_k=10)
print(f'find result = {result}')

entities = get_by_ids_in_collection(collection, ids=result[0], output_fields=['id', 'segment', 'episode'])
for entity in entities:
    print(f'{entity["id"]}: {entity["segment"]} ({entity["episode"]})')
# print(f'{entities = }')

# %%
text = 'websql'
text = df[1000]['segment'][0]
text = df[1000]['segment'][0]

text = 'игра для playstation'

print(f'{text = }')
print(f'episode = {df[1000]["episode"][0]}')

text_emb = get_embeddings([text])
text_emb_numpy = text_emb[0].detach().cpu().numpy()
print(f'embedding = {text_emb_numpy[:10]}')

result = find_in_collection(collection_concat, text_emb_numpy, top_k=10)
print(f'find result = {result}')

entities = get_by_ids_in_collection(collection_concat, ids=result[0], output_fields=['id', 'segment', 'episode'])
for entity in entities:
    print(f'{entity["id"]}: {entity["segment"]} ({entity["episode"]})')
# print(f'{entities = }')
