import polars as pl

def get_segmentation(df: pl.DataFrame, target: str) -> dict[int, str]:
    k = 0
    episodes = {}
    for group in df[['ru_sentence', target, 'episode']].group_by('episode'):
        topic = []
        topics = []
        for row in group[1].iter_rows():
            topic.append(row[0])
            if row[1] == 1:
                topics.append(' '.join(topic))
                topic = []
        if topic:
            topics.append(' '.join(topic))

        episodes[group[0]] = '|'.join(topics)
        # raise
    return episodes