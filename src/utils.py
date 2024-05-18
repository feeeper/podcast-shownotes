import polars as pl
from pathlib import Path
import re
import shutil


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


def sync_cache(from_: Path, to_: Path) -> None:
    model_sizes = ['large', 'medium', 'small']
    episodes_from = [x for x in from_.iterdir() if x.suffix.lower() == '.json']
    regexp = 'episode-(\d\d\d\d?)'
    episodes_nums = sorted({re.findall(regexp, x.name)[0] for x in episodes_from if 'episode' in x.name})
    for ep_num_from in episodes_nums:
        sync_to_dir = Path(to_ / str(int(ep_num_from)))  # ep_num_from could be with leading zeros
        if not sync_to_dir.exists():
            sync_to_dir.mkdir(parents=True)

        # sync transcription, metadata and rename
        for size in model_sizes:
            ep_from = next((x for x in episodes_from if f'episode-{ep_num_from}' in x.name and size in x.name), None)
            if not ep_from:
                continue

            ep_to = sync_to_dir / f'transcription-{size}.json'
            shutil.copy(ep_from, ep_to)
            break
