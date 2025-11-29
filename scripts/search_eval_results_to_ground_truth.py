from collections import defaultdict
from argparse import ArgumentParser
import json
from pathlib import Path


parser = ArgumentParser()
parser.add_argument("--path", "-p", type=Path, help="Path to directory with queries")
parser.add_argument("--output", "-o", type=Path, help="Path to save result file")

args = parser.parse_args()

items = defaultdict(list)
for f in args.path.iterdir():
    j = json.loads(f.read_text())
    element = j[0]  # each file contains a single element array
    err = element["result"].get("error", None)
    if err is not None:
        print(f"Error for the episode {f.name}: {err}")
        continue

    query = element["result"]["query"]
    episode_num = element["episode_number"]
    segment_num = element["segment_number"]

    items[query].append((episode_num, segment_num))

json.dump(items, open(args.output, "w", encoding="utf8"), ensure_ascii=False)
