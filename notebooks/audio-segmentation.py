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
# # Audio segmentation

# %%
import os
from pydub import AudioSegment
import torch
import datetime
from pathlib import Path
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource
)
os.environ['DEEPGRAM_API_KEY'] = '227...2b7b'

# %%
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY', None)
if os.environ.get('DEEPGRAM_API_KEY', None) is None:
    raise ValueError('DEEPGRAM_API_KEY environment variable is not set')


# %%
def transcribe(path_to_audio: Path | str) -> dict:    
    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)

        options = PrerecordedOptions(
            model="nova-2",
            language="ru",
            paragraphs=True,
            utterances=True,
            diarize=True,
        )
        
        with open(path_to_audio, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        # response = deepgram.listen.prerecorded.v("1").transcribe_url(AUDIO_URL, options)
        # print(response.to_json(indent=4))
        response = deepgram.listen.prerecorded.v('1').transcribe_file(
            source=payload,
            options=options
        )

    except Exception as e:
        print(f"Exception: {e}")
    
    return response


# %%
audio = AudioSegment.from_mp3('/mnt/d/Projects/podcast-shownotes/episodes/___/episode-469.mp3')
audio

# %%
# torch.set_num_threads(1)

# model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
# (get_speech_timestamps, _, read_audio, _, _) = utils

# wav = read_audio('/mnt/d/Projects/podcast-shownotes/episodes/___/episode-469.mp3')

# %%
# def progress_tracking_callback(progress: float) -> None:
#     print(f'{progress:.2f}% done', end='\r')

# %%
# start = datetime.datetime.now()
# speech_timestamps = get_speech_timestamps(
#     wav,
#     model,
#     return_seconds=True,
#     progress_tracking_callback=progress_tracking_callback,
#     visualize_probs=True
# )
# end = datetime.datetime.now()

# print(f'total time: {end-start}')

# %%
# speech_timestamps[:10]

# %%
audio.duration_seconds

# %%
max_bunch_size_mb = 20

p = Path('/mnt/d/Projects/podcast-shownotes/episodes/___/episode-469.mp3')
filesize_mb = p.stat().st_size / 1024 / 1024
approx_one_sec_size_mb = filesize_mb / audio.duration_seconds
approx_sec_for_bunch = max_bunch_size_mb / approx_one_sec_size_mb
bunch_count = int(filesize_mb // max_bunch_size_mb + 10)
bunch_endings = [audio.duration_seconds / bunch_count * i for i in range(bunch_count)]  # plus tail

print(f'{approx_one_sec_size_mb=}')
print(f'{approx_sec_for_bunch=}')
print(f'{bunch_count=}')
print(f'{bunch_endings=}')

# %%
# bunches = []
# current_start = 0
# current_end = 0
# i = 0

# for end in bunch_endings[1:]:
#     while speech_timestamps[i]['end'] < end:
#         current_end = speech_timestamps[i]['end']
#         i += 1

#     bunches.append({'i': i, 'start': current_start, 'end': current_end})
#     current_start = speech_timestamps[i - 1]['start']
# bunches.append({'i': i, 'start': current_start, 'end': speech_timestamps[-1]['end']})

# %%
# bunches, bunch_endings

# %%
# end_sec = bunches[0]['end']
end_sec = 2540
s = audio[:end_sec * 1000]

# %%
s.export('../data/sample.mp3')

# %%
audio_file = '../data/sample.mp3'
transcript_w_utter = transcribe(audio_file)

# %%
print(transcript.to_json(indent=4, ensure_ascii=False)[:1_000])

# %%
[x for x in transcript_w_utter.results.channels[0].alternatives[0].paragraphs.paragraphs if x.speaker == 0][1]

# %%
[x for x in transcript_wo_utter.results.channels[0].alternatives[0].paragraphs.paragraphs if x.speaker == 0][1]

# %%
import pandas as pd

[{ 'id:': idx, 'sentences': ' '.join([s.text for s in p.sentences]), 'start': p.start, 'end': p.end, 'speaker': p.speaker } for idx, p in enumerate(transcript_w_utter.results.channels[0].alternatives[0].paragraphs.paragraphs[:10])][:8]

# %%
transcript_wo_utter.results.channels[0].alternatives[0].paragraphs.paragraphs[0]

# %%
with open('../data/tr.json', 'w', encoding='utf8') as f:
    f.write(transcript_wo_utter.to_json(ensure_ascii=False))

# %%
