# source based on the SemanticTextSegmentation class from pyconverse library:
# https://github.com/maxent-ai/converse

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words
from stanza import Pipeline

from sentence_transformers import SentenceTransformer
from src.components.segmentation.text_tiling_tokenizer import TextTilingTokenizer


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

    def __init__(
            self,
            data,
            utterance,
            model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            w=100,
            k=5,
    ):
        self.data = data
        self.utterance = utterance
        self.model = SentenceTransformer(model)
        self._pipeline = Pipeline(lang='ru', processors='tokenize')
        self.tt = TextTilingTokenizer(w=w, k=k, stopwords=get_stop_words('ru'))

    def __attrs_post_init__(self):
        columns = self.data.columns.tolist()

    def get_segments(self, threshold=0.9, verbose: bool = False):
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
        merge_index = self._merge_segments(segments, threshold, verbose=verbose)
        new_segments = []
        for i in merge_index:
            seg = ' '.join([segments[_] for _ in i])
            new_segments.append(seg)
        return new_segments

    def _merge_segments(self, segments, threshold, verbose: bool = False):
        segment_map = [0]
        for index, (text1, text2) in enumerate(zip(segments[:-1], segments[1:])):
            sim = self._get_similarity(text1, text2)
            if verbose:
                print(f'{text1[:50]}\t{text2[:50]}\t{sim}')
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
        sentence_1 = [i.text.strip() for i in self._pipeline(text1).sentences if len(i.text.split(' ')) > 1]
        sentence_2 = [i.text.strip() for i in self._pipeline(text2).sentences if len(i.text.split(' ')) > 2]
        embeding_1 = self.model.encode(sentence_1)
        embeding_2 = self.model.encode(sentence_2)
        embeding_1 = np.mean(embeding_1, axis=0).reshape(1, -1)
        embeding_2 = np.mean(embeding_2, axis=0).reshape(1, -1)

        if np.any(np.isnan(embeding_1)) or np.any(np.isnan(embeding_2)):
            return 1

        sim = cosine_similarity(embeding_1, embeding_2)
        return sim

    def _text_tilling(self):
        text = '\n\n\t'.join(self.data[self.utterance].tolist())
        segment = self.tt.tokenize(text)
        segment = [i.replace("\n\n\t", ' ') for i in segment]
        return segment