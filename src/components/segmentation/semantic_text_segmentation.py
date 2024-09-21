# source based on the SemanticTextSegmentation class from pyconverse library:
# https://github.com/maxent-ai/converse

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words
from stanza import Pipeline

from sentence_transformers import SentenceTransformer
from src.components.segmentation.text_tiling_tokenizer import TextTilingTokenizer
from src.components.segmentation.sentences import Sentence


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
            sentences: list[Sentence],
            model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            w: int = 100,
            k: int = 5,
    ):
        self._sentences = sentences
        self.model = SentenceTransformer(model)
        self._pipeline = Pipeline(lang='ru', processors='tokenize')
        self.tt = TextTilingTokenizer(w=w, k=k, stopwords=get_stop_words('ru'))

    def get_segments(
            self,
            threshold=0.9,
            verbose: bool = False,
            as_sentences: bool = False
    ) -> list[list[str]]:
        """
        returns the transcript segments computed with texttiling and sentence-transformer.

        Paramters
        ---------
        threshold: float
            sentence similarity threshold. (used to merge the sentences into coherant segments)

        Return
        ------
        new_segments: list[str] | list[list[str]]
            list of segments or list of list of sentences for each segment
        """
        sentences_per_segment = self._text_tilling()
        # segments = [x.text for x in self._sentences]
        print(f'{len(sentences_per_segment)=}')

        merge_indexes = self._merge_segments(sentences_per_segment, threshold, verbose=verbose)
        new_segments = []

        print(f'{merge_indexes=}')
        for mi in merge_indexes:
            segment = []
            for i in mi:
                segment.extend(sentences_per_segment[i])

            if as_sentences:
                new_segments.append(segment)
            else:
                new_segments.append(' '.join([x.text for x in segment]))

        return new_segments

    def _merge_segments(
            self,
            sentences_per_segment: list[list[Sentence]],
            threshold,
            verbose: bool = False
    ):
        segment_map = [0]
        segments = [' '.join([x.text for x in i]) for i in sentences_per_segment]
        for index, (text1, text2) in enumerate(zip(segments[:-1], segments[1:])):
            sim = self._get_similarity(text1, text2)
            if verbose:
                print(f'{text1[:50]}\t{text2[:50]}\t{sim}')
            if sim >= threshold:
                segment_map.append(0)
            else:
                segment_map.append(1)
        return self._index_mapping(segment_map)

    def _index_mapping(
            self,
            segment_map: list[int]
    ) -> list[list[int]]:
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
        try:
            sentence_1 = [i.text.strip() for i in self._pipeline(text1).sentences if len(i.text.split(' ')) > 1]
            sentence_2 = [i.text.strip() for i in self._pipeline(text2).sentences if len(i.text.split(' ')) > 1]
            embeding_1 = self.model.encode(sentence_1)
            embeding_2 = self.model.encode(sentence_2)
            embeding_1 = np.mean(embeding_1, axis=0).reshape(1, -1)
            embeding_2 = np.mean(embeding_2, axis=0).reshape(1, -1)

            if np.any(np.isnan(embeding_1)) or np.any(np.isnan(embeding_2)):
                return 1

            sim = cosine_similarity(embeding_1, embeding_2)
            return sim
        except Exception as e:
            print(e)
            print(f'{text1 = }')
            print(f'{text2 = }\n')
            print(f'{sentence_1 = }')
            print(f'{sentence_2 = }')
            raise

    def _text_tilling(self) -> list[list[Sentence]]:
        text = '\n\n\t'.join([x.text for x in self._sentences])
        segment2 = self.tt.tokenize_sentences(self._sentences)
        # segment = self.tt.tokenize(text)
        # segment = [i.replace("\n\n\t", ' ') for i in segment]
        return segment2
