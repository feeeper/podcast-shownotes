from __future__ import annotations

import json
from logging import getLogger
import numpy as np
from openai import OpenAI
import fuzzysearch

from stop_words import get_stop_words

from .sentences import Sentence


logger = getLogger('llm_text_segmentation')

class LlmTextSegmentation:
    def __init__(
        self,
        sentences: list[Sentence],
        api_key: str,
        base_url: str
    ):
        self._sentences = sentences
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def _find_nearest(
        self,
        actual: list[int],
        expected: int,
        previous: int
    ) -> int:
        if expected in actual:
            return expected
        
        greater_than_previous = [x for x in actual if x >= previous]
        if len(greater_than_previous) == 0:
            # not sure about this decision
            return actual[-1]
        
        candidates = [num for num in actual if num > previous]
        if not candidates:
            return None
        return min(candidates, key=lambda num: abs(num - expected))
    
    def _adjust_delimiters(
        self,
        predicted_delimiters: list[int],
        min_length: int = 5
    ) -> list[int]:
        if not predicted_delimiters:
            return []

        result = [predicted_delimiters[0]]
        
        for i in range(1, len(predicted_delimiters)):
            if predicted_delimiters[i] - result[-1] >= min_length:
                result.append(predicted_delimiters[i])
        
        return result

    def _get_delimiters(
        self,
        response: dict,
        sentences: list[str],
        min_topic_len: int = 5,
        verbose: int = 0
    ) -> list[int]:
        previous_index = -1
        predicted_delimiters = []
        for item in response:
            expected_index = item['index']
            try:
                actual_index = [i for i, x in enumerate(sentences) if x == item['text']]
                
                if not actual_index:
                    actual_index = [i for i, x in enumerate(sentences) if fuzzysearch.find_near_matches(x, item['text'], max_l_dist=1)]
                
                    
                if not actual_index:
                    logger.warning(f'Warning: \"{item["text"]}\" (index={item["index"]}) not found ')
                    continue

                if len(actual_index) > 1:
                    actual_index = self._find_nearest(actual_index, expected_index, previous_index)
                else:
                    try:
                        actual_index = actual_index[0]
                    except IndexError as ie:
                        logger.exception(ie)

                if actual_index is None:
                    logger.warning(f'Warning: \"{item["text"]}\": actual_index is None ({previous_index=})')
                    continue
                    
                if previous_index > -1 and previous_index > actual_index:
                    logger.warning(f'Warning: \"{item["text"]}\": {previous_index=} is larger than {actual_index=}')
                    continue

                previous_index = actual_index
                predicted_delimiters.append(actual_index)
            except ValueError as e:
                logger.exception(e, exc_info=True, stack_info=True, extra={'item': item})        
        
        return self._adjust_delimiters(predicted_delimiters, min_topic_len)
    
    def _call_llm(
        self,
        sentences: list[str],
        model: str = 'google/gemini-3-flash-preview'
    ) -> dict:
        prompt = """You are a podcast host, and you have recorded an episode that you want to split into distinct topic-based segments.

You are given an array of sentences representing the transcript of the episode. Your task is to identify the starting point of each new topic and return the indexes of the sentences where new topics begin.

Guidelines:
- The first sentence should always be considered the start of a topic.  
- A new topic may begin when there is a noticeable shift in context, such as phrases like "Now let's talk about..." or a clear change in subject matter.  
- You cannot modify the sentences — only determine the starting points of new topics.  

### Output Requirements:
- The output should be a valid JSON array of objects.  
- Each object should have two keys:
- `"index"` — the integer index of the starting sentence  
- `"sentence"` — the sentence itself  
- **Keys and strings must be enclosed in double quotes** (not single quotes).  
- The output must not contain trailing commas or invalid characters.  

### Example:
Input:
[
"Welcome to the show!",
"Today we will discuss climate change.",
"Global temperatures have risen significantly in the last decade.",
"Now let's talk about renewable energy.",
"Solar and wind power are becoming more affordable."
]
Output:
[
{"index": 0, "sentence": "Welcome to the show!"},
{"index": 3, "sentence": "Now let's talk about renewable energy."}
]

### Correct JSON Format Example:
[
{"index": 0, "sentence": "Welcome to the show!"},
{"index": 3, "sentence": "Now let's talk about renewable energy."}
]"""

        response = self._client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': json.dumps(sentences, ensure_ascii=False)}
            ],
            temperature=0,
            model=model,
            response_format={
                'type': 'json_schema',
                'json_schema': {
                    'name': 'sentences',
                    'strict': True,
                    'schema': {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "properties": {
                            "sentences": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index": {
                                            "type": "integer",
                                            "description": "Sentence index."
                                        },
                                        "text": {
                                            "type": "string",
                                            "description": "The sentence."
                                        }
                                    },
                                    "required": ["index", "text"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["sentences"],
                        "additionalProperties": False
                    },
                },
            },
        )

        logger.info(
            'Request completed',
            extra={
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens}
        )
        
        try:
            content = response.choices[0].message.content
            jcontent = json.loads(content)
            return jcontent.get('sentences', {})
        except Exception as e:
            logger.exception(e, exc_info=True, stack_info=True, extra={'response': response})
            logger.exception(f"Content (type: {type(content)}): {content}")
            raise

    def _get_segments(self, sentences: list[Sentence], delimiters: list[int]) -> list[list[Sentence]]:
        if len(delimiters) > 1:
            borders = [{'start': delimiters[i], 'end': delimiters[i+1]} for i in range(0, len(delimiters)-1)]
            borders.append({'start': borders[-1]['end'], 'end': len(sentences)})
        else:
            borders = [{'start': 0, 'end': len(sentences)}]
        result = [sentences[x['start']:x['end']] for x in borders]
        return result
    
    def get_segments(self,
        threshold: float = 0.9,
        verbose: bool = False,
        as_sentences: bool = False
    ) -> list[list[str | Sentence]]:
        sentences = [x.text for x in self._sentences]
        llm_response = self._call_llm(sentences)
        delimiters = self._get_delimiters(llm_response, sentences)
        segments = self._get_segments(self._sentences, delimiters)
        if as_sentences:
            return segments
        
        return [list(map(lambda x: x.text, segment)) for segment in segments]
