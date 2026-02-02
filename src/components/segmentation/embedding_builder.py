import numpy as np
from sentence_transformers import SentenceTransformer

# Module-level cache: one model instance per model name
# per worker process. Avoids reloading ~1-2 GiB onto GPU
# on every DB() / EmbeddingBuilder instantiation.
_model_cache: dict[str, SentenceTransformer] = {}


class EmbeddingBuilder:
    def __init__(self, model_name: str = 'deepvk/USER-bge-m3') -> None:
        if model_name not in _model_cache:
            _model_cache[model_name] = SentenceTransformer(
                model_name
            )
        self.model = _model_cache[model_name]

    def get_embeddings(self, texts: list[str]) -> np.array:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )
        return embeddings
