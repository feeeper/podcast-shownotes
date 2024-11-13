import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingBuilder:
    def __init__(self, model_name: str = 'deepvk/USER-bge-m3') -> None:
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts: list[str]) -> np.array:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )
        return embeddings
