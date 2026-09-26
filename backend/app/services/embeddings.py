from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings


class EmbeddingService:

    def __init__(self):
        settings = get_settings()

        self.model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return vector.tolist()

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:

        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return vectors.tolist()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()