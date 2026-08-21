from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings


class EmbeddingService:
    """
    Handles text -> vector conversion.

    The same model must be used for:
        1. Document embeddings during indexing
        2. Query embeddings during retrieval
    """

    def __init__(self) -> None:
        settings = get_settings()

        self.model_name = settings.embedding_model
        self.device = settings.embedding_device

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding for a single piece of text.
        """

        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

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
        """
        Generate embeddings for multiple texts.

        Batch encoding is much faster than calling embed_text()
        repeatedly during offline indexing.
        """

        if not texts:
            return []

        cleaned_texts = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not cleaned_texts:
            return []

        vectors = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return vectors.tolist()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """
    Return one shared embedding model instance.

    Loading SentenceTransformer repeatedly would waste
    significant memory and add unnecessary latency.
    """

    return EmbeddingService()