"""
embedder.py
-----------
Wraps ``sentence-transformers`` to produce dense text embeddings for
CCTV document chunks and natural-language queries.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_BATCH_SIZE = 64


class TextEmbedder:
    """Produces dense embeddings using a Sentence-Transformers model.

    Args:
        model_name: Name of the Sentence-Transformers model to load.
            Defaults to ``'all-MiniLM-L6-v2'``.

    Example::

        embedder = TextEmbedder()
        vecs = embedder.embed(["hello world", "goodbye world"])
        query_vec = embedder.embed_query("who is at the gate?")
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        logger.info("Loading sentence-transformer model '%s'", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Model loaded successfully")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts in batches.

        Args:
            texts: Strings to embed.

        Returns:
            A 2-D float32 numpy array of shape ``(len(texts), dim)``.
        """
        if not texts:
            return np.empty((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)

        logger.info("Embedding %d texts in batches of %d", len(texts), _BATCH_SIZE)
        all_embeddings: List[np.ndarray] = []

        batches = [
            texts[i : i + _BATCH_SIZE]
            for i in range(0, len(texts), _BATCH_SIZE)
        ]

        for batch in tqdm(batches, desc="Embedding batches", unit="batch"):
            embeddings = self._model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            all_embeddings.append(embeddings.astype(np.float32))

        result = np.vstack(all_embeddings)
        logger.info("Produced embeddings of shape %s", result.shape)
        return result

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            query: The natural-language query.

        Returns:
            A 1-D float32 numpy array of shape ``(dim,)``.
        """
        logger.debug("Embedding query: '%s'", query[:80])
        embedding = self._model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return embedding.astype(np.float32)

    @property
    def dimension(self) -> int:
        """Embedding dimensionality of the underlying model."""
        return self._model.get_sentence_embedding_dimension()
