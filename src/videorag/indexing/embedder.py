"""
embedder.py
-----------
Multimodal CLIP embedder using ``sentence-transformers`` to produce dense
embeddings for CCTV keyframe images and natural-language text queries.
"""

import logging
from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "clip-ViT-B-32"
_BATCH_SIZE = 32


class MultimodalEmbedder:
    """Produces 512-D multimodal embeddings using a CLIP Sentence-Transformers model.

    Supports direct image encoding (for instant ingestion) and text encoding
    (for natural-language queries and captions).

    Args:
        model_name: Name of the Sentence-Transformers CLIP model to load.
            Defaults to ``'clip-ViT-B-32'``.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        logger.info("Loading multimodal CLIP model '%s'...", model_name)
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension() or 512
        logger.info("Multimodal CLIP model loaded successfully (dimension=%d)", self.dimension)

    # ------------------------------------------------------------------
    # Image Embedding API (Ingestion)
    # ------------------------------------------------------------------

    def embed_image(self, image_path_or_pil: Union[str, Path, Image.Image]) -> np.ndarray:
        """Embed a single image file or PIL Image object into a 1-D float32 vector.

        Args:
            image_path_or_pil: File path to image or loaded PIL Image.

        Returns:
            A 1-D float32 numpy array of shape ``(dimension,)``.
        """
        if isinstance(image_path_or_pil, (str, Path)):
            img = Image.open(str(image_path_or_pil)).convert("RGB")
        else:
            img = image_path_or_pil.convert("RGB")

        emb = self._model.encode(
            img,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return np.ascontiguousarray(emb, dtype=np.float32).flatten()

    def embed_images(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = _BATCH_SIZE,
    ) -> np.ndarray:
        """Embed a list of image paths or PIL Image objects in batches.

        Args:
            images: List of image paths or PIL Image objects.
            batch_size: Batch size for encoding.

        Returns:
            A 2-D float32 numpy array of shape ``(len(images), dimension)``.
        """
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)

        logger.info("Embedding %d images in batches of %d...", len(images), batch_size)
        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(images), batch_size):
            batch_items = images[i : i + batch_size]
            pil_batch: List[Image.Image] = []
            for item in batch_items:
                if isinstance(item, (str, Path)):
                    pil_batch.append(Image.open(str(item)).convert("RGB"))
                else:
                    pil_batch.append(item.convert("RGB"))

            batch_embs = self._model.encode(
                pil_batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            all_embeddings.append(np.ascontiguousarray(batch_embs, dtype=np.float32))

        result = np.vstack(all_embeddings)
        logger.info("Produced image embeddings of shape %s", result.shape)
        return result

    # ------------------------------------------------------------------
    # Text Embedding API (Query / Retrieval)
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single natural-language text query string.

        Args:
            query: The natural-language search query.

        Returns:
            A 1-D float32 numpy array of shape ``(dimension,)``.
        """
        logger.debug("Embedding query: '%s'", query[:80])
        embedding = self._model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return np.ascontiguousarray(embedding, dtype=np.float32).flatten()

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings in batches.

        Args:
            texts: Strings to embed.

        Returns:
            A 2-D float32 numpy array of shape ``(len(texts), dimension)``.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        logger.info("Embedding %d texts in batches of %d...", len(texts), _BATCH_SIZE)
        all_embeddings: List[np.ndarray] = []

        batches = [
            texts[i : i + _BATCH_SIZE]
            for i in range(0, len(texts), _BATCH_SIZE)
        ]

        for batch in tqdm(batches, desc="Embedding text batches", unit="batch"):
            embeddings = self._model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            all_embeddings.append(np.ascontiguousarray(embeddings, dtype=np.float32))

        result = np.vstack(all_embeddings)
        logger.info("Produced text embeddings of shape %s", result.shape)
        return result


# Backwards-compatibility alias
TextEmbedder = MultimodalEmbedder
