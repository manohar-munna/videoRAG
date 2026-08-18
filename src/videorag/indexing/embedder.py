"""
src/videorag/indexing/embedder.py
---------------------------------
Multimodal CLIP & MobileCLIP embedder producing continuous dense embeddings
for CCTV keyframe images and natural-language text queries.

Supports:
- Apple's ultra-lightweight MobileCLIP (`MobileCLIP-S2`, `MobileCLIP-B`, `MobileCLIP-S1`) via `open_clip`
- Multimodal CLIP (`clip-ViT-B-32`, `clip-ViT-L-14`) via `sentence_transformers`
"""

import logging
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "clip-ViT-B-32"
_BATCH_SIZE = 32


class MultimodalEmbedder:
    """Produces 512-D multimodal embeddings using MobileCLIP or Sentence-Transformers CLIP.

    Supports direct image encoding (for instant ~25ms ingestion) and text encoding
    (for natural-language queries and captions).

    Args:
        model_name: Name of the model to load (e.g. ``'MobileCLIP-S2'``, ``'mobileclip'``,
            or ``'clip-ViT-B-32'``).
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.is_mobileclip = "mobileclip" in model_name.lower()

        if self.is_mobileclip:
            self._init_mobileclip(model_name)
        else:
            self._init_sentence_transformer(model_name)

    def _init_mobileclip(self, model_name: str) -> None:
        """Initialize Apple MobileCLIP model via open_clip."""
        try:
            import open_clip
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_name = "MobileCLIP-S2"
            if "s0" in model_name.lower():
                clip_name = "MobileCLIP2-S0"
            elif "s1" in model_name.lower():
                clip_name = "MobileCLIP-S1"
            elif "b" in model_name.lower():
                clip_name = "MobileCLIP-B"

            logger.info("Loading MobileCLIP model '%s' on device '%s'...", clip_name, self._device)
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                clip_name, pretrained="datacompdr", device=self._device
            )
            self._model.eval()
            self._tokenizer = open_clip.get_tokenizer(clip_name)
            self.dimension = 512
            logger.info("MobileCLIP loaded successfully (dimension=%d, device=%s)", self.dimension, self._device)
        except Exception as exc:
            logger.warning("MobileCLIP open_clip init failed (%s). Falling back to clip-ViT-B-32.", exc)
            self.is_mobileclip = False
            self._init_sentence_transformer("clip-ViT-B-32")

    def _init_sentence_transformer(self, model_name: str) -> None:
        """Initialize SentenceTransformer CLIP model."""
        from sentence_transformers import SentenceTransformer
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

        if self.is_mobileclip:
            import torch
            tensor = self._preprocess(img).unsqueeze(0).to(self._device)
            with torch.no_grad():
                features = self._model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten().astype(np.float32)

        emb = self._model.encode(
            img,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
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

        all_embeddings: List[np.ndarray] = []

        if self.is_mobileclip:
            import torch
            for i in range(0, len(images), batch_size):
                batch_items = images[i : i + batch_size]
                tensors = []
                for item in batch_items:
                    if isinstance(item, (str, Path)):
                        img = Image.open(str(item)).convert("RGB")
                    else:
                        img = item.convert("RGB")
                    tensors.append(self._preprocess(img))
                stacked = torch.stack(tensors).to(self._device)
                with torch.no_grad():
                    features = self._model.encode_image(stacked)
                    features = features / features.norm(dim=-1, keepdim=True)
                all_embeddings.append(features.cpu().numpy().astype(np.float32))
            return np.vstack(all_embeddings).astype(np.float32)

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
                normalize_embeddings=True,
            )
            all_embeddings.append(batch_embs)

        return np.vstack(all_embeddings).astype(np.float32)

    # ------------------------------------------------------------------
    # Text Embedding API (Query / Captions)
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string into a 1-D float32 vector in the multimodal space."""
        if self.is_mobileclip:
            import torch
            tokens = self._tokenizer([query]).to(self._device)
            with torch.no_grad():
                features = self._model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten().astype(np.float32)

        emb = self._model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.ascontiguousarray(emb, dtype=np.float32).flatten()

    def embed(self, texts: List[str], batch_size: int = _BATCH_SIZE) -> np.ndarray:
        """Embed a list of text strings into a 2-D numpy array."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.is_mobileclip:
            import torch
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                tokens = self._tokenizer(batch_texts).to(self._device)
                with torch.no_grad():
                    features = self._model.encode_text(tokens)
                    features = features / features.norm(dim=-1, keepdim=True)
                all_embs.append(features.cpu().numpy().astype(np.float32))
            return np.vstack(all_embs).astype(np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.ascontiguousarray(embeddings, dtype=np.float32)


# Backward-compatibility alias
TextEmbedder = MultimodalEmbedder
