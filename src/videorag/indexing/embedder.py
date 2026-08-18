"""
embedder.py
-----------
Multimodal Apple MobileCLIP & CLIP embedder to produce dense embeddings
for CCTV keyframe images and natural-language text queries.
"""

import logging
import os
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "MobileCLIP-S2"
_BATCH_SIZE = 32
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class MultimodalEmbedder:
    """Produces 512-D multimodal embeddings using Apple MobileCLIP or standard CLIP models.

    Supports direct image encoding (for instant ~15ms ingestion) and text encoding
    (for natural-language queries and captions) in a unified geometric space.

    Args:
        model_name: Name or path of the model to load.
            Options: ``'MobileCLIP-S2'``, ``'MobileCLIP-S0'``, ``'clip-ViT-B-32'``,
            or path to local checkpoint (e.g. ``'models/mobileclip_s2'``).
        model_path: Optional explicit path to model checkpoint/directory.
        device: Device to run inference on (``'cuda'``, ``'cpu'``, or ``None`` for auto-detect).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._is_open_clip = False
        self._model = None
        self._preprocess = None
        self._tokenizer = None

        logger.info("Initializing MultimodalEmbedder (model='%s', device='%s')...", model_name, self.device)
        self._load_model(model_name, model_path)

    def _load_model(self, model_name: str, model_path: Optional[str] = None) -> None:
        """Load MobileCLIP via open_clip or SentenceTransformer."""
        is_mobileclip = (
            "mobileclip" in model_name.lower()
            or (model_path and "mobileclip" in model_path.lower())
        )

        if is_mobileclip:
            try:
                import open_clip

                # Resolve local checkpoint path
                candidate_paths = []
                if model_path:
                    candidate_paths.append(Path(model_path))
                    candidate_paths.append(_PROJECT_ROOT / model_path)

                default_local_dir = _PROJECT_ROOT / "models" / "mobileclip_s2"
                candidate_paths.append(default_local_dir / "open_clip_model.safetensors")
                candidate_paths.append(default_local_dir / "open_clip_pytorch_model.bin")
                candidate_paths.append(default_local_dir)

                local_pt = None
                for cand in candidate_paths:
                    if cand.exists():
                        local_pt = cand
                        break

                m_low = model_name.lower()
                if "s0" in m_low:
                    clip_arch = "MobileCLIP-S0"
                elif "s1" in m_low:
                    clip_arch = "MobileCLIP-S1"
                elif "s2" in m_low:
                    clip_arch = "MobileCLIP-S2"
                elif "s3" in m_low:
                    clip_arch = "MobileCLIP-S3"
                elif "s4" in m_low:
                    clip_arch = "MobileCLIP-S4"
                elif "mobileclip-b" in m_low or m_low.endswith("-b") or m_low == "b":
                    clip_arch = "MobileCLIP-B"
                else:
                    clip_arch = "MobileCLIP-S2"

                if local_pt:
                    pt_arg = str(local_pt) if local_pt.is_file() else str(local_pt / "open_clip_model.safetensors")
                    if not Path(pt_arg).exists() and (local_pt / "open_clip_pytorch_model.bin").exists():
                        pt_arg = str(local_pt / "open_clip_pytorch_model.bin")
                    logger.info("Loading MobileCLIP from local checkpoint: %s", pt_arg)
                    self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                        clip_arch,
                        pretrained=pt_arg,
                        device=self.device,
                    )
                else:
                    logger.info("Loading MobileCLIP via open_clip (pretrained='datacompdr')...")
                    self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                        clip_arch,
                        pretrained="datacompdr",
                        device=self.device,
                    )

                self._tokenizer = open_clip.get_tokenizer(clip_arch)
                self._model.eval()
                self._is_open_clip = True

                # Determine embedding dimension
                with torch.no_grad():
                    dummy_txt = self._tokenizer(["test"]).to(self.device)
                    dummy_feat = self._model.encode_text(dummy_txt)
                    self.dimension = int(dummy_feat.shape[-1])

                logger.info("Apple MobileCLIP model loaded successfully (dimension=%d, device=%s)", self.dimension, self.device)
                return

            except Exception as exc:
                logger.warning("Failed to load MobileCLIP with open_clip (%s). Falling back to SentenceTransformer.", exc)

        # Fallback to SentenceTransformers
        from sentence_transformers import SentenceTransformer
        fallback_name = model_name if not is_mobileclip else "clip-ViT-B-32"
        logger.info("Loading SentenceTransformer model '%s'...", fallback_name)
        self._model = SentenceTransformer(fallback_name, device=self.device)
        self.dimension = self._model.get_sentence_embedding_dimension() or 512
        self._is_open_clip = False
        logger.info("SentenceTransformer CLIP model loaded successfully (dimension=%d)", self.dimension)

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

        if self._is_open_clip:
            tensor = self._preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self._model.encode_image(tensor)
                features /= features.norm(dim=-1, keepdim=True)
                emb = features.cpu().numpy()
        else:
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

            if self._is_open_clip:
                tensors = torch.stack([self._preprocess(img) for img in pil_batch]).to(self.device)
                with torch.no_grad():
                    features = self._model.encode_image(tensors)
                    features /= features.norm(dim=-1, keepdim=True)
                    batch_embs = features.cpu().numpy()
            else:
                batch_embs = self._model.encode(
                    pil_batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
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
        if self._is_open_clip:
            tokens = self._tokenizer([query]).to(self.device)
            with torch.no_grad():
                features = self._model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
                embedding = features.cpu().numpy()
        else:
            embedding = self._model.encode(
                query,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
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
            if self._is_open_clip:
                tokens = self._tokenizer(batch).to(self.device)
                with torch.no_grad():
                    features = self._model.encode_text(tokens)
                    features /= features.norm(dim=-1, keepdim=True)
                    embeddings = features.cpu().numpy()
            else:
                embeddings = self._model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
            all_embeddings.append(np.ascontiguousarray(embeddings, dtype=np.float32))

        result = np.vstack(all_embeddings)
        logger.info("Produced text embeddings of shape %s", result.shape)
        return result


class TextEmbedder(MultimodalEmbedder):
    """Backwards-compatibility alias for MultimodalEmbedder."""
    pass
