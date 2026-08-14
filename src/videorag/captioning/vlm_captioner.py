"""
src/videorag/captioning/vlm_captioner.py
-----------------------------------------
VLM-based image frame captioning module.

Converts sampled CCTV video frames into detailed natural language surveillance descriptions:
- Objects & vehicles (type, color, direction)
- People (clothing, count, behavior, loitering)
- Security events (access, perimeter interaction, anomalous activity)
"""

import base64
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

_CCTV_CAPTION_PROMPT = (
    "Analyze this CCTV surveillance video frame concisely for a security audit log. "
    "Identify and describe in 1-3 sentences: "
    "1. Key subjects (people, clothing color, count, actions, loitering). "
    "2. Vehicles (type, color, movement or parking status). "
    "3. Any notable security events, restricted area presence, or clear activity. "
    "Be direct, objective, and factual. Avoid meta-commentary."
)


def _encode_image_base64(image_path: str, max_dim: int = 640) -> str:
    """Encode image file to base64 string with optional downsampling for fast VLM processing."""
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buf).decode("utf-8")
    except Exception:
        pass

    with open(image_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


class VLMCaptioner:
    """Converts CCTV frame images into timestamped observation records."""

    def __init__(
        self,
        backend: str = "local",
        model: str = "Local LLM 3VL 4Q/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or "http://127.0.0.1:8080/v1"

        if backend in ("local", "openai"):
            self._init_openai_vlm()
        elif backend == "gemini":
            self._init_gemini_vlm()
        elif backend in ("mock", "heuristic"):
            logger.info("VLMCaptioner initialized in heuristic/mock mode.")
        else:
            raise ValueError(f"Unknown VLM backend '{backend}'.")

    def _init_openai_vlm(self) -> None:
        try:
            import openai
            self._client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "local",
            )
            logger.info("VLM client initialised (backend='%s', base_url='%s')", self.backend, self.base_url)
        except ImportError as exc:
            raise ImportError("openai package required for VLM captioner.") from exc

    def _init_gemini_vlm(self) -> None:
        try:
            import google.generativeai as genai
            key = self.api_key or os.environ.get("GOOGLE_API_KEY", "")
            if not key:
                raise ValueError("Gemini VLM requires GOOGLE_API_KEY.")
            genai.configure(api_key=key)
            self._gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("Gemini VLM client initialised.")
        except ImportError as exc:
            raise ImportError("google-generativeai package required for Gemini VLM.") from exc

    def caption_frame(self, image_path: str) -> str:
        """Generate a surveillance description for a single frame image."""
        if not Path(image_path).exists():
            return "Frame image unavailable."

        if self.backend in ("local", "openai"):
            return self._caption_openai(image_path)
        elif self.backend == "gemini":
            return self._caption_gemini(image_path)
        return self._caption_heuristic(image_path)

    def _caption_openai(self, image_path: str) -> str:
        try:
            b64_img = _encode_image_base64(image_path)
            res = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _CCTV_CAPTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                            },
                        ],
                    }
                ],
                max_tokens=256,
                temperature=0.2,
            )
            content = res.choices[0].message.content or ""
            return content.strip()
        except Exception as exc:
            logger.warning("VLM call failed for %s: %s. Falling back to heuristic description.", image_path, exc)
            return self._caption_heuristic(image_path)

    def _caption_gemini(self, image_path: str) -> str:
        try:
            from PIL import Image
            img = Image.open(image_path)
            res = self._gemini_model.generate_content([_CCTV_CAPTION_PROMPT, img])
            return (res.text or "").strip()
        except Exception as exc:
            logger.warning("Gemini VLM failed: %s", exc)
            return self._caption_heuristic(image_path)

    def _caption_heuristic(self, image_path: str) -> str:
        filename = Path(image_path).stem
        parts = filename.split("_")
        ts = parts[-2] if len(parts) >= 2 else "00:00:00"
        return f"CCTV feed frame at {ts}. General traffic and ambient surveillance visual scene observed."

    def caption_batch(
        self,
        extracted_frames: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> List[Dict[str, str]]:
        """Caption a list of extracted frame dicts and return RAG-compatible JSON records."""
        records: List[Dict[str, str]] = []
        logger.info("Generating captions for %d extracted frames...", len(extracted_frames))

        for idx, item in enumerate(extracted_frames, 1):
            cam = item["camera"]
            ts = item["timestamp"]
            img_path = item["image_path"]

            description = self.caption_frame(img_path)

            records.append({
                "camera": cam,
                "timestamp": ts,
                "description": description,
                "image_path": img_path,
            })

            if show_progress and idx % 5 == 0:
                logger.info("Captioned %d / %d frames...", idx, len(extracted_frames))

        return records
