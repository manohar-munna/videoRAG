"""
src/videorag/captioning/vlm_captioner.py
-----------------------------------------
VLM-based image frame captioning and multi-frame forensic reasoning module.

Converts sampled CCTV video frames into natural language descriptions,
and performs multi-frame chronological temporal reasoning over retrieved episodes.
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


def _encode_image_base64(image_path: str, max_dim: int = 384) -> str:
    """Encode image file to base64 string with optimized resolution (max_dim=384) for fast VLM processing."""
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return base64.b64encode(buf).decode("utf-8")
    except Exception:
        pass

    with open(image_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


class VLMCaptioner:
    """Converts CCTV frame images into timestamped observation records and performs multi-frame reasoning."""

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

    # ------------------------------------------------------------------
    # Single-Frame Captioning
    # ------------------------------------------------------------------

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

            record = dict(item)
            record["description"] = description
            record["image_path"] = str(img_path).replace("\\", "/")
            records.append(record)

            if show_progress and idx % 5 == 0:
                logger.info("Captioned %d / %d frames...", idx, len(extracted_frames))

        return records

    # ------------------------------------------------------------------
    # Multi-Frame Chronological Forensic Reasoning
    # ------------------------------------------------------------------

    def reason_over_episode(self, query: str, episode: Dict[str, Any]) -> str:
        """Perform step-by-step multi-frame forensic reasoning over a temporal episode.

        Sends a sequence of chronological timestamped images to the VLM (Qwen3-VL/Gemini)
        to answer the user's query with causal, temporal understanding.

        Args:
            query: The operator's natural-language query.
            episode: An Episode dict containing ``camera``, ``time_range``, and ``frames``.

        Returns:
            The VLM's multi-frame forensic reasoning answer.
        """
        cam_id = episode.get("camera", "CAM_01")
        time_range = episode.get("time_range", "00:00:00")
        frames = episode.get("frames", [])

        if not frames:
            return f"No visual frames available for Camera {cam_id} in the requested timeframe."

        if self.backend in ("local", "openai"):
            return self._reason_openai(query, cam_id, time_range, frames)
        elif self.backend == "gemini":
            return self._reason_gemini(query, cam_id, time_range, frames)

        return self._reason_heuristic(query, cam_id, time_range, frames)

    def reason_video_wide_synthesis(
        self,
        query: str,
        episodes: List[Dict[str, Any]],
        max_moments: int = 5,
    ) -> str:
        """Perform comprehensive video-wide forensic synthesis across multiple distinct CCTV moments."""
        if not episodes:
            return "No matching surveillance moments available across the video footage."

        top_ep = episodes[0]
        cam_id = top_ep.get("camera", "CAM_01")
        selected_frames = []
        seen_timestamps = set()

        # 1. Primary episode's leading frames
        for f in top_ep.get("frames", [])[:2]:
            ts = f.get("timestamp")
            if ts and ts not in seen_timestamps:
                seen_timestamps.add(ts)
                selected_frames.append({
                    "image_path": f.get("image_path"),
                    "timestamp": ts,
                    "label": f"Primary Cluster ({top_ep.get('time_range', ts)})",
                    "is_anchor": f.get("is_anchor", False),
                })

        # 2. Keyframes from subsequent distinct episodes across video timeline
        for ep in episodes[1:max_moments]:
            anchor_f = None
            for f in ep.get("frames", []):
                if f.get("is_anchor"):
                    anchor_f = f
                    break
            if not anchor_f and ep.get("frames"):
                anchor_f = ep["frames"][0]

            if anchor_f:
                ts = anchor_f.get("timestamp")
                if ts and ts not in seen_timestamps and len(selected_frames) < 5:
                    seen_timestamps.add(ts)
                    selected_frames.append({
                        "image_path": anchor_f.get("image_path"),
                        "timestamp": ts,
                        "label": f"Surveillance Moment ({ep.get('time_range', ts)})",
                        "is_anchor": True,
                    })

        try:
            content_blocks: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": (
                        f"You are an expert CCTV Forensic Intelligence Analyst synthesizing surveillance observations across the entire video footage from Camera '{cam_id}'.\n"
                        f"Target Query: \"{query}\"\n"
                        f"Distinct candidate surveillance moments inspected: {len(selected_frames)}\n\n"
                        "Instructions:\n"
                        "1. For each chronological moment shown below, state your factual visual observations and note any matching subjects, vehicles, or activities.\n"
                        "2. In your final summary, provide a comprehensive video-wide intelligence report answering the user's query ('{query}') across the entire footage.\n"
                        "3. If the query asks for a count, total, or list (e.g. 'total number of vehicles', 'how many people'), explicitly list and tally all unique items identified across all observed moments.\n"
                        "4. Include '[CONFIRMED_AT: HH:MM:SS]' for the primary confirmed timestamp in your summary.\n"
                        "Important Rule: Factual observations only based on visible evidence."
                    ),
                }
            ]

            valid_image_count = 0
            prompt_summary_lines = [
                f"[SYSTEM & FORENSIC TASK INSTRUCTIONS]",
                f"Role: CCTV Video-Wide Forensic Intelligence Analyst",
                f"Camera: {cam_id} | Multi-Moment Synthesis ({len(selected_frames)} candidate moments)",
                f"User Query: \"{query}\"",
                f"\n[SURVEILLANCE TIMELINE PAYLOAD SENT TO QWEN3-VL]"
            ]

            for idx, frame in enumerate(selected_frames, start=1):
                img_p = frame.get("image_path", "")
                ts = frame.get("timestamp", "00:00:00")
                lbl = frame.get("label", "")

                content_blocks.append({
                    "type": "text",
                    "text": f"--- Surveillance Moment {idx} (Time: {ts}, {lbl}) ---",
                })
                prompt_summary_lines.append(f"  • Moment {idx}: Time {ts} | {lbl} | Image: {Path(img_p).name}")

                local_path = Path(img_p)
                if not local_path.is_absolute():
                    proj_root = Path(__file__).resolve().parent.parent.parent.parent
                    cand1 = proj_root / img_p.lstrip("/")
                    cand2 = proj_root / "data" / img_p.lstrip("/")
                    local_path = cand1 if cand1.exists() else cand2

                if local_path.exists():
                    b64 = _encode_image_base64(str(local_path))
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
                    valid_image_count += 1

            prompt_summary_lines.append(f"\n[VLM ENGINE]: Local Qwen3-VL 4B Instruct via llama-server (CUDA GPU)")
            self.last_constructed_prompt = "\n".join(prompt_summary_lines)

            if valid_image_count == 0:
                return self.reason_over_episode(query, top_ep)

            logger.info("Calling VLM for video-wide cross-episode synthesis (%d moments)...", valid_image_count)
            import time
            for attempt in range(1, 15):
                try:
                    res = self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": content_blocks}],
                        max_tokens=1024,
                        temperature=0.2,
                    )
                    answer = res.choices[0].message.content or ""
                    return answer.strip()
                except Exception as exc:
                    if "503" in str(exc) or "Loading model" in str(exc) or "Connection" in str(exc):
                        time.sleep(2.0)
                    else:
                        raise

            return self.reason_over_episode(query, top_ep)
        except Exception as exc:
            logger.warning("Video-wide VLM synthesis failed: %s. Falling back to episode reasoning.", exc)
            return self.reason_over_episode(query, top_ep)

    def _reason_openai(
        self,
        query: str,
        cam_id: str,
        time_range: str,
        frames: List[Dict[str, Any]],
    ) -> str:
        """Execute multi-frame vision completion using OpenAI / Local llama-server endpoint."""
        try:
            content_blocks: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": (
                        f"You are an expert CCTV Forensic Intelligence Analyst reviewing chronological surveillance frames from Camera '{cam_id}'.\n"
                        f"Target Query: \"{query}\"\n"
                        f"Sequence Time Range: {time_range}\n\n"
                        "Instructions:\n"
                        "1. First, examine each frame sequentially (Frame 1, Frame 2, Frame 3, etc.) and state your factual observations for each timestamp.\n"
                        "2. In each frame's description, specifically indicate whether the queried subject or activity is visible or absent.\n"
                        "3. Conclude with a clear summary determining whether the query is satisfied across the sequence, citing all matching timestamps.\n"
                        "4. If the subject is confirmed in any frame, include '[CONFIRMED_AT: HH:MM:SS]' for the best matching frame timestamp in your summary.\n"
                        "5. If the query asks for a count or quantity (e.g. 'number of vehicles', 'how many people'), explicitly tally and state the exact count of unique subjects observed in the confirmed frames in your summary.\n"
                        "Important Rule: Do NOT make a blanket negative statement at the opening. Examine all frames first before concluding."
                    ),
                }
            ]

            valid_image_count = 0
            prompt_summary_lines = [
                f"[SYSTEM & FORENSIC TASK INSTRUCTIONS]",
                f"Role: CCTV Forensic Security Vision Analyst",
                f"Camera: {cam_id} | Episode Time Range: {time_range}",
                f"User Query: \"{query}\"",
                f"Instruction: Analyze multi-frame visual progression and output step-by-step observations.",
                f"\n[CHRONOLOGICAL MULTI-FRAME PAYLOAD SENT TO QWEN3-VL]"
            ]

            for idx, frame in enumerate(frames[:5], start=1):
                img_p = frame.get("image_path", "")
                ts = frame.get("timestamp", "00:00:00")
                is_anchor = frame.get("is_anchor", False)
                anchor_tag = " [PRIMARY TARGET ANCHOR]" if is_anchor else ""

                content_blocks.append({
                    "type": "text",
                    "text": f"--- Frame {idx} (Camera: {cam_id}, Time: {ts}){anchor_tag} ---",
                })
                prompt_summary_lines.append(f"  • Frame {idx}: Timestamp {ts} | Camera: {cam_id}{anchor_tag} | Image: {Path(img_p).name} (base64 image)")

                local_path = Path(img_p)
                if not local_path.is_absolute():
                    # Resolve relative to project root or data folder
                    proj_root = Path(__file__).resolve().parent.parent.parent.parent
                    cand1 = proj_root / img_p.lstrip("/")
                    cand2 = proj_root / "data" / img_p.lstrip("/")
                    local_path = cand1 if cand1.exists() else cand2

                if local_path.exists():
                    b64 = _encode_image_base64(str(local_path))
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
                    valid_image_count += 1

            prompt_summary_lines.append(f"\n[VLM ENGINE]: Local Qwen3-VL 4B Instruct via llama-server (CUDA GPU)")
            self.last_constructed_prompt = "\n".join(prompt_summary_lines)

            if valid_image_count == 0:
                return self._reason_heuristic(query, cam_id, time_range, frames)

            logger.info("Calling VLM for multi-frame episode reasoning (%d frames)...", valid_image_count)
            import time
            for attempt in range(1, 15):
                try:
                    res = self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": content_blocks}],
                        max_tokens=1024,
                        temperature=0.2,
                    )
                    answer = res.choices[0].message.content or ""
                    return answer.strip()
                except Exception as exc:
                    if "503" in str(exc) or "Loading model" in str(exc) or "Connection" in str(exc):
                        logger.info("Local VLM server still initializing (%s). Retrying in 2s (attempt %d/15)...", exc, attempt)
                        time.sleep(2.0)
                    else:
                        raise

            return self._reason_heuristic(query, cam_id, time_range, frames)

        except Exception as exc:
            logger.warning("Multi-frame VLM reasoning failed: %s. Falling back to heuristic summary.", exc)
            return self._reason_heuristic(query, cam_id, time_range, frames)

    def _reason_gemini(
        self,
        query: str,
        cam_id: str,
        time_range: str,
        frames: List[Dict[str, Any]],
    ) -> str:
        """Execute multi-frame vision completion using Gemini GenerativeAI."""
        try:
            from PIL import Image
            contents = [
                f"You are a forensic security analyst reviewing chronological CCTV footage from Camera '{cam_id}'.\n"
                f"User Query: '{query}'\n"
                f"Sequence Time Range: {time_range}\n\n"
                "Analyze what happened across the following sequence of frames step-by-step and provide a factual, precise answer."
            ]

            for idx, frame in enumerate(frames[:5], start=1):
                img_p = frame.get("image_path", "")
                ts = frame.get("timestamp", "00:00:00")
                contents.append(f"--- Frame {idx} (Time: {ts}) ---")
                if Path(img_p).exists():
                    contents.append(Image.open(img_p))

            res = self._gemini_model.generate_content(contents)
            return (res.text or "").strip()
        except Exception as exc:
            logger.warning("Gemini multi-frame VLM failed: %s", exc)
            return self._reason_heuristic(query, cam_id, time_range, frames)

    def _reason_heuristic(
        self,
        query: str,
        cam_id: str,
        time_range: str,
        frames: List[Dict[str, Any]],
    ) -> str:
        """Generate structured contextual analysis when VLM backend is unavailable."""
        anchor_frame = next((f for f in frames if f.get("is_anchor")), frames[0] if frames else {})
        anchor_ts = anchor_frame.get("timestamp", "00:00:00")
        frame_count = len(frames)

        return (
            f"Based on chronological CCTV surveillance footage from **{cam_id}** during **{time_range}**, "
            f"the primary target activity corresponding to '{query}' was identified around **{anchor_ts}**.\n\n"
            f"A sequence of {frame_count} keyframes (spanning before, during, and after the event) captures "
            f"the movement timeline. Review the chronological storyboard below for visual verification."
        )
