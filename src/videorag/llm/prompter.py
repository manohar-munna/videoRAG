"""
prompter.py
-----------
Prompt construction and LLM invocation for the CCTV RAG pipeline.

Supports three backends:
* ``'gemini'`` — uses the ``google-generativeai`` SDK (Gemini models).
* ``'openai'`` — uses the ``openai`` Python SDK (compatible with Ollama
  and any OpenAI-compatible endpoint via ``base_url``).
* ``'mock'`` — returns a deterministic, realistic-looking answer for
  offline testing without API credentials.
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert CCTV security analysis assistant. "
    "You are given excerpts from CCTV event logs recorded by multiple "
    "cameras across a monitored facility. Each excerpt includes the camera "
    "identifier, the exact timestamp of the event, and a description of "
    "what was observed. "
    "Your task is to answer the operator's question accurately and concisely "
    "using only the provided evidence. "
    "Always cite the camera name and timestamp when referencing a specific "
    "event. If the evidence is insufficient to answer, say so clearly."
)


class RAGPrompter:
    """Builds RAG prompts for the CCTV analysis assistant.

    The prompt includes a system-level instruction, the retrieved evidence
    chunks, and the operator's query.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_prompt(self, query: str, retrieved_chunks: List[dict]) -> str:
        """Construct a full prompt from the query and retrieved evidence.

        Args:
            query: The operator's natural-language question.
            retrieved_chunks: List of result dicts.  Each must contain a
                ``metadata`` dict with ``camera``, ``start_timestamp``,
                and (optionally) ``description`` keys.

        Returns:
            A formatted string prompt suitable for a chat completion API.
        """
        context_lines: List[str] = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            meta = chunk.get("metadata", {})
            camera = chunk.get("camera", meta.get("camera", "Unknown Camera"))
            timestamp = chunk.get("time_range", meta.get("start_timestamp", meta.get("timestamp", "Unknown Time")))
            description = meta.get("description", chunk.get("text", "No description"))
            score = chunk.get("rerank_score", chunk.get("score", None))
            score_str = f" [relevance={score:.3f}]" if score is not None else ""
            context_lines.append(
                f"[{idx}] Camera: {camera} | Time: {timestamp} | "
                f"Event: {description}{score_str}"
            )

        context_block = "\n".join(context_lines) if context_lines else "No relevant events found."

        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"=== Retrieved CCTV Evidence ===\n"
            f"{context_block}\n\n"
            f"=== Operator Query ===\n"
            f"{query}\n\n"
            f"=== Your Analysis ===\n"
        )
        logger.debug("Built prompt with %d context chunks", len(retrieved_chunks))
        return prompt


class LLMClient:
    """Thin wrapper around an LLM backend for answer generation.

    Args:
        backend: One of ``'openai'`` or ``'mock'``.
        model: Model identifier (used for the OpenAI backend).
        api_key: OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
            environment variable if omitted.
        base_url: Custom base URL for OpenAI-compatible endpoints such
            as Ollama (e.g. ``'http://localhost:11434/v1'``).
    """

    def __init__(
        self,
        backend: str = "mock",
        model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

        if backend == "gemini":
            self._init_gemini()
        elif backend in ("openai", "local"):
            self._init_openai()
        elif backend == "mock":
            logger.info("LLMClient initialised in mock mode")
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'local', 'gemini', 'openai', or 'mock'.")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_gemini(self) -> None:
        """Initialise the Google Generative AI (Gemini) client."""
        try:
            import google.generativeai as genai

            if self.api_key:
                genai.configure(api_key=self.api_key)
            else:
                # Falls back to GOOGLE_API_KEY env variable
                import os
                key = os.environ.get("GOOGLE_API_KEY", "")
                if not key:
                    raise ValueError(
                        "Gemini backend requires an API key. Set GOOGLE_API_KEY "
                        "env variable or pass api_key= to LLMClient."
                    )
                genai.configure(api_key=key)

            self._client = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=_SYSTEM_PROMPT,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 1024,
                    "top_p": 0.95,
                },
            )
            logger.info("Gemini client initialised (model='%s')", self.model)
        except ImportError as exc:
            raise ImportError(
                "The 'google-generativeai' package is required for the Gemini backend. "
                "Install it with: pip install google-generativeai"
            ) from exc

    def _init_openai(self) -> None:
        """Initialise the OpenAI/Local SDK client."""
        try:
            import openai  # noqa: F401 – imported for side-effects
            import urllib.request
            import subprocess
            import time
            from pathlib import Path

            kwargs: dict = {}

            if self.backend == "local":
                base_url = self.base_url or "http://127.0.0.1:8080/v1"
                api_key = self.api_key or "local"
                kwargs["base_url"] = base_url
                kwargs["api_key"] = api_key

                # Check if server is running
                server_url = base_url.rstrip("/") + "/models"
                server_running = False
                try:
                    with urllib.request.urlopen(server_url, timeout=2) as resp:
                        if resp.status == 200:
                            server_running = True
                except Exception:
                    server_running = False

                if not server_running:
                    logger.info("Local server at %s not detected. Attempting to start llama-server...", base_url)
                    project_root = Path(__file__).resolve().parent.parent.parent.parent
                    llama_server = project_root / "tools" / "llama" / "llama-server.exe"
                    model_file = project_root / "Local LLM 3VL 4Q" / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"

                    mmproj_file = project_root / "Local LLM 3VL 4Q" / "mmproj-Qwen3VL-4B-Instruct-F16.gguf"
                    if llama_server.exists() and model_file.exists():
                        cmd = [
                            str(llama_server),
                            "-m", str(model_file),
                        ]
                        if mmproj_file.exists():
                            cmd.extend(["--mmproj", str(mmproj_file)])

                        cmd.extend([
                            "-ngl", "99",
                            "--port", "8080",
                            "--host", "127.0.0.1",
                        ])
                        logger.info("Launching: %s", " ".join(cmd))
                        subprocess.Popen(cmd, cwd=str(project_root))
                        
                        # Wait for server to come up
                        for _ in range(30):
                            time.sleep(1)
                            try:
                                with urllib.request.urlopen(server_url, timeout=2) as resp:
                                    if resp.status == 200:
                                        server_running = True
                                        logger.info("Local llama-server started successfully.")
                                        break
                            except Exception:
                                pass
            else:
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                if self.base_url:
                    kwargs["base_url"] = self.base_url

            self._client = openai.OpenAI(**kwargs)
            logger.info(
                "OpenAI/Local client initialised (backend='%s', model='%s', base_url=%s)",
                self.backend,
                self.model,
                kwargs.get("base_url", self.base_url or "<default>"),
            )
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAI/Local backend. "
                "Install it with: pip install openai"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Generate an answer for *prompt*.

        Args:
            prompt: Full prompt string (system instruction + context + query).

        Returns:
            The model's response as a plain string.
        """
        if self.backend == "gemini":
            return self._generate_gemini(prompt)
        if self.backend in ("openai", "local"):
            return self._generate_openai(prompt)
        return self._generate_mock(prompt)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _generate_gemini(self, prompt: str) -> str:
        """Call the Gemini GenerativeAI API."""
        logger.info("Sending request to Gemini backend (model='%s')", self.model)
        response = self._client.generate_content(prompt)  # type: ignore[union-attr]
        answer: str = response.text or ""
        logger.info("Received Gemini response (%d chars)", len(answer))
        return answer.strip()

    def _generate_openai(self, prompt: str) -> str:
        """Call the OpenAI-compatible chat completions API."""
        logger.info("Sending request to OpenAI backend (model='%s')", self.model)
        response = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        answer: str = response.choices[0].message.content or ""
        logger.info("Received response (%d chars)", len(answer))
        return answer.strip()

    def _generate_mock(self, prompt: str) -> str:
        """Return a realistic mock CCTV analysis response.

        Extracts timestamp and camera information from the prompt to make
        the response look contextually grounded.
        """
        # Pull first camera + timestamp from the prompt evidence block
        camera_match = re.search(r"Camera:\s*([^\|]+)\|", prompt)
        time_match = re.search(r"Time:\s*([^\|]+)\|", prompt)
        event_match = re.search(r"Event:\s*(.+?)(?:\[relevance|$)", prompt, re.MULTILINE)

        camera_str = camera_match.group(1).strip() if camera_match else "CAM-01"
        time_str = time_match.group(1).strip() if time_match else "unknown time"
        event_str = event_match.group(1).strip() if event_match else "unspecified activity"

        # Count evidence items
        evidence_count = prompt.count("[relevance=") or prompt.count("\n[")

        mock_response = (
            f"Based on the retrieved CCTV evidence, the most relevant event "
            f"was recorded by **{camera_str}** at **{time_str}**: "
            f"{event_str}.\n\n"
            f"A total of {max(evidence_count, 1)} related footage segment(s) "
            f"were identified across the monitored cameras. "
            f"The timestamps indicate a sequential progression of activity "
            f"that warrants further review by the security team.\n\n"
            f"*[Mock response — connect a real LLM backend for live inference.]*"
        )
        logger.info("Mock LLM response generated (%d chars)", len(mock_response))
        return mock_response
