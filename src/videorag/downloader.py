"""Fetch the desktop VLM weights from a static model server on first run.

The Windows/desktop app loads GGUF weights from ``models/`` under the project root
(``vlm_process_manager.RUNTIME_PROFILES``). Those are 2.3-3.3 GB depending on the active
profile and are not shipped with the source, so this module downloads them on demand
from the same manifest the Android app uses. MobileCLIP is not downloaded here: the
embedder fetches it from HuggingFace via open_clip on first use.

Because the weights live inside the project's own ``models/`` directory, removing the
project (or the folder a packaged build unpacks into) removes them too - so "uninstall
deletes the local models" needs no separate step on Windows.

Resumes partial downloads with an HTTP Range request and verifies each file against its
manifest sha256, so an interrupted or corrupt fetch fails here rather than as a cryptic
llama-server load error.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def base_url(explicit: Optional[str] = None) -> str:
    """Resolve the model server root: explicit arg > env > config.yaml > empty."""
    if explicit:
        return explicit.rstrip("/")
    env = os.environ.get("VIDEORAG_MODEL_BASE_URL", "").strip()
    if env:
        return env.rstrip("/")
    try:
        import yaml  # PyYAML is already a dependency
        cfg = yaml.safe_load((_PROJECT_ROOT / "config" / "config.yaml").read_text())
        url = ((cfg or {}).get("models", {}) or {}).get("download_base_url", "")
        if url:
            return str(url).rstrip("/")
    except Exception:
        pass
    return ""


def fetch_manifest(url: Optional[str] = None) -> dict:
    """The manifest shipped in config/, or a remote copy if a base URL is configured.

    Most entries point straight at HuggingFace, so shipping the manifest means there is
    nothing to host for those weights at all. A base URL is only needed to repoint an
    installed copy without regenerating it.
    """
    root = base_url(url)
    if root:
        with urllib.request.urlopen(f"{root}/manifest.json", timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    local = _PROJECT_ROOT / "config" / "model_manifest.json"
    if not local.is_file():
        raise RuntimeError(
            "No model manifest. Run scripts/gen_model_manifest.py, or set "
            "models.download_base_url / VIDEORAG_MODEL_BASE_URL to fetch one."
        )
    return json.loads(local.read_text())


def _entry_url(entry: dict, root: str) -> str:
    """Absolute 'url' (current form) or 'path' relative to the base URL (older form)."""
    if entry.get("url"):
        return entry["url"]
    return f"{root}/{entry['path']}"


def profile_entries(profile: str, url: Optional[str] = None) -> List[dict]:
    man = fetch_manifest(url)
    win = man.get("windows", {})
    if profile not in win:
        raise RuntimeError(f"profile '{profile}' not in manifest (have: {list(win)})")
    return win[profile]


def _present(entry: dict) -> bool:
    p = _PROJECT_ROOT / entry["dest"]
    return p.is_file() and p.stat().st_size == entry["bytes"]


def status(profile: str, url: Optional[str] = None) -> dict:
    """Which of the active profile's weights are present vs missing."""
    try:
        entries = profile_entries(profile, url)
    except Exception as exc:
        return {"configured": False, "error": str(exc),
                "present": [], "missing": [], "ready": False}
    present = [e for e in entries if _present(e)]
    missing = [e for e in entries if not _present(e)]
    return {
        "configured": True,
        "ready": len(missing) == 0,
        "present": [e["dest"] for e in present],
        "missing": [{"dest": e["dest"], "bytes": e["bytes"]} for e in missing],
        "missing_bytes": sum(e["bytes"] for e in missing),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(src_url: str, dest: Path, expected: int, sha: str,
                  on_bytes: Callable[[int], None]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if have > expected:
        part.unlink()
        have = 0

    if have < expected:
        req = urllib.request.Request(src_url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(req, timeout=60) as resp:
            # 200 means the server ignored the Range; restart from zero.
            mode = "ab"
            if have and resp.status == 200:
                have = 0
                part.unlink(missing_ok=True)
                mode = "wb"
            with open(part, mode) as out:
                done = have
                while True:
                    buf = resp.read(1 << 16)
                    if not buf:
                        break
                    out.write(buf)
                    done += len(buf)
                    on_bytes(done)

    if sha and _sha256(part) != sha.lower():
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name} failed checksum; deleted, please retry")
    if dest.exists():
        dest.unlink()
    part.rename(dest)


# ── shared progress state so an HTTP caller can poll a background download ──────────
_STATE: Dict[str, object] = {"running": False, "profile": None, "index": 0, "total": 0,
                             "name": "", "file_done": 0, "file_bytes": 0,
                             "error": None, "done": False}
_LOCK = threading.Lock()


def get_state() -> dict:
    with _LOCK:
        return dict(_STATE)


def download_profile(profile: str, url: Optional[str] = None,
                     on_progress: Optional[Callable[[dict], None]] = None) -> None:
    """Download every missing weight for [profile]; updates module progress state."""
    root = base_url(url)
    entries = profile_entries(profile, url)
    missing = [e for e in entries if not _present(e)]

    with _LOCK:
        _STATE.update(running=True, profile=profile, total=len(missing), index=0,
                      name="", file_done=0, file_bytes=0, error=None, done=False)
    try:
        for i, e in enumerate(missing):
            with _LOCK:
                _STATE.update(index=i, name=Path(e["dest"]).name, file_done=0,
                              file_bytes=e["bytes"])

            def _on_bytes(done: int, _e=e, _i=i):
                with _LOCK:
                    _STATE.update(file_done=done)
                if on_progress:
                    on_progress(get_state())

            _download_one(_entry_url(e, root), _PROJECT_ROOT / e["dest"],
                          e["bytes"], e.get("sha256", ""), _on_bytes)
        with _LOCK:
            _STATE.update(running=False, done=True)
    except Exception as exc:
        with _LOCK:
            _STATE.update(running=False, error=str(exc))
        raise


def start_background(profile: str, url: Optional[str] = None) -> bool:
    """Kick off a download in a daemon thread. Returns False if one is already running."""
    with _LOCK:
        if _STATE.get("running"):
            return False
    threading.Thread(target=lambda: _safe(profile, url), daemon=True).start()
    return True


def _safe(profile: str, url: Optional[str]) -> None:
    try:
        download_profile(profile, url)
    except Exception:
        pass  # error is recorded in _STATE
