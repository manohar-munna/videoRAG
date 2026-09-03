#!/usr/bin/env python3
"""Download the desktop VLM weights for the active runtime profile.

    python scripts/download_models.py                 # active profile from config
    python scripts/download_models.py --profile mobile
    python scripts/download_models.py --url https://models.example.com/videorag

The URL can also come from config.yaml (models.download_base_url) or the
VIDEORAG_MODEL_BASE_URL environment variable. MobileCLIP is fetched separately by
open_clip on first server run and is not downloaded here.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from videorag import downloader  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=None,
                    help="desktop (4B) or mobile (2B); default: config.yaml llm profile")
    ap.add_argument("--url", default=None, help="model server root URL")
    args = ap.parse_args()

    profile = args.profile or _default_profile()
    print(f"Model server : {downloader.base_url(args.url) or '(not configured)'}")
    print(f"Profile      : {profile}")

    st = downloader.status(profile, args.url)
    if not st.get("configured"):
        print("\nNo model server configured. Set models.download_base_url in "
              "config/config.yaml or VIDEORAG_MODEL_BASE_URL.", file=sys.stderr)
        return 2
    if st.get("error"):
        print(f"\nCould not read manifest: {st['error']}", file=sys.stderr)
        return 2
    if st["ready"]:
        print("\nAll weights already present. Nothing to download.")
        return 0

    print(f"Missing      : {len(st['missing'])} file(s), "
          f"{st['missing_bytes'] / 1e9:.2f} GB\n")

    last = [0.0]

    def show(state: dict) -> None:
        now = time.time()
        if now - last[0] < 0.5:
            return
        last[0] = now
        done, tot = state["file_done"], max(state["file_bytes"], 1)
        pct = 100 * done / tot
        print(f"\r  [{state['index'] + 1}/{state['total']}] {state['name']}  "
              f"{done / 1e6:.0f}/{tot / 1e6:.0f} MB ({pct:4.1f}%)   ", end="", flush=True)

    try:
        downloader.download_profile(profile, args.url, on_progress=show)
    except Exception as exc:
        print(f"\n\nDownload failed: {exc}", file=sys.stderr)
        return 1
    print("\n\nAll weights downloaded and verified.")
    return 0


def _default_profile() -> str:
    try:
        import yaml
        root = Path(__file__).resolve().parent.parent
        cfg = yaml.safe_load((root / "config" / "config.yaml").read_text())
        # honour an explicit profile key if present, else infer from the model path
        prof = ((cfg or {}).get("llm", {}) or {}).get("profile")
        if prof:
            return str(prof)
        model = ((cfg or {}).get("llm", {}) or {}).get("model", "")
        return "mobile" if "qwen2_vl_2b" in model or "2B" in model else "desktop"
    except Exception:
        return "desktop"


if __name__ == "__main__":
    raise SystemExit(main())
