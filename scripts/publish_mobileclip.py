#!/usr/bin/env python3
"""Publish the two MobileCLIP ONNX towers and point the apps at them.

These are the only weights this project cannot pull from someone else's HuggingFace
repo: they are exported locally by export_mobileclip_onnx.py and patched by
patch_clip_text_argmax.py (onnxruntime-android rejects the int64 ArgMax the standard
export emits), so they exist nowhere public until you put them there.

Until they are hosted, a fresh Android install downloads its GGUFs, fails on the two
towers and stays on "Download models" - with no way to embed frames or queries, so
search does not work at all.

Run once:

    huggingface-cli login          # needs a token with WRITE access
    python scripts/publish_mobileclip.py

then rebuild the APK. Re-run it if you ever re-export the towers.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    (ROOT / "models/mobileclip_onnx/mobileclip_image.onnx", "mobileclip_image.onnx"),
    (ROOT / "models/mobileclip_onnx/mobileclip_text.onnx", "mobileclip_text.onnx"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="videorag-mobileclip",
                    help="repo name under your account (default: videorag-mobileclip)")
    ap.add_argument("--private", action="store_true",
                    help="create the repo private - only for testing; the app downloads "
                         "without credentials, so a private repo will 401 on device")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed:  pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        print("Not logged in. Run:  huggingface-cli login\n"
              "Create a token with WRITE access at "
              "https://huggingface.co/settings/tokens", file=sys.stderr)
        return 1

    for src, _ in FILES:
        if not src.exists():
            print(f"missing weight: {src.relative_to(ROOT)}\n"
                  "Export it first with scripts/export_mobileclip_onnx.py", file=sys.stderr)
            return 1

    repo_id = args.repo if "/" in args.repo else f"{user}/{args.repo}"
    print(f"Publishing to {repo_id} as {user}")
    api.create_repo(repo_id=repo_id, repo_type="model",
                    private=args.private, exist_ok=True)

    for src, name in FILES:
        print(f"  uploading {name}  ({src.stat().st_size / 1e6:.0f} MB) ...", flush=True)
        api.upload_file(path_or_fileobj=str(src), path_in_repo=name,
                        repo_id=repo_id, repo_type="model")

    base = f"https://huggingface.co/{repo_id}/resolve/main"
    print(f"\nUploaded. Regenerating the manifest against {base}")
    rc = subprocess.call([sys.executable, str(ROOT / "scripts/gen_model_manifest.py"),
                          "--onnx-base", base])
    if rc != 0:
        print("manifest generation failed", file=sys.stderr)
        return rc

    print("\nNow rebuild the APK so it ships the new manifest:")
    print("    cd android && ./gradlew assembleDebug")
    if args.private:
        print("\nWARNING: the repo is private. The app downloads without credentials, so "
              "make it public before sending the APK to anyone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
