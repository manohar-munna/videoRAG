#!/usr/bin/env python3
"""Generate the model manifest both apps ship with.

Most weights are public GGUFs on HuggingFace, so the manifest points straight at them:
nothing to host, and size + SHA-256 come from the HF API without downloading anything.
URLs are pinned to the repo's current commit so a re-upload upstream cannot silently
change what a client fetches.

The two MobileCLIP ONNX towers are the exception. They are produced locally by
scripts/export_mobileclip_onnx.py + patch_clip_text_argmax.py (the ArgMax int32 patch
that onnxruntime-android needs), so they exist nowhere public and must be hosted. Point
--onnx-base at wherever you put them - a free HuggingFace model repo of your own is the
easiest option:

    huggingface-cli repo create videorag-mobileclip --type model
    huggingface-cli upload <you>/videorag-mobileclip \\
        models/mobileclip_onnx/mobileclip_image.onnx mobileclip_image.onnx
    huggingface-cli upload <you>/videorag-mobileclip \\
        models/mobileclip_onnx/mobileclip_text.onnx  mobileclip_text.onnx

    python scripts/gen_model_manifest.py \\
        --onnx-base https://huggingface.co/<you>/videorag-mobileclip/resolve/main

Writes the manifest to both places the apps read it from:
  android/app/src/main/assets/model_manifest.json
  config/model_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# GGUF weights that live on HuggingFace: (repo, remote filename, local destination)
#
# Repo choice is not arbitrary - it is pinned to builds verified on-device. ggml-org's
# Qwen2-VL-2B Q4_K_M looks equivalent (same arch, same tensor count, correct sha256) but
# makes this app's vendored llama.cpp emit EOS after 2 tokens: "White truck at 00:02:42."
# instead of a description. bartowski's Q4_K_M is byte-identical to the build every
# quality result in this project was measured against (986047232 bytes, sha256
# 4ef095263343fc12), so that is the one to ship. Same for the f16 projector.
# The Q8_0 projector only exists in ggml-org's repo, and that file IS byte-identical to
# our validated copy (a0ad91f00a7a80dc), so it is safe to take from there.
HF_GGUF = {
    "android": [
        ("bartowski/Qwen2-VL-2B-Instruct-GGUF", "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
         "Qwen2-VL-2B-Instruct-Q4_K_M.gguf"),
        ("ggml-org/Qwen2-VL-2B-Instruct-GGUF", "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf",
         "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"),
    ],
    "windows.desktop": [
        ("unsloth/Qwen3-VL-4B-Instruct-GGUF", "Qwen3-VL-4B-Instruct-Q4_K_M.gguf",
         "models/qwen3_vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf"),
        ("unsloth/Qwen3-VL-4B-Instruct-GGUF", "mmproj-F16.gguf",
         "models/qwen3_vl/mmproj-Qwen3VL-4B-Instruct-F16.gguf"),
    ],
    "windows.mobile": [
        ("bartowski/Qwen2-VL-2B-Instruct-GGUF", "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
         "models/qwen2_vl_2b/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"),
        ("bartowski/Qwen2-VL-2B-Instruct-GGUF", "mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
         "models/qwen2_vl_2b/mmproj-Qwen2-VL-2B-Instruct-f16.gguf"),
    ],
}

# Locally-built ONNX towers that must be hosted (Android only; the desktop embedder
# pulls MobileCLIP from HuggingFace itself via open_clip).
LOCAL_ONNX = [
    ("models/mobileclip_onnx/mobileclip_image.onnx", "mobileclip_image.onnx"),
    ("models/mobileclip_onnx/mobileclip_text.onnx", "mobileclip_text.onnx"),
]

_repo_cache: dict = {}


def hf_repo(repo: str) -> dict:
    if repo not in _repo_cache:
        url = f"https://huggingface.co/api/models/{repo}?blobs=true"
        with urllib.request.urlopen(url, timeout=30) as r:
            _repo_cache[repo] = json.loads(r.read().decode())
    return _repo_cache[repo]


def hf_entry(repo: str, filename: str, dest: str) -> dict:
    info = hf_repo(repo)
    sha_rev = info.get("sha")
    for sib in info.get("siblings", []):
        if sib.get("rfilename") == filename:
            lfs = sib.get("lfs") or {}
            size, digest = lfs.get("size"), (lfs.get("sha256") or lfs.get("oid"))
            if not size or not digest:
                raise SystemExit(f"{repo}/{filename}: no LFS size/sha256 in the HF API")
            # pin to the commit so upstream re-uploads cannot change what clients get
            return {"url": f"https://huggingface.co/{repo}/resolve/{sha_rev}/{filename}",
                    "dest": dest, "bytes": size, "sha256": digest}
    raise SystemExit(f"{filename} not found in {repo}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_entry(src: str, remote_name: str, onnx_base: str) -> dict:
    p = ROOT / src
    if not p.exists():
        raise SystemExit(f"missing locally-built weight: {src}")
    print(f"  hashing {src} ...", flush=True)
    return {"url": f"{onnx_base.rstrip('/')}/{remote_name}", "dest": remote_name,
            "bytes": p.stat().st_size, "sha256": sha256(p)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx-base", default="",
                    help="base URL hosting mobileclip_image.onnx / mobileclip_text.onnx")
    ap.add_argument("--allow-localhost", action="store_true",
                    help="permit a loopback --onnx-base (local testing via `adb reverse`)")
    args = ap.parse_args()

    # A loopback base is genuinely useful while testing - the phone reaches a dev server
    # through `adb reverse tcp:8765 tcp:8765`. It is also silent poison in a shipped
    # build: on someone else's phone 127.0.0.1 is *their* phone, so the CLIP towers never
    # arrive, the app sits on "Download models", and nothing explains why search is dead.
    # This manifest shipped that way once. Make it deliberate rather than accidental.
    if any(h in args.onnx_base for h in ("127.0.0.1", "localhost", "0.0.0.0", "10.0.2.2")):
        if not args.allow_localhost:
            raise SystemExit(
                f"--onnx-base {args.onnx_base!r} is a loopback address, which only works on\n"
                "the machine serving it. Host the towers somewhere reachable (see\n"
                "scripts/publish_mobileclip.py), or pass --allow-localhost for local testing.")
        print(f"!! loopback --onnx-base {args.onnx_base} - for local testing only; this\n"
              "   manifest must not be shipped.", file=sys.stderr)

    print("Resolving GGUF weights from HuggingFace (no download)...")
    android = [hf_entry(*t) for t in HF_GGUF["android"]]
    windows = {
        "desktop": [hf_entry(*t) for t in HF_GGUF["windows.desktop"]],
        "mobile": [hf_entry(*t) for t in HF_GGUF["windows.mobile"]],
    }
    for e in android + windows["desktop"] + windows["mobile"]:
        print(f"  {e['dest']:<52} {e['bytes'] / 1e6:8.0f} MB")

    if args.onnx_base:
        print("Hashing locally-built ONNX towers...")
        android += [local_entry(src, name, args.onnx_base) for src, name in LOCAL_ONNX]
    else:
        print("\n!! --onnx-base not given: the manifest will have NO CLIP towers, so the\n"
              "   Android app cannot search. Host the two files and re-run (see --help).",
              file=sys.stderr)

    manifest = {"version": 2, "android": android, "windows": windows}
    text = json.dumps(manifest, indent=2)
    for out in (ROOT / "android/app/src/main/assets/model_manifest.json",
                ROOT / "config/model_manifest.json"):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"wrote {out.relative_to(ROOT)}")

    def gb(items):
        return sum(e["bytes"] for e in items) / 1e9

    print("\n==== per-install download ====")
    print(f"  Android            {gb(android):.2f} GB"
          f"{'  (CLIP towers MISSING)' if not args.onnx_base else ''}")
    print(f"  Windows desktop 4B {gb(windows['desktop']):.2f} GB")
    print(f"  Windows mobile 2B  {gb(windows['mobile']):.2f} GB")
    print("\nYou host: " + ("nothing — every file comes from HuggingFace"
                            if not LOCAL_ONNX else
                            "only mobileclip_image.onnx + mobileclip_text.onnx (398 MB)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
