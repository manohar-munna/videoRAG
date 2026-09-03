#!/usr/bin/env python3
"""Generate models/manifest.json + the exact upload layout for the model server.

The apps download their weights from a static file server (any HTTP host / Docker
container serving a directory). This script hashes the local weights, writes a
manifest the clients read, and stages a `dist/models/` tree laid out exactly as it
must appear on the server. Point the apps' MODEL_BASE_URL at wherever that tree is
served and each install pulls only the files its platform/profile needs.

Run from the repo root:  python scripts/gen_model_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist" / "models"

# (source path on disk, server-relative path, on-device destination relative to the
#  app's models root). dest is what the client writes locally.
ANDROID = [
    ("models/qwen2_vl_2b/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
     "android/Qwen2-VL-2B-Instruct-Q4_K_M.gguf", "Qwen2-VL-2B-Instruct-Q4_K_M.gguf"),
    ("models/qwen2_vl_2b/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf",
     "android/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"),
    ("models/mobileclip_onnx/mobileclip_image.onnx",
     "android/mobileclip_image.onnx", "mobileclip_image.onnx"),
    ("models/mobileclip_onnx/mobileclip_text.onnx",
     "android/mobileclip_text.onnx", "mobileclip_text.onnx"),
]

# Windows CLIP is fetched by open_clip from HuggingFace on first use, so only the
# GGUF weights need hosting. Two selectable runtime profiles (see RUNTIME_PROFILES
# in vlm_process_manager.py); a desktop install downloads only its active profile.
WINDOWS = {
    "desktop": [
        ("models/qwen3_vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
         "windows/qwen3_vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
         "models/qwen3_vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf"),
        ("models/qwen3_vl/mmproj-Qwen3VL-4B-Instruct-F16.gguf",
         "windows/qwen3_vl/mmproj-Qwen3VL-4B-Instruct-F16.gguf",
         "models/qwen3_vl/mmproj-Qwen3VL-4B-Instruct-F16.gguf"),
    ],
    "mobile": [
        ("models/qwen2_vl_2b/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
         "windows/qwen2_vl_2b/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
         "models/qwen2_vl_2b/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"),
        ("models/qwen2_vl_2b/mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
         "windows/qwen2_vl_2b/mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
         "models/qwen2_vl_2b/mmproj-Qwen2-VL-2B-Instruct-f16.gguf"),
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(src: str, url_path: str, dest: str, stage: bool) -> dict:
    p = ROOT / src
    if not p.exists():
        raise SystemExit(f"missing source weight: {src}")
    size = p.stat().st_size
    print(f"  hashing {src} ({size/1e6:.0f} MB) ...", flush=True)
    digest = sha256(p)
    if stage:
        out = DIST / url_path
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists() or out.stat().st_size != size:
            shutil.copy2(p, out)
    return {"path": url_path, "dest": dest, "bytes": size, "sha256": digest}


def main() -> None:
    import sys
    stage = "--stage" in sys.argv  # copy files into dist/models/ ready to upload
    print("Android weights:")
    android = [entry(s, u, d, stage) for (s, u, d) in ANDROID]
    print("Windows weights:")
    windows = {prof: [entry(s, u, d, stage) for (s, u, d) in items]
               for prof, items in WINDOWS.items()}

    manifest = {"version": 1, "android": android, "windows": windows}
    out_manifest = (DIST / "manifest.json") if stage else (ROOT / "dist" / "manifest.json")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2))

    def total(items):
        return sum(e["bytes"] for e in items) / 1e9

    print("\n==== upload layout (dist/models/) ====")
    print(f"  manifest.json")
    for e in android:
        print(f"  {e['path']:<52} {e['bytes']/1e6:8.0f} MB")
    for prof, items in windows.items():
        for e in items:
            print(f"  {e['path']:<52} {e['bytes']/1e6:8.0f} MB")
    print("\n==== per-install download size ====")
    print(f"  Android            {total(android):.2f} GB")
    print(f"  Windows desktop 4B {total(windows['desktop']):.2f} GB")
    print(f"  Windows mobile 2B  {total(windows['mobile']):.2f} GB")
    print(f"\nmanifest written to {out_manifest}")
    if not stage:
        print("(re-run with --stage to also copy the weights into dist/models/ "
              "in this exact layout, ready to upload)")


if __name__ == "__main__":
    main()
