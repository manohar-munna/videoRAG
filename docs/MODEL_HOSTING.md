# Hosting the model weights

The apps ship without weights (Android needs ~2.1 GB, Windows 3.4–4.4 GB). On first run
each app downloads what it needs, verifies every file against a SHA-256, and stores it
locally. Uninstall the app and the weights go with it — see **Uninstall** below.

Almost nothing needs hosting. Both apps read a manifest that is **bundled with the build**
(`android/app/src/main/assets/model_manifest.json`, `config/model_manifest.json`), and
every GGUF entry in it points straight at HuggingFace, pinned to a repository commit so an
upstream re-upload cannot change what clients fetch.

| Platform | Downloads | You host |
|---|---|---|
| Android | 2.09 GB | the two MobileCLIP ONNX towers — **398 MB** |
| Windows desktop (Qwen3-VL 4B) | 4.37 GB | nothing |
| Windows mobile (Qwen2-VL 2B) | 3.36 GB | nothing |

The Windows figures include 1.04 GB every desktop install needs whatever the profile, because
a source checkout has none of it (`models/` and `tools/` are gitignored):

| What | Size | Source | Without it |
|---|---|---|---|
| llama.cpp runtime b10549 (CUDA 12.4) | 251 MB | GitHub release | the VLM cannot start at all |
| CUDA 12.4 runtime libraries | 391 MB | GitHub release | GPU offload unavailable |
| MobileCLIP-S2 embedder | 398 MB | `apple/MobileCLIP-S2-OpenCLIP` | embedder silently falls back to `clip-ViT-B-32`, a different embedding space |

The two runtime zips are unpacked into `tools/llama/` and then deleted; GitHub publishes a
SHA-256 for each release asset, so they are verified like the weights. Pinned to the build
the app is validated on — a fresh download reproduces a working `tools/llama/` file for file
(56 of 56 identical). A runtime that is already present, of whatever build, is left alone.
Anything already in the HuggingFace cache (`~/.cache/huggingface`) is copied rather than
downloaded again, and still checksummed.

## The one thing you must host

`mobileclip_image.onnx` (144 MB) and `mobileclip_text.onnx` (254 MB) are built locally by
`scripts/export_mobileclip_onnx.py` + `scripts/patch_clip_text_argmax.py` (the ArgMax int32
patch that onnxruntime-android requires). They exist nowhere public, so they need a home.

Without them the Android app downloads its GGUFs, then fails on the towers and stays on
**Download models** — it has no way to embed frames or queries, so search does not work at
all. This is the only step that blocks shipping the Android app.

They currently live at
**https://huggingface.co/manoharmabbu/videorag-mobileclip** — public, and what the shipped
manifest points at. The rest of this section is how to move them somewhere else.

A free HuggingFace model repo is the easiest host — public CDN, honours HTTP `Range` (so
interrupted downloads resume), no bandwidth cost:

```bash
huggingface-cli login
huggingface-cli repo create videorag-mobileclip --type model
huggingface-cli upload <you>/videorag-mobileclip \
    models/mobileclip_onnx/mobileclip_image.onnx mobileclip_image.onnx
huggingface-cli upload <you>/videorag-mobileclip \
    models/mobileclip_onnx/mobileclip_text.onnx  mobileclip_text.onnx
```

Any static host works instead (nginx/Caddy, S3, R2, a GitHub Release) as long as it serves
the two files over HTTPS and honours `Range`.

Then regenerate the manifest so the apps point at your copies, and rebuild the APK:

```bash
python scripts/gen_model_manifest.py \
    --onnx-base https://huggingface.co/<you>/videorag-mobileclip/resolve/main
cd android && ./gradlew assembleDebug
```

`gen_model_manifest.py` re-reads size and SHA-256 for the GGUFs from the HuggingFace API
without downloading them, and hashes the two local ONNX files. Run it again whenever you
replace a weight.

> Check before you ship: no URL in `android/app/src/main/assets/model_manifest.json` may
> point at `127.0.0.1` or `localhost`. Those are left behind by local testing (the app
> reaches a development host through `adb reverse`), and on someone else's phone they
> resolve to that phone.

## What the user sees

- **Android** — a fresh install shows **Download models** where the model badge normally
  reads "Model ready". Tapping it downloads the set with a progress bar, resumes if
  interrupted, and flips to **Model ready** when every file is present and verified. If the
  device is short on space it says so up front instead of failing part-way through.
- **Windows** — the header of the web UI has a permanent model button, the counterpart of
  the Android badge: **✓ Models ready** when everything the active profile needs is on disk,
  otherwise **⬇ Download models · <size>**. While downloading it becomes **■ Stop · <n>%**
  — progress is kept, and the next press resumes. A strip under the header says what is
  missing and shows progress through download, checksum and unpacking. When it finishes the
  VLM server is started for you; nothing needs restarting. Switching profile re-checks what
  is required. Or from the command line:
  ```bash
  python scripts/download_models.py                 # active profile
  python scripts/download_models.py --profile mobile
  ```

## Uninstall — models are removed automatically

- **Android** — weights live in the app's own external files directory,
  `/sdcard/Android/data/com.cctv.videorag/files/models`. Android deletes that directory
  when the app is uninstalled. The extracted keyframes, the search index and the local copy
  of the imported video are in internal storage (`filesDir`), which is deleted too. Nothing
  is written outside app-private storage, so an uninstall leaves nothing behind — verified
  on a clean emulator by installing, downloading the full set, uninstalling and confirming
  both directories were gone.
- **Windows** — weights live in `models/` and the runtime in `tools/llama/`, both inside the
  project directory. Deleting the project folder (or the folder a packaged build unpacks
  into) removes them. The one exception is a copy that `open_clip` may have put in
  `~/.cache/huggingface` on an earlier run, before the embedder's checkpoint was part of the
  download — the app no longer needs it once `models/mobileclip_s2/` is populated.

## Integrity

Every manifest entry carries a SHA-256, and both clients verify each file before moving it
into place, so a truncated or corrupted transfer fails at download time with a clear message
rather than as an unreadable-model crash later. Partial downloads are kept as `.part` files
and resumed via `Range`; a file that fails its checksum is deleted so a retry starts clean.

A note on which GGUF build to use: `bartowski/Qwen2-VL-2B-Instruct-GGUF` is pinned
deliberately. `ggml-org`'s Q4_K_M has the same architecture and a valid checksum, but makes
this app's vendored llama.cpp emit EOS after two tokens — answers collapse to a bare
"White truck at 00:02:42." Swap a weight only after checking the answers it produces.
