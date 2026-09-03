# Hosting the model weights

The apps do not ship their model weights (the Android set is ~2 GB, the Windows set
2.3–3.3 GB). Instead, on first run each app downloads what it needs from a static file
server you host, verifies each file against a checksum, and stores it locally. Remove
the app and the weights go with it (see **Uninstall** below).

You host a plain directory over HTTP. Any static host works: an nginx/Caddy container,
an S3/R2 bucket with public objects, a GitHub Release, etc. The only requirement for
resumable downloads is that the server honours HTTP `Range` requests (nginx, Caddy, S3
and R2 all do).

## 1. What to upload

Run the generator to hash your local weights and (optionally) stage the exact upload
tree:

```bash
python scripts/gen_model_manifest.py --stage
```

This writes `dist/models/` laid out exactly as it must appear on the server:

```
<server root>/
  manifest.json
  android/
    Qwen2-VL-2B-Instruct-Q4_K_M.gguf          986 MB
    mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf     710 MB
    mobileclip_image.onnx                     144 MB
    mobileclip_text.onnx                      254 MB
  windows/
    qwen3_vl/
      Qwen3VL-4B-Instruct-Q4_K_M.gguf        2497 MB   (desktop profile)
      mmproj-Qwen3VL-4B-Instruct-F16.gguf     836 MB
    qwen2_vl_2b/
      Qwen2-VL-2B-Instruct-Q4_K_M.gguf        986 MB   (mobile profile)
      mmproj-Qwen2-VL-2B-Instruct-f16.gguf   1332 MB
```

Per-install download size: **Android 2.09 GB**, **Windows desktop 4B 3.33 GB**,
**Windows mobile 2B 2.32 GB**. You only need to upload the Windows profile(s) you
actually run — a desktop install downloads only its active profile.

Windows does **not** need a MobileCLIP file: the desktop embedder fetches MobileCLIP-S2
from HuggingFace via `open_clip` on first use.

Upload the whole `dist/models/` tree so that `manifest.json` sits at the server root and
the files sit at the paths the manifest lists.

## 2. Point the apps at your server

Let `BASE` be the URL that serves `manifest.json` (i.e. `manifest.json` is reachable at
`$BASE/manifest.json`).

- **Windows** — set it in `config/config.yaml`:
  ```yaml
  models:
    download_base_url: "https://models.example.com/videorag"
  ```
  or export `VIDEORAG_MODEL_BASE_URL`. No rebuild needed.

- **Android** — set `DEFAULT_BASE_URL` in
  `android/app/src/main/java/com/cctv/videorag/ModelDownloader.kt` and rebuild the APK.
  (For testing without a rebuild you can push a `source.txt` containing the URL into the
  app's models directory:
  `adb push url.txt /sdcard/Android/data/com.cctv.videorag/files/models/source.txt`.)

## 3. What the user sees

- **Android** — a fresh install shows **Download models** in place of the model badge.
  Tapping it downloads the ~2 GB set with a progress bar; it resumes if interrupted and
  flips to **Model ready** when done.
- **Windows** — the web UI shows a banner, **Download local models**, whenever weights
  for the active profile are missing. Or run it from the command line:
  ```bash
  python scripts/download_models.py                 # active profile
  python scripts/download_models.py --profile mobile
  ```

## Uninstall — models are removed automatically

- **Android** — the weights live in the app's own external files directory
  (`/sdcard/Android/data/com.cctv.videorag/files/models`). Android deletes that directory
  when the app is uninstalled, so the models go with it. Nothing extra to do.
- **Windows** — the weights live in `models/` inside the project directory. Deleting the
  project folder (or the folder a packaged build unpacks into) removes them. They are not
  written anywhere else on the system.

## Integrity

Every file in the manifest carries a SHA-256. Both clients verify each download against
it before putting the file in place, so a truncated or corrupted transfer fails at
download time with a clear message rather than as an unreadable-model crash later. If you
requantise or replace a weight, re-run `gen_model_manifest.py` and re-upload
`manifest.json` alongside it.
