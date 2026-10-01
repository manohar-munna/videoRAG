# Model configuration

What the desktop app needs, where each file goes, and what has to change when the models
are downloaded fresh.

**Short answer: nothing has to be edited by hand.** Pressing **⬇ Download models** in the
header of the web UI performs every step below automatically. This file records what
those steps are, and the few edits you *would* need if you ever swap one model for another.

---

## What the button installs

| What | Size | Pinned source | Lands at | Read by |
|---|---|---|---|---|
| llama.cpp runtime b10549 | 251 MB | GitHub `ggml-org/llama.cpp`, release `b10549`, `llama-b10549-bin-win-cuda-12.4-x64.zip` | `tools/llama/` | `vlm_process_manager.LLAMA_SERVER_EXE` |
| CUDA 12.4 runtime | 391 MB | same release, `cudart-llama-bin-win-cuda-12.4-x64.zip` | `tools/llama/` | `llama-server.exe` (GPU offload) |
| MobileCLIP-S2 embedder | 398 MB | `apple/MobileCLIP-S2-OpenCLIP` @ `8e8a808` | `models/mobileclip_s2/open_clip_model.safetensors` | `config.yaml` → `indexing.model_path`; `embedder.py` |
| Desktop 4B model | 2.50 GB | `unsloth/Qwen3-VL-4B-Instruct-GGUF` @ `00c00da` | `models/qwen3_vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf` | `RUNTIME_PROFILES['desktop']`; `config.yaml` → `llm.model` |
| Desktop 4B projector | 836 MB | same repo and commit | `models/qwen3_vl/mmproj-Qwen3VL-4B-Instruct-F16.gguf` | `RUNTIME_PROFILES['desktop']` |
| Mobile 2B model | 986 MB | `bartowski/Qwen2-VL-2B-Instruct-GGUF` @ `2160e26` | `models/qwen2_vl_2b/Qwen2-VL-2B-Instruct-Q4_K_M.gguf` | `RUNTIME_PROFILES['mobile']` |
| Mobile 2B projector | 1.33 GB | same repo and commit | `models/qwen2_vl_2b/mmproj-Qwen2-VL-2B-Instruct-f16.gguf` | `RUNTIME_PROFILES['mobile']` |

Only the active profile's model is downloaded: **Desktop 4.37 GB**, **Mobile 3.36 GB**
(both include the 1.04 GB runtime + embedder). Every file has a SHA-256 in
`config/model_manifest.json` and is verified before it is put in place.

---

## What happens when the button is pressed

In order, all automatic:

1. **Path check.** Every path the app loads must be one the download installs. The app
   reads model paths from four places — the manifest, `RUNTIME_PROFILES` and
   `LLAMA_SERVER_EXE` in `vlm_process_manager.py`, and `config.yaml` — and if any of them
   disagree, the button turns into **⚠ Model config mismatch**, names the path, and
   refuses to download. Without this, a mismatch would fetch gigabytes into a folder the
   app never opens and then report "Models ready" over an app that cannot start.
   (`downloader.config_mismatches()`, checked on every status poll.)
2. **Space check.** If the disk is too full, it says how much is needed and how much is
   free before the first byte, instead of failing halfway through a 2.5 GB file.
3. **Download** each missing file. Interrupted downloads resume from where they stopped;
   **■ Stop** keeps the partial file. Anything already in the HuggingFace cache is
   copied instead of downloaded.
4. **Verify** each file's SHA-256. A full-length file with the wrong checksum is deleted;
   a short one is treated as interrupted and kept.
5. **Unpack** the two runtime archives into `tools/llama/` and delete the zips.
6. **Switch the embedder to MobileCLIP** if the app started without it. (With no
   checkpoint and no network at startup, the embedder falls back to `clip-ViT-B-32`, a
   different embedding space in which an existing index returns nonsense. This step used
   to need a manual restart.)
7. **Start the model server** (`llama-server`) for the active profile.

The button then reads **✓ Models ready** and the app can index and answer.

One related step happens at **startup** rather than on the button: if
`models/mobileclip_s2/` is missing, the server fetches the pinned MobileCLIP checkpoint
straight into it. (Previously `open_clip` downloaded its own copy into
`~/.cache/huggingface` on every start, outside the project, so it was stored twice once
the button ran.)

---

## What has to change after a fresh download: nothing

The download paths were chosen to match what the code already reads, so a fresh
download needs no edits:

| Code reading the path | Path | Installed by |
|---|---|---|
| `config.yaml` `indexing.model_path`, `embedder.py` | `models/mobileclip_s2/open_clip_model.safetensors` | MobileCLIP entry |
| `config.yaml` `llm.model`, `RUNTIME_PROFILES['desktop'].model_file`, `prompter.py` | `models/qwen3_vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf` | Desktop 4B model |
| `RUNTIME_PROFILES['desktop'].mmproj_file`, `prompter.py` | `models/qwen3_vl/mmproj-Qwen3VL-4B-Instruct-F16.gguf` | Desktop 4B projector |
| `RUNTIME_PROFILES['mobile']` | `models/qwen2_vl_2b/…` (both files) | Mobile 2B entries |
| `LLAMA_SERVER_EXE` | `tools/llama/llama-server.exe` | runtime archive |

### Is the freshly downloaded Desktop model the same as the old one?

Not byte-for-byte, and that was checked before relying on it:

- **Projector:** the tensor data is **identical**; only 384 bytes of header metadata differ.
- **4B model:** **re-quantised by unsloth** — same architecture, same quantisation type,
  same size within 672 bytes, but different weight bytes.

It was run against the previous build on the same frames, same prompt, same runtime,
deterministically (temperature 0). It produced the same verdicts and the same level of
detail for "white truck", "what is written on the truck", "yellow bus" (present) and
"yellow car" (absent). Both builds misjudge "yellow car" in the same way — they call the
yellow minibus "visible" — which is the model's known colour-binding weakness, not a
difference between builds.

---

## If you change a model: the only manual edits

Swapping in a different model or runtime is the one case that needs editing. All of
these must agree, or the button will show **⚠ Model config mismatch**:

1. **`scripts/gen_model_manifest.py`** — the source: `HF_GGUF` (VLM weights per
   profile), `HF_COMMON_WINDOWS` (MobileCLIP), or `LLAMA_BUILD` (runtime build).
2. **`src/videorag/llm/vlm_process_manager.py`** — `RUNTIME_PROFILES[...]["model_file"]`
   and `["mmproj_file"]`; `LLAMA_SERVER_EXE` only if the runtime layout changes.
3. **`config/config.yaml`** — `llm.model` and `indexing.model_path`.
4. **Regenerate the manifest:**
   ```bash
   python scripts/gen_model_manifest.py --onnx-base https://huggingface.co/manoharmabbu/videorag-mobileclip/resolve/main
   ```
5. **Check consistency** — open the app: the button shows **⚠ Model config mismatch** with
   the exact path if anything disagrees. Or from a terminal:
   ```bash
   python -c "import sys; sys.path.insert(0,'src'); from videorag import downloader as d; print(d.config_mismatches() or 'consistent')"
   ```
6. **Test the new build before trusting it.** A different file of the "same" model is not
   guaranteed to behave the same: ggml-org's Qwen2-VL Q4_K_M had the right architecture
   and a valid checksum, and made this project's llama.cpp stop after two tokens. Compare
   answers on known frames before switching.
7. **Android:** the manifest is bundled into the APK
   (`android/app/src/main/assets/model_manifest.json`), so rebuild the APK to ship it.

---

## Deleting models to save space

Everything under `models/` and `tools/llama/` can be deleted; the button puts it back
exactly as listed above (verified: a fresh install reproduces all 56 runtime files
byte-for-byte). Until you press it, the app cannot answer questions. The header button
shows what is missing and its size.

Keep `models/mobileclip_onnx/` (1.5 MB of tokenizer files used by the Android tokenizer
tests). The two `.onnx` towers that used to live there are hosted on HuggingFace and are
no longer needed locally.

Command-line equivalent of the button:

```bash
python scripts/download_models.py                  # active profile
python scripts/download_models.py --profile mobile
```
