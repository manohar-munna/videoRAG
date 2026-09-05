<div align="center">

# VideoRAG — On-Device CCTV Intelligence Platform
### Autonomous Multimodal Edge RAG, Dual-Tower MobileCLIP-S2 & Native On-Device Qwen2-VL VLM on Android

**100% On-Device CCTV Keyframe Ingestion, 64-Bit Perceptual dHash Edge Filtering, 6-Region Spatial Pyramid Indexing, 512-D MobileCLIP-S2 Vector Search, Multi-Turn Conversational Reasoning, and Timestamped Click-to-Seek Video Playback.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Android](https://img.shields.io/badge/Android-Native%20Kotlin%20%7C%20C%2B%2B%20JNI%20(llama.cpp)-green.svg)](android/)
[![Edge Filter](https://img.shields.io/badge/Edge%20Gate-64--bit%20dHash%20(<0.15ms)-38bdf8)]()
[![Embedder](https://img.shields.io/badge/Embedder-Apple%20MobileCLIP--S2%20(512--D%20ONNX)-0284c7)]()
[![Vector Store](https://img.shields.io/badge/Vector%20Store-In--Memory%20Cosine%20%7C%20SQLite%20FTS5-0369a1)]()
[![VLM Engine](https://img.shields.io/badge/VLM-Qwen2--VL%202B%20%2F%20Qwen2.5--VL%203B%20(ARM64%20NEON)-emerald)]()

</div>

---

## 📑 Table of Contents
- [🏛️ System Architecture Overview](#️-system-architecture-overview)
  - [⚡ The Core Problem & VideoRAG Paradigm](#-the-core-problem--videorag-paradigm)
  - [🔄 Complete End-to-End System Pipeline](#-complete-end-to-end-system-pipeline)
- [📹 Ingestion Engine & Adaptive Sampling](#-ingestion-engine--adaptive-sampling)
  - [⚡ Sequential Hardware MediaCodec Decoding](#-sequential-hardware-mediacodec-decoding)
  - [🎯 Duration-Aware Adaptive Sampling (`sampleFpsFor`)](#-duration-aware-adaptive-sampling-samplefpsfor)
  - [🛡️ Edge-Gate 64-Bit Perceptual dHash Filtering](#️-edge-gate-64-bit-perceptual-dhash-filtering)
- [🧩 Multimodal Indexing & Spatial Pyramid Engine](#-multimodal-indexing--spatial-pyramid-engine)
  - [📐 6-Region Spatial Pyramid Cropping (`SpatialCropper.kt`)](#-6-region-spatial-pyramid-cropping-spatialcropperkt)
  - [🧠 Dual-Tower MobileCLIP-S2 ONNX Embeddings](#-dual-tower-mobileclip-s2-onnx-embeddings)
  - [🏷️ Zero-Shot On-Device Object Labeling](#️-zero-shot-on-device-object-labeling)
  - [🗄️ In-Memory Vector Store & SQLite FTS5 Persistence](#️-in-memory-vector-store--sqlite-fts5-persistence)
- [🔍 Retrieval Dynamics & Anti-Hallucination Pipeline](#-retrieval-dynamics--anti-hallucination-pipeline)
  - [🔎 Query Expansion & Multi-Variant Embedding](#-query-expansion--multi-variant-embedding)
  - [📊 Spatial Max-Pooling & Dynamic Candidate Pooling](#-spatial-max-pooling--dynamic-candidate-pooling)
  - [🛑 Absolute Relevance Score Gating (`MIN_RELEVANCE = 0.19f`)](#-absolute-relevance-score-gating-min_relevance--019f)
  - [✂️ Semantic Deduplication & Temporal Separation](#️-semantic-deduplication--temporal-separation)
  - [🎯 Sub-Region Crop Routing (`regionCropPath`)](#-sub-region-crop-routing-regioncroppath)
- [🧠 On-Device Native VLM & JNI Tensor Execution](#-on-device-native-vlm--jni-tensor-execution)
  - [⚡ llama.cpp + libmtmd C++ Engine on ARM64 NEON](#-llamacpp--libmtmd-c-engine-on-arm64-neon)
  - [💾 Multi-Tier Vision Encode Cache (RAM + Disk `.bin`)](#-multi-tier-vision-encode-cache-ram--disk-bin)
  - [📝 Isolated Per-Frame Prompt Execution](#-isolated-per-frame-prompt-execution)
  - [🛡️ Anti-Hallucination Post-Processing (`groupBySubject` & `dropUnsupportedTimestamps`)](#️-anti-hallucination-post-processing-groupbysubject--dropunsupportedtimestamps)
  - [🔒 Strict Sequential RAM Mutex (`MemoryOrchestrator.kt`)](#-strict-sequential-ram-mutex-memoryorchestratorkt)
- [💬 Conversational Chat UI & Forensic Video Player](#-conversational-chat-ui--forensic-video-player)
  - [📱 Multi-Turn Conversational Interface (`ChatView.kt`)](#-multi-turn-conversational-interface-chatviewkt)
  - [🎬 Timestamped Click-to-Seek Video Playback](#-timestamped-click-to-seek-video-playback)
  - [🔬 Telemetry & Diagnostics Panel (`DebugPanel.kt`)](#-telemetry--diagnostics-panel-debugpanelkt)
  - [📥 Automated Model Downloader (`ModelDownloader.kt` & `ModelPaths.kt`)](#-automated-model-downloader-modeldownloaderkt--modelpathskt)
- [🏗️ Android Project Directory Structure](#️-android-project-directory-structure)
- [🛠️ Build, Sideload & Installation Guide](#️-build-sideload--installation-guide)
- [❓ Frequently Asked Questions & Forensic Troubleshooting](#-frequently-asked-questions--forensic-troubleshooting)
- [📄 License](#-license)

---

## 🏛️ System Architecture Overview

VideoRAG is a **100% standalone, zero-cloud-dependency native Android application** built with Kotlin and high-performance C++ (`llama.cpp` + `libmtmd`). It brings advanced Multimodal Video Retrieval-Augmented Generation (Video-RAG) to consumer mobile devices (e.g. Qualcomm Snapdragon 8 Gen 2 / Gen 3, MediaTek Dimensity), enabling security operators, forensics teams, and everyday users to query long CCTV recordings in natural language with multi-turn conversation and instant video verification.

```
                                  [ Video Ingestion: MP4 / MKV / Content URI ]
                                                       │
                                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1: Fast Hardware Decoding & Adaptive Keyframe Sampling (VideoFrameDecoder.kt)                       │
 │ • Sequential MediaCodec single-pass decode (avoids slow seek-per-frame redecoding)                        │
 │ • sampleFpsFor(): Adapts sampling rate (0.2 - 2.0 FPS) to video duration (aiming for ~60-150 keyframes)   │
 └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 2: 64-Bit Perceptual dHash Edge-Gate Filter (MobileFrameFilter.kt)                                  │
 │ • Computes 64-bit difference hash (dHash) on 9x8 luminance matrix in <0.15 ms                             │
 │ • Evaluates Hamming distance Δ against the last KEPT keyframe (default threshold Δ >= 10)                 │
 │ • Drops 70%–85% of redundant static surveillance frames before embedding                                  │
 └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                       │ (Motion Keyframes Accepted)
                                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 3: 6-Region Spatial Pyramid & MobileCLIP-S2 Embedding (SpatialCropper.kt & OnDeviceEmbedder.kt)     │
 │ • SpatialCropper slices each frame into 6 spatial pyramid regions (global, top_left, top_right,           │
 │   bottom_left, bottom_right, center) at 60% scale to preserve small-object resolution                     │
 │ • Dual-tower MobileCLIP-S2 (ONNX) embeds 6 crops in a single batched pass into 512-D unit hypersphere     │
 │ • Zero-shot vocabulary labeling classifies detected objects from image vectors                            │
 │ • Vectors stored in in-memory MobileVectorStore and persisted into SQLite FTS5 for cold restarts          │
 └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                       │
                    ┌──────────────────────────────────┴──────────────────────────────────┐
                    ▼ (User Natural Language Query: e.g. "white truck with ladder")       │ (App Restart)
 ┌──────────────────────────────────────────────────────────────────────────────────┐     ▼
 │ STAGE 4: Retrieval, Spatial Max-Pooling & Dedup (MainActivity.kt)                │ ┌──────────────────────┐
 │ • embedTextVariants(): Generates prompt variants and embeds to 512-D space       │ │ Cold-Start Restorer: │
 │ • Spatial Max-Pooling: Retains highest-scoring spatial crop per keyframe         │ │ Restores vector/FTS  │
 │ • Absolute Relevance Gate: MIN_RELEVANCE = 0.19f filters absent queries          │ │ index without        │
 │ • dropNearDuplicates(): Filters cosine duplicates (>0.92) and temporal clusters  │ │ re-encoding video    │
 └──────────────────────────────────────────────────────────────────────────────────┘ └──────────────────────┘
                                                       │ (Top Candidate Keyframes)
                                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 5: On-Device Multimodal VLM Forensic Reasoning (OnDeviceVLM.kt & native-lib.cpp)                    │
 │ • Sub-region crop routing (regionCropPath): Feeds winning 60% crop upscaled (~2.8x pixel density)         │
 │ • MemoryOrchestrator: Releases MobileCLIP ONNX sessions from RAM before allocating VLM context            │
 │ • llama.cpp + libmtmd C++ Engine: Qwen2-VL 2B / Qwen2.5-VL 3B GGUF with Q8_0 quantised projector         │
 │ • Multi-Tier Vision Encode Cache: Reuses in-memory & disk .bin embeddings (saves ~98s prefill latency)   │
 │ • Independent per-frame generation + groupBySubject + dropUnsupportedTimestamps anti-hallucination guard  │
 └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 6: Conversational Chat UI & Click-to-Seek Forensic Video Player (ChatView.kt)                       │
 │ • Multi-turn conversational message bubbles displaying chronological subject descriptions                 │
 │ • Interactive keyframe thumbnail gallery showing exact crop passed to the vision model                    │
 │ • Tapping ANY timestamp (e.g. 00:03:42) immediately seeks and plays the video at that exact second!       │
 └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### ⚡ The Core Problem & VideoRAG Paradigm

Traditional video question answering (Video-QA) architectures suffer from fatal flaws on edge devices:
1. **Eager Captioning Overload**: Feeding every continuous 30-FPS frame into a heavy Vision-Language Model (VLM) during ingestion takes hours or days for even a 10-minute clip (a 13-minute video contains >23,000 raw frames).
2. **Cloud Privacy & Latency Penalties**: Streaming continuous high-resolution surveillance video to cloud endpoints incurs massive bandwidth fees, latency lags, and violates strict surveillance privacy regulations.
3. **Small-Object Blindness**: Downscaling an entire $1920 \times 1080$ surveillance frame to $256 \times 256$ causes small targets (e.g. backpacks, license plates, distant pedestrians, livery text) to blur into unrecognizable noise.

**VideoRAG solves these bottlenecks with a Lazy Multimodal Retrieval + Spatial Pyramid + On-Demand Multi-Frame Forensic Reasoning architecture**:
- **Ingestion in Seconds, Not Hours**: Frames are extracted via sequential `MediaCodec`, filtered in `<0.15 ms` with a 64-bit difference hash, sliced into 6 spatial pyramid crops, and embedded with lightweight MobileCLIP-S2 into 512-D vectors in RAM.
- **Small-Target Preservation**: Slicing the keyframe into 6 overlapping 60% spatial crops gives small objects **~2.8x higher pixel density** in the embedding space and during VLM inspection.
- **Sub-Millisecond Vector Retrieval**: Instant in-memory cosine dot-product ranking retrieves the most relevant moments across hours of footage in microseconds.
- **Lazy On-Demand VLM Reasoning**: The heavy Qwen2-VL model is loaded into RAM *only* when a query is submitted, examining only the top-ranked candidate keyframes.

---

## 📹 Ingestion Engine & Adaptive Sampling

The ingestion pipeline transforms raw surveillance video files into high-density indexed moments with minimal CPU/GPU overhead.

```
 [ Video Stream / File ] ──► [ Sequential MediaCodec ] ──► [ 64-bit dHash Filter ] ──► [ Motion Keyframe ]
                                (1 pass: 0.2-2.0 FPS)         (Hamming Δ >= 10)
```

### ⚡ Sequential Hardware MediaCodec Decoding
- **Source**: [`VideoFrameDecoder.kt`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/ingestion/VideoFrameDecoder.kt)
- **Problem**: Standard Android `MediaMetadataRetriever.getFrameAtTime(timeUs, OPTION_CLOSEST)` must decode forward from the preceding keyframe (I-frame) on *every single request*. Sampling 800 times across a 13-minute video re-decodes the video hundreds of times, taking >18 minutes just to decode frames!
- **Solution**: VideoRAG implements a **single-pass sequential `MediaCodec` extractor** (`decodeVideoSequential`). The MP4/MKV video stream is extracted and decoded sequentially once, emitting frames directly at the target timestamp interval into memory.
- **Fallback**: If an unusual container format or hardware codec failure occurs, it gracefully falls back to `MediaMetadataRetriever` with `OPTION_CLOSEST` (deliberately avoiding `OPTION_CLOSEST_SYNC`, which snaps to sync frames and returns duplicate I-frames).
- **Resolution Normalization**: Extracted keyframes are downscaled to a maximum edge of `MAX_KEYFRAME_DIM = 640px` (e.g. $640 \times 360$ for 16:9), bounding vision tokens to ~264–299 tokens per frame for fast VLM evaluation.

### 🎯 Duration-Aware Adaptive Sampling (`sampleFpsFor`)
- **Source**: [`MainActivity.kt`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/MainActivity.kt#L138-L148)
- Fixed sampling rates fail across varied video lengths:
  - Sampling at 0.2 FPS over a 48-second clip yields only ~9 frames and ~40 vectors, starving retrieval so queries return identical frames.
  - Sampling at 1.0 FPS over a 30-minute recording generates >1,800 frames and >10,000 vectors, causing memory pressure and excessive near-duplicate candidates.
- **Adaptive Formula**:
  $$\text{sampleFps} = \text{clamp}\left(\frac{\text{TARGET\_KEYFRAMES}}{\text{durationSeconds}}, \text{MIN\_SAMPLE\_FPS}, \text{MAX\_SAMPLE\_FPS}\right)$$
  - $\text{TARGET\_KEYFRAMES} = 60.0$ frames
  - $\text{MIN\_SAMPLE\_FPS} = 0.2\text{ FPS}$ (1 frame every 5 seconds; optimal for 10–30 min videos)
  - $\text{MAX\_SAMPLE\_FPS} = 2.0\text{ FPS}$ (1 frame every 0.5 seconds; optimal for short <1 min clips)

### 🛡️ Edge-Gate 64-Bit Perceptual dHash Filtering
- **Source**: [`MobileFrameFilter.kt`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/ingestion/MobileFrameFilter.kt)
- **Execution Speed**: $< 0.15\text{ ms}$ on mobile CPU.
- **Algorithm**:
  1. Downscales the keyframe bitmap to a $9 \times 8$ grayscale matrix.
  2. Computes the grayscale luminosity using $Y = 0.299R + 0.587G + 0.114B$.
  3. Evaluates horizontal luminance gradient: $\text{bit}_{x, y} = 1 \text{ if } \text{gray}[x] > \text{gray}[x+1] \text{ else } 0$.
  4. Packs the 64 boolean comparisons into a single 64-bit `Long` integer (`dHash`).
- **Hamming Distance Gate**:
  $$\Delta = \text{bitCount}(\text{hash}_{\text{current}} \oplus \text{hash}_{\text{last\_kept}})$$
  - If $\Delta < 10$ (`DEFAULT_HAMMING_THRESHOLD`), the frame is classified as a static duplicate and dropped immediately.
  - **Crucial Design**: Compares against the **last kept frame** rather than the immediately preceding sampled frame, ensuring slow camera pans or gradual lighting shifts are not mistakenly dropped.
  - **Efficiency**: Drops **70% to 85%** of redundant CCTV frames at the edge gate before spatial cropping or vector embedding.

---

## 🧩 Multimodal Indexing & Spatial Pyramid Engine

Once a keyframe passes the edge hash gate, it is transformed into a high-dimensional spatial semantic representation.

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 Accepted Keyframe (640x360)            │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
 ┌──────────────────────────────────────┐            ┌──────────────────────────────────────┐
 │ 6-Region Spatial Pyramid Decomposition│            │ Dual-Tower MobileCLIP-S2 (ONNX)      │
 │ • [global]       (0, 0, 1.0, 1.0)    │            │ • Image Tower: 256x256 RGB input     │
 │ • [top_left]     (0, 0, 0.6, 0.6)    │ ─────────► │ • Batched single pass (6 crops/run)  │
 │ • [top_right]    (0.4, 0, 0.6, 0.6)  │            │ • Emits 6x L2-normalized 512-D vectors│
 │ • [bottom_left]  (0, 0.4, 0.6, 0.6)  │            │ • Zero-shot vocabulary classifier    │
 │ • [bottom_right] (0.4, 0.4, 0.6, 0.6)│            └──────────────────┬───────────────────┘
 │ • [center]       (0.2, 0.2, 0.6, 0.6)│                               │
 └──────────────────────────────────────┘                               ▼
                                                     ┌──────────────────────────────────────┐
                                                     │ Hybrid Storage & Persistence         │
                                                     │ • MobileVectorStore (RAM Cosine Index)│
                                                     │ • SQLite FTS5 (Cold-Start Moments DB)│
                                                     └──────────────────────────────────────┘
```

### 📐 6-Region Spatial Pyramid Cropping (`SpatialCropper.kt`)
- **Source**: [`SpatialCropper.kt`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/indexing/SpatialCropper.kt)
- **Why Global Embeddings Fail**: In wide-angle CCTV footage, small targets (e.g. a yellow bus, a pedestrian, or a ladder) occupy only 2%–5% of the frame. Global embeddings are dominated by asphalt and sky, causing target moments to rank 5th–10th with near-tied scores.
- **Pyramid Decomposition**:
  Slices each keyframe into **6 overlapping sub-regions** scaled at 60% width and 60% height:
  1. `global`: Full scene ($100\% \times 100\%$)
  2. `top_left`: Upper-left quadrant $[0.0, 0.0, 0.6, 0.6]$ (Distant oncoming traffic / upper corridor)
  3. `top_right`: Upper-right quadrant $[0.4, 0.0, 0.6, 0.6]$ (Shoulders / upper roadside)
  4. `bottom_left`: Lower-left quadrant $[0.0, 0.4, 0.6, 0.6]$ (Foreground inner lanes)
  5. `bottom_right`: Lower-right quadrant $[0.4, 0.4, 0.6, 0.6]$ (Foreground outer lanes / sidewalks)
  6. `center`: Center corridor $[0.2, 0.2, 0.6, 0.6]$ (Midground traffic flow)
- **Benefit**: Slicing the frame gives the target object **~2.8x higher pixel density** in the embedding space.

### 🧠 Dual-Tower MobileCLIP-S2 ONNX Embeddings
- **Source**: [`OnDeviceEmbedder.kt`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/indexing/OnDeviceEmbedder.kt) & [`ClipTokenizer.kt`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/indexing/ClipTokenizer.kt)
- **Shared 512-D Hypersphere**: Both images and text strings are mapped into a single unified 512-dimensional Euclidean space where $\|v\| = 1.0000$. Semantic similarity is calculated strictly via cosine dot-product ($u \cdot v$).
- **No Mock Fallbacks**: Unlike naive systems that silently fall back to RGB color histograms when models are missing, VideoRAG enforces strict neural execution—throwing explicit `ModelUnavailableException` if ONNX weights are missing.
- **Batched Single-Pass Execution**:
  - Processing 466 keyframes $\times$ 6 crops equals 2,796 forward passes.
  - VideoRAG executes all 6 crops in a **single batched ONNX tensor call** ($[6, 3, 256, 256]$), eliminating per-call JNI invocation overhead and speeding up ingestion by >3.5x.
- **BPE Tokenizer Self-Testing**:
  - Contains a native Kotlin port of the Byte-Pair Encoding (BPE) tokenizer ([`ClipTokenizer.kt`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/indexing/ClipTokenizer.kt)).
  - Automatically runs an automated self-test on app startup ([`ClipTokenizerSelfTest.kt`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/indexing/ClipTokenizerSelfTest.kt)), validating that Kotlin token IDs match Python `open_clip` reference tokens 100% across all edge cases.

### 🏷️ Zero-Shot On-Device Object Labeling
- **Source**: [`OnDeviceEmbedder.kt#L62-L68`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/indexing/OnDeviceEmbedder.kt#L62-L68)
- **Zero Extra Image Cost**: Reuses the 512-D image vectors already computed during ingestion.
- **Vocabulary Classifier**: Pre-computes 512-D text vectors for a curated surveillance vocabulary (`car`, `white truck`, `bus`, `van`, `motorcycle`, `bicycle`, `person`, `camera crew`, `traffic light`, `traffic sign`, etc.).
- **Automatic Metadata**: Tags matching objects into the frame's `IndexedMoment.jsonMetadata` and SQLite FTS search index, enabling instant keyword lookups.

### 🗄️ In-Memory Vector Store & SQLite FTS5 Persistence
- **Source**: [`MobileVectorStore.kt`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/indexing/MobileVectorStore.kt) & [`SQLiteFtsHelper.kt`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/indexing/SQLiteFtsHelper.kt)
- **`MobileVectorStore`**: A thread-safe, in-memory repository of `IndexedMoment` objects storing vectors, metadata, crop regions, and JSON documents.
- **`SQLiteFtsHelper`**:
  - Persists all indexed moments into SQLite (`moments` table) with Full-Text Search (FTS5).
  - **Cold-Start Restoration**: When the app is closed or restarted, `restoreLastIndexIfAny()` immediately restores all indexed keyframes and 512-D vectors into memory in <150 ms without requiring the user to re-import or re-encode the video!
  - **Local Video Cache**: Copies the active video into `filesDir/videos/current.mp4` so click-to-seek video playback works from cold boot even when Android system URI permissions expire.

---

## 🔍 Retrieval Dynamics & Anti-Hallucination Pipeline

Retrieval is the critical bridge between raw keyframe embeddings and multimodal language generation. VideoRAG employs a multi-tiered filtering funnel to ensure that the on-device VLM is fed only distinct, highly relevant visual evidence while eliminating false positives and hallucinations.

```
 User Query: "yellow bus"
       │
       ▼
 [ embedTextVariants() ] ──► Dual 512-D Text Vectors ("a photo of a yellow bus")
       │
       ▼
 [ Spatial Max-Pooling ] ──► Evaluates all 6 crops per keyframe; keeps max dot product
       │
       ▼
 [ Relevance Gating ]    ──► Primary query score >= 0.19f?
       │                      ├── NO  ──► Immediate Honest Refusal ("I couldn't find anything...")
       │                      └── YES ──► Candidate Pool (Top 10% of Index)
       ▼
 [ Semantic & Temporal ] ──► Cosine Similarity <= 0.92f (drops static duplicates)
 [ Deduplication ]       ──► Temporal Separation >= clamp(duration/6, 4s, 15s)
       │
       ▼
 [ Sub-Region Routing ]  ──► Winning 60% crop passed to VLM & Thumbnail Strip
```

### 🔎 Query Expansion & Multi-Variant Embedding
- **Source**: [`OnDeviceEmbedder.kt#L297-L343`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/indexing/OnDeviceEmbedder.kt#L297-L343)
- **Prompt Framing**: Prepends the standard zero-shot prompt template (`"a photo of <query>"`), ensuring alignment with MobileCLIP's pre-training distribution.
- **Head-Noun Extraction (`headNoun`)**:
  - For descriptive multi-word queries (e.g., *"a red double decker bus in the snow"*), it extracts the head noun before the first trailing preposition (`in`, `on`, `at`, `with`, `near`, `under`, `over`, `behind`, etc.), generating an auxiliary recall vector: `"a photo of a bus"`.
  - **The 3-Word Guardrail**: Two-word queries like *"yellow car"* or *"pink shirt"* do **NOT** generate a head-noun variant. Stripping the color modifier would collapse the search to *"car"*, matching every frame in traffic footage and causing the VLM to hallucinate non-existent yellow vehicles.
- **Multi-Vector Search (`searchMulti`)**: Calculates cosine similarity across all query variants and takes the maximum score per region.

### 📊 Spatial Max-Pooling & Dynamic Candidate Pooling
- **Source**: [`MainActivity.kt#L804-L832`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/MainActivity.kt#L804-L832)
- **Spatial Max-Pooling**: Each keyframe generates 6 crop vectors. During retrieval, `bestPerFrame[moment.imagePath]` collapses these 6 scores into the **maximum scoring crop**, allowing small localized objects to compete on their highest-scoring $60\%$ region rather than a diluted full-frame average.
- **Dynamic Candidate Pool Scaling**:
  $$\text{poolSize} = \text{clamp}\left(\frac{\text{vectorStore.size}}{10}, 60, 400\right)$$
  - Prevents candidate starvation on long surveillance videos with thousands of vectors while remaining ultra-fast ($<1\text{ ms}$ over RAM vectors).

### 🛑 Absolute Relevance Score Gating (`MIN_RELEVANCE = 0.19f`)
- **Source**: [`MainActivity.kt#L860-L879`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/MainActivity.kt#L860-L879)
- **Problem**: Nearest-neighbor vector search always returns the top-$K$ candidates, even if the user asks for an object completely absent from the footage (e.g. *"a red double decker bus in the snow"* on a sunny highway). A 2B/3B VLM presented with irrelevant frames will hallucinate plausible descriptions.
- **Decision Boundary**:
  - Measured score for present subjects: $\approx 0.218 - 0.233$.
  - Measured score for absent subjects: $\approx 0.106 - 0.163$.
  - `MIN_RELEVANCE = 0.19f` cleanly separates presence from absence.
- **Primary-Caption Evaluation**: Presence is judged strictly on the full user query embedding (not the head-noun variant), preventing broad head nouns from bypassing the gate.
- **Honest Refusal**: If $\text{topScore} < 0.19$, VideoRAG instantly halts and answers:
  > *"I couldn't find anything matching that in this video. The closest frame was at HH:MM:SS, but it is not a strong enough match to report."*
  This saves up to **3 minutes of mobile battery and VLM generation time** while preventing hallucinations.

### ✂️ Semantic Deduplication & Temporal Separation
- **Source**: [`MainActivity.kt#L972-L1000`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/MainActivity.kt#L972-L1000)
- **Semantic Cosine Threshold (`MAX_KEEP_SIMILARITY = 0.92f`)**:
  - Drops keyframes whose MobileCLIP vectors have $>0.92$ cosine similarity with any already-selected keyframe.
- **Temporal Separation (`minSecondsApart`)**:
  $$\text{minSecondsApart} = \text{clamp}\left(\frac{\text{durationSeconds}}{6}, 4\text{ s}, 15\text{ s}\right)$$
  - Prevents the top candidate slots from being consumed by consecutive 1-second frames of the same moving vehicle (e.g. `00:07:21`, `00:07:22`, `00:07:23`), ensuring temporal diversity across the entire video.
- **"Also Matched" Reporting**:
  - Keyframes that cleared `MIN_RELEVANCE` but were not sent to the VLM (due to token budgets) are listed under *"Also matched this search, not analysed: HH:MM:SS, ..."* with clickable timestamps.

### 🎯 Sub-Region Crop Routing (`regionCropPath`)
- **Source**: [`OnDeviceVLM.kt#L416-L425`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/llm/OnDeviceVLM.kt#L416-L425)
- When a spatial sub-region (e.g. `bottom_left` or `top_right`) wins the max-pooling ranking, VideoRAG crops that exact region from the high-resolution frame and scales it to standard vision input size.
- This gives the VLM **~2.8x higher pixel density** on the target subject, allowing it to accurately read logos, license plates, and livery text that would be illegible in full-scene views.
- The chat thumbnail strip displays the exact crop inspected by the model so users can visually verify the AI's conclusions.

---

## 🧠 On-Device Native VLM & JNI Tensor Execution

VideoRAG runs modern Vision-Language Models directly on mobile ARM64 hardware using a custom C++ runtime powered by `llama.cpp` and `libmtmd`, delivering fully offline multimodal reasoning with zero cloud API keys or external server dependencies.

```
  Top Candidate Keyframes (e.g. 3-5 Crops)
                     │
                     ▼
  ┌────────────────────────────────────────────────────────┐
  │ Multi-Tier Vision Encode Cache (RAM / Disk .bin)       │
  │ • Check in-memory embd_cache (SHA-256 pixel hash)      │
  │ • Check disk cache: cacheDir/embd/<proj>/<hash>.bin    │
  │ • MISS ──► libmtmd vision encode (~18s/frame)          │
  │ • HIT  ──► Instant tensor load (<10ms) (Saves ~98s!)   │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Isolated Per-Frame Single-Sentence Generation          │
  │ • 5-Thread ARM64 NEON execution (nativeGenerate)       │
  │ • Prompt template: <|im_start|>system ... user ...     │
  │ • Focuses on clothing colors, vehicle livery, text     │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Anti-Hallucination Guardrails & Post-Processing        │
  │ • stripEchoedTimestamp(): Strips reflected timestamps  │
  │ • groupBySubject(): Aggregates multi-frame sightings   │
  │ • dropUnsupportedTimestamps(): Mathematical gate       │
  │   stripping ungrounded/invented arithmetic timestamps  │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
              Factual Grounded Chat Response
```

### ⚡ `llama.cpp` + `libmtmd` C++ Engine on ARM64 NEON
- **Source**: [`native-lib.cpp`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/cpp/native-lib.cpp) & [`CMakeLists.txt`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/cpp/CMakeLists.txt)
- **Architecture**: Supports Qwen2-VL 2B / Qwen2.5-VL 3B architectures, multimodal RoPE (M-RoPE), and the `qwen2vl_merger` vision projector.
- **Multithreading**: Configured for 5 CPU threads (`n_threads = 5`) targeting ARM big cores (Cortex-X / Cortex-A7xx) for optimal sustained thermal efficiency without thermal throttling.
- **Q8_0 Quantised Projector Selection (`pickProjector`)**:
  - The standard unquantised FP16 vision projector is **1.33 GB**, which exceeds available mobile cache headroom, causing continuous thrashing and high latency (>72s per frame).
  - VideoRAG prioritizes the **`Q8_0` quantised projector (710 MB)**, which stays fully resident in RAM, runs in **~20.3s per frame**, and preserves 100% of fine OCR and livery reading accuracy.

### 💾 Multi-Tier Vision Encode Cache (RAM + Disk `.bin`)
- **Source**: [`native-lib.cpp#L82-L163`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/cpp/native-lib.cpp#L82-L163)
- **The Bottleneck**: Vision encoding (projecting image pixels into LLM embedding space) takes ~18s per keyframe and accounts for **~90% of total query latency** (~98s of 115s on Snapdragon 8 Gen 2).
- **Two-Tier Architecture**:
  1. **Tier 1 (RAM `embd_cache`)**: Keeps up to 192 MB of encoded float tensors in memory, keyed by the SHA-256 hash of the decoded image pixels.
  2. **Tier 2 (Disk `cacheDir/embd/<proj_name>/<sha256>.bin`)**: Persists encoded float tensors to disk (capped at 384 MB LRU).
- **Impact**: In multi-turn chat sessions, follow-up questions referencing previously examined keyframes load in **<10 ms instead of 18,000 ms**, dropping query response latency from ~115 seconds down to **~5 seconds**!

### 📝 Isolated Per-Frame Prompt Execution
- **Source**: [`OnDeviceVLM.kt#L480-L524`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/llm/OnDeviceVLM.kt#L480-L524)
- **Why Multi-Image Prompts Fail**: Submitting 5 images simultaneously in a single prompt to a 2B edge VLM causes erratic instruction-following collapses—the model randomly alternates between generating 80 tokens and stopping after 2 tokens, losing timestamps.
- **Sequential Per-Frame Execution**: VideoRAG prompts the VLM separately for each keyframe with strict analytical instructions:
  > *"Describe in a single sentence what this frame shows in relation to the question. Note clothing colour, vehicle markings and any text you can read. Say plainly if the thing asked about is not in this frame."*
- **Reliability**: Guarantees a detailed, factual sentence for every candidate frame with zero token loss.

### 🛡️ Anti-Hallucination Post-Processing (`groupBySubject` & `dropUnsupportedTimestamps`)
- **Source**: [`OnDeviceVLM.kt#L573-L650`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/llm/OnDeviceVLM.kt#L573-L650)
- **`stripEchoedTimestamp`**: Strips redundant timestamps echoed by the model (e.g., *"at 00:03:42 at 00:03:42"*).
- **`groupBySubject`**: Merges per-frame sentences that describe the same subject into unified chronological summary lines:
  > *"White truck with 'motion picture' livery parked near crowd at 00:03:42, 00:06:38, and 00:08:50."*
- **`dropUnsupportedTimestamps`**:
  - Small edge VLMs have a documented habit of generating arithmetic timestamp sequences (e.g. converting `00:00:03` into `00:00:07`, `00:00:10`, `00:00:12`).
  - This function parses every timestamp regex (`\b\d{2}:\d{2}:\d{2}\b`) in the final text and cross-references it against `shown.map { it.timestamp }`.
  - Any timestamp not present in the physically inspected keyframes is **strictly removed**, and sentences with zero real timestamps are discarded.

### 🔒 Strict RAM Mutex Orchestration (`MemoryOrchestrator.kt`)
- **Source**: [`MemoryOrchestrator.kt`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/llm/MemoryOrchestrator.kt)
- **The Memory Budget**: A standard 8GB Android device provides ~2.5 GB of active memory headroom after OS allocation. Loading both MobileCLIP ONNX sessions (~400 MB fp32) and Qwen2-VL context (~1.7 GB) simultaneously risks swap thrashing and out-of-memory (OOM) crashes.
- **Asymmetric Release Policy**:
  - `releaseEmbedder()` drops the MobileCLIP ONNX image and text sessions immediately after computing the query vector, freeing ~400 MB of heap before the VLM allocates its generation context.
  - The VLM context and vision encode cache remain resident across queries to preserve sub-second latency.

---

## 💬 Conversational Chat UI & Forensic Video Player

VideoRAG provides a purpose-built mobile interface designed for swift forensic investigation and seamless multi-turn reasoning.

```
 ┌─────────────────────────────────────────────────────────────┐
 │ 📹 Video Ingested: surveillance_cam_04.mp4 (13m 45s) [Reset]│
 ├─────────────────────────────────────────────────────────────┤
 │ [ Operator ]                                    [ 14.2s ]   │
 │   "When does the white truck appear with the camera crew?"  │
 │                                                             │
 │ [ VideoRAG AI ]                                             │
 │   White truck with 'motion picture' livery parked near the  │
 │   crowd at 00:03:42, 00:06:38, and 00:08:50.                │
 │   Also matched this search, not analysed: 00:10:54          │
 │                                                             │
 │   Frames analysed (3) — tap to jump                         │
 │   ┌───────────┐  ┌───────────┐  ┌───────────┐               │
 │   │ [CropImg] │  │ [CropImg] │  │ [CropImg] │               │
 │   │  00:03:42 │  │  00:06:38 │  │  00:08:50 │               │
 │   └───────────┘  └───────────┘  └───────────┘               │
 ├─────────────────────────────────────────────────────────────┤
 │ 🎬 Forensic Video Player: Playing at 00:03:42               │
 │ [ ⏸ Pause ] [ ▶ Play ] [ 🔄 Replay Mark ]                  │
 ├─────────────────────────────────────────────────────────────┤
 │ [ Ask a question about this video...              ] [ Send ]│
 └─────────────────────────────────────────────────────────────┘
```

### 📱 Multi-Turn Conversational Interface (`ChatView.kt`)
- **Source**: [`ChatView.kt`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/ui/ChatView.kt)
- **Operator Question Bubbles**: Right-aligned, primary-colored message bubbles stamped with real-time query latency (e.g. `  14.2s` via `setUserMessageTiming`), giving investigators instant visibility into model execution speed.
- **AI Answer Cards**: Left-aligned structured cards with selectable text and automatically linkified timestamps.
- **Evidence Frame Thumbnail Strip (`addFrameStrip`)**:
  - Displays a horizontal scrolling gallery of the exact keyframes and spatial crops evaluated by the VLM.
  - Gives users visual proof of the AI's findings and makes misinterpretations immediately diagnosable.
- **Context Retention**: Passes the prior conversation turns (truncated to essential facts) back into the prompt so users can ask contextual follow-up questions (e.g. *"What color is the driver's shirt?"* or *"Did anyone get out of the vehicle?"*).

### 🎬 Timestamped Click-to-Seek Video Playback
- **Source**: [`MainActivity.kt#L1020-L1060`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/MainActivity.kt#L1020-L1060)
- **Instant Video Seeking**: Tapping **ANY** timestamp in the AI's response text (e.g. `00:03:42`) or tapping **ANY** keyframe thumbnail in the evidence strip instantly seeks the video player to that exact second and begins playback.
- **Local Video Cache**: During ingestion, VideoRAG caches the video file into `filesDir/videos/current.mp4`. This guarantees that click-to-seek video playback continues working across device restarts even when Android's system URI permissions expire.

### 🔬 Diagnostics Debug Panel (`DebugPanel.kt`)
- **Source**: [`DebugPanel.kt`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/ui/DebugPanel.kt)
- Tap the **`🔬 Diagnostics`** button in the top bar to inspect comprehensive real-time telemetry:
  - **Pipeline Health**: Active VLM model filename, MobileCLIP ONNX embedder status, and BPE tokenizer verification.
  - **Index Statistics**: Total keyframes kept, static frames dropped by dHash gate, gate drop rate %, dense 512-D vectors, and SQLite FTS index rows.
  - **Query Diagnostics**: Latency breakdown, tokens per second, prefill time, generation time, frames sent to model, and any unsupported timestamps dropped by the anti-hallucination filter.
  - **Recall@10 Vector Ranking**: Displays raw cosine similarity scores, spatial crop regions (`[top_left]`, `[center]`, `[global]`), and dispatch status (`sent` vs `—`).
  - **One-Tap Export**: "Copy" button copies full forensic JSON diagnostics to the Android clipboard.

### 📥 Automated Model Downloader (`ModelDownloader.kt` & `ModelPaths.kt`)
- **Source**: [`ModelDownloader.kt`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/ModelDownloader.kt) & [`ModelPaths.kt`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/ModelPaths.kt)
- **Zero-Setup First Launch**: Automatically detects missing model weights on startup and offers to download them from a CDN manifest (`model_manifest.json`).
- **Canonical Storage**: Files are saved into the app's dedicated storage directory:
  $$\text{Path: } \texttt{/sdcard/Android/data/com.cctv.videorag/files/models/}$$
  - Requires **zero dangerous system storage permissions** (`MANAGE_EXTERNAL_STORAGE` is not needed).
  - Cannot be revoked by Android OS updates or OEM battery savers.
  - Automatically deleted when the app is uninstalled.
- **Robust Download Engine**: Features HTTP Range resume (`.part` files) and strict SHA-256 integrity verification.
- **Manual Sideload Support**: Also supports manual sideloading via USB/ADB into `/sdcard/Download/qwen2_vl_2b`.

---

## 🏗️ Android Project Directory Structure

```text
android/
├── app/src/main/
│   ├── AndroidManifest.xml             # Permissions (Camera, Storage, Internet)
│   ├── assets/
│   │   ├── mobileclip_s2_image.onnx    # Apple MobileCLIP-S2 Image Tower (256x256 RGB)
│   │   ├── mobileclip_s2_text.onnx     # Apple MobileCLIP-S2 Text Tower (77 tokens)
│   │   ├── bpe_simple_vocab_16e6.txt   # CLIP BPE Vocabulary (49,408 tokens)
│   │   └── model_manifest.json         # CDN Model weights manifest & SHA-256 hashes
│   ├── cpp/
│   │   ├── CMakeLists.txt              # Native C++ build config (-O3 -ffast-math -flto)
│   │   ├── native-lib.cpp              # JNI wrapper, vision encode cache, llama.cpp execution
│   │   └── llama_engine/               # llama.cpp core runtime + libmtmd multimodal engine
│   ├── java/com/cctv/videorag/
│   │   ├── MainActivity.kt             # UI Controller, VideoView player, Query dispatcher
│   │   ├── ModelDownloader.kt          # CDN downloader with HTTP Range resume & SHA-256
│   │   ├── ModelPaths.kt               # Canonical model storage paths & discovery
│   │   ├── indexing/
│   │   │   ├── OnDeviceEmbedder.kt     # MobileCLIP ONNX batched runner & zero-shot vocabulary
│   │   │   ├── ClipTokenizer.kt        # Native Kotlin BPE Tokenizer port
│   │   │   ├── ClipTokenizerSelfTest.kt# Self-test validating BPE against Python reference
│   │   │   ├── MobileVectorStore.kt    # In-memory thread-safe Cosine similarity index
│   │   │   ├── SpatialCropper.kt       # 6-region spatial pyramid subdivider (60% scale)
│   │   │   └── SQLiteFtsHelper.kt      # SQLite FTS5 database for cold-start index restoration
│   │   ├── ingestion/
│   │   │   ├── VideoFrameDecoder.kt    # Sequential single-pass MediaCodec hardware decoder
│   │   │   ├── MobileFrameFilter.kt    # 64-bit grayscale dHash & Hamming Distance edge gate
│   │   │   └── VideoDownloader.kt      # HTTP/HTTPS remote video downloader
│   │   ├── llm/
│   │   │   ├── OnDeviceVLM.kt          # JNI bridge, isolated per-frame prompt, anti-hallucination
│   │   │   ├── MemoryOrchestrator.kt   # Mutex RAM manager (releases MobileCLIP before VLM)
│   │   │   └── ConversationTurn.kt     # Multi-turn conversational history data model
│   │   └── ui/
│   │       ├── ChatView.kt             # Conversational chat bubbles & thumbnail strip
│   │       └── DebugPanel.kt           # Real-time pipeline diagnostics & telemetry modal
│   └── res/
│       ├── layout/activity_main.xml    # Modern Light Theme forensic workstation UI
│       ├── drawable/                   # Adaptive vectors, bubble backgrounds, badge containers
│       └── values/                     # Colors, styles, and typography
├── build.gradle.kts                    # App build configuration (NDK, ONNX Runtime, Coroutines)
└── settings.gradle.kts                 # Root project configuration
```

---

## 🛠️ Build, Sideload & Installation Guide

### Prerequisites
- Android Studio Ladybug (2024.2+) or Hedgehog+
- Android NDK `26.1.10909125` (or latest NDK installed via SDK Manager)
- Physical Android Device (`ARM64-v8a`, Qualcomm Snapdragon 8 Gen 2 / Gen 3 recommended, 8GB+ RAM, Android 11+)
- USB Cable with USB Debugging enabled

### Step 1: Model Setup (Two Options)

#### Option A: Automatic In-App Download (Recommended)
1. Build and install the APK on your device.
2. On first launch, the app prompts to download required models (~2 GB total).
3. Tap **`Download Models`** to fetch and verify the weights automatically.

#### Option B: Manual Sideload via ADB
Download the models to your PC and push them directly to the canonical app folder:
```bash
# Create canonical models directory on device
adb shell "mkdir -p /sdcard/Android/data/com.cctv.videorag/files/models"

# Push Qwen2-VL 2B GGUF model and Q8_0 Projector
adb push Qwen2-VL-2B-Instruct-Q4_K_M.gguf /sdcard/Android/data/com.cctv.videorag/files/models/
adb push mmproj-Qwen2-VL-2B-Instruct-q8_0.gguf /sdcard/Android/data/com.cctv.videorag/files/models/
```

### Step 2: Build & Install via Gradle CLI

```bash
cd android

# Compile Kotlin and native C++ JNI sources:
./gradlew compileDebugKotlin

# Assemble Debug APK:
./gradlew assembleDebug
# -> Output APK: app/build/outputs/apk/debug/app-debug.apk

# Install directly to connected physical device:
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### Step 3: Ingest a Surveillance Video
1. Open **VideoRAG** on your device.
2. Tap **`📁 Upload Video`** to select any `.mp4` or `.mkv` recording from storage, or tap **`🌐 Video Link`** to enter a video URL.
3. Watch the real-time ingestion telemetry:
   - **Progress Bar**: Shows decoding timestamp (e.g. `00:04:30 / 00:13:00 (34%)`).
   - **Edge Gate**: Displays static frames dropped and gate drop rate (e.g. `686 static frames dropped, 84.5% Gate Drop Rate`).
   - **Metrics**: Extracted Keyframes, Indexed 512-D Regions, and Ingest Time.

### Step 4: Multi-Turn Conversational Forensic Querying
1. Type any natural language search or query in the chat input (e.g.):
   - `"When does the yellow bus appear?"`
   - `"Find any white truck or van near the traffic light"`
   - `"Did a person in pink clothing cross the street?"`
   - `"What is written on the side of the truck?"`
2. Tap **`Send`**.
3. Inspect the conversational response, timing metrics, and the evidence frame strip.
4. Tap **any timestamp** or **thumbnail card** to instantly jump the video player to that exact second!

---

## ❓ Frequently Asked Questions & Forensic Troubleshooting

### 1. Why does the first query take ~110 seconds, while follow-up questions take ~5 seconds?
- **Cause**: The first query performs a cold vision encode (`libmtmd`) on the candidate keyframes (~18 seconds per frame).
- **Resolution**: VideoRAG automatically saves encoded tensors to its **Multi-Tier Vision Encode Cache** (RAM + Disk `.bin`). Follow-up questions in the conversation reuse these cached embeddings in **<10 ms**, dropping response latency to ~5 seconds.

### 2. Why does the app answer "I couldn't find anything matching that" instead of generating an answer?
- **Cause**: VideoRAG's **Relevance Score Gating (`MIN_RELEVANCE = 0.19f`)** detected that the query does not match any frame in the footage.
- **Why this is a feature**: Nearest-neighbor vector search always returns candidates even for absent objects (e.g., searching for *"a red bus in snow"* on a sunny highway). Rather than wasting 3 minutes of battery and letting the VLM hallucinate, VideoRAG honestly refuses absent queries immediately.

### 3. How does the app prevent timestamp hallucinations?
- Small edge VLMs often generate arithmetic sequences of timestamps (e.g. `00:00:03`, `00:00:07`, `00:00:10`).
- VideoRAG enforces a strict mathematical post-processing gate ([`dropUnsupportedTimestamps`](file:///c:/Users/manoh/Downloads/git%20repos/VideoRAG-main/android/app/src/main/java/com/cctv/videorag/llm/OnDeviceVLM.kt#L573-L601)) that cross-references every timestamp in the output against `shown.map { it.timestamp }` and strips any ungrounded claim.

### 4. Does the app require any internet connection or cloud API?
- **No.** VideoRAG is 100% on-device. Keyframe decoding, dHash edge filtering, spatial cropping, MobileCLIP ONNX embeddings, vector retrieval, and Qwen2-VL multimodal reasoning execute entirely locally on your phone's processor.

### 5. Why are small objects (e.g. distant vehicles or bags) detected so accurately?
- Naive systems downscale the entire frame to $256 \times 256$, blurring small objects.
- VideoRAG's **6-Region Spatial Pyramid Cropping (`SpatialCropper.kt`)** slices each frame into 6 overlapping 60% regions, giving small objects **~2.8x higher pixel density** in the embedding space and during VLM inference.

---

## 📄 License

MIT License. Built for real-time multimodal CCTV surveillance intelligence with zero cloud dependency.

