<div align="center">

# VideoRAG — CCTV Intelligence Platform
### Multimodal Edge RAG, 2-Stage Cross-Encoder Reranking & Dynamic Dual-Profile VLM (Desktop 4B GPU ⇄ Mobile 2B CPU)

**Instant CCTV keyframe ingestion, 512-D Apple MobileCLIP-S2 vector search, deep Transformer cross-attention reranking, and on-demand multi-frame forensic surveillance reasoning.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Edge Filter](https://img.shields.io/badge/Edge%20Gate-dHash%20%7C%20pHash%20(<0.2ms)-38bdf8)]()
[![Embedder](https://img.shields.io/badge/Embedder-Apple%20MobileCLIP--S2%20(512--D)-0284c7)]()
[![Vector Store](https://img.shields.io/badge/FAISS-IndexFlatIP%20(Cosine)-0369a1)]()
[![Reranker](https://img.shields.io/badge/Reranker-MS--MARCO%20Cross--Encoder-purple)]()
[![VLM Engine](https://img.shields.io/badge/VLM-Qwen3--VL%204B%20%26%20Qwen2--VL%202B-green)]()

</div>

---

## 📑 Table of Contents
- [📌 System Architecture & Pipeline Overview](#-system-architecture--pipeline-overview)
- [⚡ Dynamic Dual-Profile VLM Execution (Desktop 4B vs. Mobile 2B)](#-dynamic-dual-profile-vlm-execution-desktop-4b-vs-mobile-2b)
- [📱 Android On-Device Native System (Kotlin & C++ NPU/GPU Execution)](#-android-on-device-native-system-kotlin--c-npugpu-execution)
- [🔍 Two-Stage Retrieval: FAISS Cosine + Cross-Encoder Reranking](#-two-stage-retrieval-faiss-cosine--cross-encoder-reranking)
- [📹 Edge-Gate Frame Filtering (dHash / pHash)](#-edge-gate-frame-filtering-dhash--phash)
- [🖥️ Surveillance Command Center Web UI](#️-surveillance-command-center-web-ui)
- [🛠️ Quickstart & Installation](#️-quickstart--installation)
- [🔌 REST API Reference](#-rest-api-reference)
- [⚙️ Configuration Reference (config.yaml)](#️-configuration-reference-configyaml)
- [🧪 Benchmark Telemetry & Test Matrix](#-benchmark-telemetry--test-matrix)
- [📂 Project Directory Layout](#-project-directory-layout)
- [📄 License](#-license)

---

## 📌 System Architecture & Pipeline Overview

VideoRAG replaces computationally prohibitive *eager video captioning* with an ultra-efficient **Lazy Multimodal Retrieval + On-Demand Multi-Frame Forensic Reasoning** architecture:

```text
 CCTV Camera Stream (MP4 / RTSP / YouTube Live DVR)
      │
      ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  STAGE 1: Edge Hash Gate (<0.2ms / frame)                   │
 │  • 64-bit dHash / pHash gradient fingerprinting            │
 │  • Drops 50–85% static/duplicate scenes before LLM ingestion│
 └─────────────────────────────────────────────────────────────┘
      │ (Significant scene shifts & motion keyframes)
      ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  STAGE 2: Apple MobileCLIP-S2 Embedder (512-D)              │
 │  • Fast zero-shot vision-text embedding                     │
 │  • Stored in FAISS IndexFlatIP (Unit Cosine Normalized)     │
 └─────────────────────────────────────────────────────────────┘
      │
      ▼ (User Natural Language Query)
 ┌─────────────────────────────────────────────────────────────┐
 │  STAGE 3: 2-Stage Multimodal Retrieval                      │
 │  1. FAISS Cosine Search: Retrieves top candidate episodes   │
 │  2. Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2):        │
 │     Joint query-passage attention promotes true targets     │
 └─────────────────────────────────────────────────────────────┘
      │ (Top-1 Reranked Chronological Episode)
      ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  STAGE 4: Dynamic VLM Forensic Reasoning                    │
 │  • Desktop Profile: Qwen3-VL 4B (CUDA GPU, 5-frame window) │
 │  • Mobile Profile:  Qwen2-VL 2B (CPU Q8_0, 3-frame window) │
 │  • Verifies target pixels & outputs [CONFIRMED_AT: HH:MM:SS]│
 └─────────────────────────────────────────────────────────────┘
```

---

## ⚡ Dynamic Dual-Profile VLM Execution (Desktop 4B vs. Mobile 2B)

VideoRAG features runtime switching between high-throughput **Desktop GPU** and lightweight **Mobile/Edge CPU** execution modes with zero process leakage. Switching cleanly unloads model weights and releases VRAM before initializing the target runtime.

| Feature / Metric | ⚡ Desktop 4B (GPU) Profile | 📱 Mobile 2B (CPU) Profile |
| :--- | :--- | :--- |
| **Model** | `Qwen3-VL-4B-Instruct-Q4_K_M.gguf` | `Qwen2-VL-2B-Instruct-Q4_K_M.gguf` |
| **Vision Projector** | `mmproj-Qwen3VL-4B-Instruct-F16.gguf` | `mmproj-Qwen2-VL-2B-Instruct-f16.gguf` |
| **Compute Device** | NVIDIA CUDA GPU (`-ngl 99`) | CPU-Only (`-ngl 0`, 6 threads) |
| **Context Window** | 4,096 tokens (4 slots) | 2,048 tokens (1 slot) |
| **KV Cache Type** | `FP16` | **`Q8_0` Quantized** |
| **Storyboard Window** | 5 frames (`[-2, -1, 0, +1, +2]`) | **3 frames (`[-1, 0, +1]`)** |
| **Inference Scaling** | 768 px max dimension | **512 px max dimension (In-Memory)** |
| **GPU Dedicated VRAM** | `5.31 GB / 6.00 GB (88.5%)` | **`0 MB (CPU Mode Safe ✅)`** |
| **Process RSS Memory** | ~1.85 GB | **~1.50 – 1.95 GB** |
| **Token Throughput** | **`~42 – 45 tok/s`** | **`~10 – 18 tok/s`** |
| **Switch Latency** | — | **~3.5 seconds** |

---

## 📱 Android On-Device Native System (Kotlin & C++ NPU/GPU Execution)

VideoRAG includes a **100% standalone, zero-dependency native Android application** that performs full surveillance video decoding, 64-bit dHash keyframe filtering, 6-region spatial pyramid embedding, cosine vector indexing, and VLM timeline reasoning entirely on physical mobile device hardware (Snapdragon, Dimensity, Tensor).

### 🛡️ Strict 6GB Mobile RAM Budget Orchestration
On a typical 6GB RAM phone, Android OS + system services consume ~2.8 GB, leaving **~2.5 GB of usable active RAM headroom**. To prevent Low Memory Killer (LMK) termination:
- **Sequential Model Orchestration (`MemoryOrchestrator.kt`)**: The ONNX Runtime feature embedder (NPU) and Qwen2-VL 2B (Vulkan GPU) **never co-exist in active RAM**.
- **Auto-Closing Native Tensors**: `OnnxTensor` and `OrtSession.Result` closures are scoped in Kotlin `.use { ... }` blocks for immediate native heap reclamation.
- **Query-Time Bitmap Recycling**: Temporary placeholder bitmaps generated for expanded query embeddings are explicitly recycled.

### 🏗️ Android Project Architecture (`/android`)
```text
android/
├── app/src/main/
│   ├── AndroidManifest.xml             # Camera & storage permissions
│   ├── cpp/
│   │   ├── CMakeLists.txt              # Native C++ build config (-O3 -ffast-math)
│   │   └── native-lib.cpp              # JNI wrapper for Vulkan/GPU VLM & fast dHash
│   ├── java/com/cctv/videorag/
│   │   ├── MainActivity.kt             # UI Controller, RAG lifecycle & query expander
│   │   ├── ingestion/
│   │   │   ├── VideoFrameDecoder.kt   # Video & CameraX stream decoder
│   │   │   └── MobileFrameFilter.kt   # 64-bit grayscale dHash & Hamming Distance
│   │   ├── indexing/
│   │   │   ├── OnDeviceEmbedder.kt    # ONNX Runtime NNAPI MobileCLIP-S2 embedder
│   │   │   ├── MobileVectorStore.kt   # Thread-safe in-memory flat Cosine scanner
│   │   │   └── SpatialCropper.kt      # 6-region spatial pyramid subdivider
│   │   └── llm/
│   │       ├── OnDeviceVLM.kt         # JNI bridge for on-device Qwen2-VL 2B
│   │       └── MemoryOrchestrator.kt  # Strict sequential Mutex RAM loader
│   └── res/
│       ├── layout/activity_main.xml   # Modern dark surveillance control center layout
│       └── values/                    # Strings, colors, and themes
├── build.gradle.kts                    # Root build script
└── settings.gradle.kts                 # Project settings
```

### 📱 Build, Install & Sideload Model Weights

#### Step 1: Open in Android Studio
1. Open **Android Studio** and select **Open**.
2. Navigate to the `/android` directory inside the repository.
3. Sync Gradle dependencies and verify CMake links `libllama_jni.so`.

#### Step 2: Build the Standalone APK
```bash
cd android
./gradlew assembleRelease
```
The compiled installer will be located at:
`android/app/build/outputs/apk/release/app-release.apk`

#### Step 3: Install via ADB
```bash
adb install -r android/app/build/outputs/apk/release/app-release.apk
```

#### Step 4: Transfer Model Weights to App Sandbox
Stage weights to SD storage, then move them into the application's isolated sandbox (`filesDir`):
```bash
# 1. Stage models to shared storage
adb push models/mobileclip_s2.onnx /sdcard/
adb push models/qwen2_vl_2b /sdcard/

# 2. Move files securely into application sandbox
adb shell "run-as com.cctv.videorag mkdir -p /data/user/0/com.cctv.videorag/files/qwen2_vl_2b"
adb shell "run-as com.cctv.videorag cp /sdcard/mobileclip_s2.onnx /data/user/0/com.cctv.videorag/files/"
adb shell "run-as com.cctv.videorag cp /sdcard/qwen2_vl_2b/Qwen2-VL-2B-Instruct-Q4_K_M.gguf /data/user/0/com.cctv.videorag/files/qwen2_vl_2b/"
adb shell "run-as com.cctv.videorag cp /sdcard/qwen2_vl_2b/mmproj-Qwen2-VL-2B-Instruct-f16.gguf /data/user/0/com.cctv.videorag/files/qwen2_vl_2b/"

# 3. Clean up staging area
adb shell rm -rf /sdcard/mobileclip_s2.onnx /sdcard/qwen2_vl_2b
```

---

## 🔍 Two-Stage Retrieval: FAISS Cosine + Cross-Encoder Reranking

Dense vector dot-products can sometimes rank generic frames close to target frames (e.g. difference of `< 0.002` in cosine score). VideoRAG implements a **2-Stage Retrieval Pipeline**:

1. **Stage 1 (FAISS Vector Search)**:
   - Uses Apple MobileCLIP-S2 to embed queries and image keyframes into a 512-D unit hypersphere ($\|v\| = 1.0000$).
   - Retrieves a candidate pool of `12+` chronological candidate episodes in `< 100 ms`.
2. **Stage 2 (Transformer Cross-Encoder Reranking)**:
   - Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to jointly score the user natural language query against multi-frame forensic episode descriptions.
   - Genuine target moments receive high positive logit scores (`+0.2998`), while irrelevant candidate frames are penalized (`-1.3047`), promoting the exact target sequence to **Rank #1**.

---

## 📹 Edge-Gate Frame Filtering (dHash / pHash)

To eliminate redundant computation on continuous CCTV surveillance streams, VideoRAG uses 64-bit difference hashing (`dHash`) and perceptual hashing (`pHash`) directly at ingestion:

- **Execution Speed**: `< 0.25 ms` per frame on CPU.
- **Compute Reduction**: Drops **50% to 85%** of duplicate frames at the edge gate before reaching the VLM or vector store.
- **Configurable Threshold**: Interactive slider (Optimal: `8–12`) to preserve motion while discarding static frames.

---

## 🖥️ Surveillance Command Center Web UI

The built-in web control room at `http://127.0.0.1:8000/` includes:

1. **Dual Profile Switcher**: Interactive `[ ⚡ Desktop 4B (GPU) ]` ⇄ `[ 📱 Mobile 2B (CPU) ]` toggle buttons in the top navigation bar.
2. **Multi-Camera Monitor**: Synchronized DVR player supporting local MP4 footage, RTSP streams, and YouTube Live feeds with frame-accurate seeking.
3. **Forensic Storyboard**: Chronological keyframe display citing exact timestamps (`[CONFIRMED_AT: HH:MM:SS]`).
4. **Developer Mode Multi-Tab Inspector**:
   - **Edge Gate & Hash Filter**: Live Hamming distance distribution and motion percentage KPIs.
   - **Vector & Execution Trace**: Real-time breakdown of embedding, FAISS retrieval, Cross-Encoder reranking, and VLM generation latencies.
   - **JSON Event Chunks Explorer**: Live viewer and search tool for indexed surveillance chunks.
   - **RTSP Live Stream Manager**: Register and monitor live camera streams.
   - **Query Telemetry & Evaluation Benchmark Log**: Persistent history tracking Process RSS, RAM %, dedicated GPU VRAM, token speed, and full model responses.

---

## 🛠️ Quickstart & Installation

### 1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/manohar-munna/videoRAG.git
cd videoRAG

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Downloads

VideoRAG uses local GGUF models for zero cloud dependency:

#### Desktop 4B Model (Qwen3-VL 4B Instruct):
- Model: `models/qwen3_vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf` (~2.38 GB)
- Vision Projector: `models/qwen3_vl/mmproj-Qwen3VL-4B-Instruct-F16.gguf` (~797 MB)

#### Mobile 2B Model (Qwen2-VL 2B Instruct):
- Model: `models/qwen2_vl_2b/Qwen2-VL-2B-Instruct-Q4_K_M.gguf` (~940 MB)
- Vision Projector: `models/qwen2_vl_2b/mmproj-Qwen2-VL-2B-Instruct-f16.gguf` (~1.27 GB)

*(Download models automatically using `python scripts/download_models.py` or place existing weights into the respective `models/` folders)*.

### 3. Ensure Llama Server Binaries
Place `llama-server.exe` and CUDA DLLs into `tools/llama/`. VideoRAG automatically manages the lifecycle of `llama-server.exe`.

### 4. Start the Application
```powershell
# Launch in default Desktop (GPU) mode:
python src/videorag/server.py --port 8000

# Or launch directly in Mobile (CPU) mode:
python src/videorag/server.py --profile mobile --port 8000
```
Open **`http://127.0.0.1:8000/`** in your browser.

---

## 🔌 Complete REST API Reference

### 1. Execute Semantic Search
- **Endpoint**: `POST /api/search`
- **Request Body**:
```json
{
  "query": "people wearing pink cloths",
  "top_k": 5,
  "camera_f### 2. Runtime Profile Switching
- **Endpoint**: `POST /api/profile/switch`
- **Request Body**:
```json
{
  "profile": "mobile"
}
```
- **Response**:
```json
{
  "status": "switched",
  "active_profile": "mobile",
  "name": "Mobile 2B CPU",
  "llm_backend": "local",
  "model": "models/qwen2_vl_2b/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
  "base_url": "http://127.0.0.1:8080/v1"
}
```

### 3. System Health & System Info
- **Endpoint**: `GET /api/health`
- **Response**:
```json
{
  "status": "online",
  "vector_count": 2238,
  "vector_dimension": 512,
  "embedder_model": "MobileCLIP-S2",
  "active_profile": "desktop",
  "profile_name": "Desktop 4B GPU",
  "reranker": "CrossEncoderReranker",
  "server_time": 1787590454.123
}
```

### 4. Keyframe Vectors List (Lazy VLM Grounding)
- **Endpoint**: `GET /api/lazy_vlm/vectors?limit=100`
- **Response**: List of 512-D unit vector samples, thumbnail image paths, and timestamp metadata.

### 5. Camera Feeds & RTSP Registry
- **Endpoint**: `GET /api/cameras/feeds` / `GET /api/cameras`
- **Response**: Active camera list and stream configurations.

---

## ⚙️ Configuration Reference (`config.yaml`)

Edit [`config/config.yaml`](file:///C:/Users/manoh/Downloads/git%20repos/VideoRAG-main/config/config.yaml) to customize pipeline parameters:

```yaml
# Edge Frame Extraction & dHash Filtering
edge_filter:
  enabled: true
  hash_method: "dhash"          # Options: "dhash", "phash", "average_hash"
  hamming_threshold: 10         # Min bitwise difference to keep frame
  min_motion_percent: 1.5

# Multimodal Indexing & Embedding
indexing:
  model_name: "MobileCLIP-S2"   # Apple MobileCLIP-S2 (512-D)
  dimension: 512
  index_type: "flat_ip"         # FAISS IndexFlatIP for normalized cosine similarity
  index_save_path: "index/cctv_index"

# Retrieval & Temporal Episode Bundling
retrieval:
  top_k: 10
  context_window: 2             # ±2 neighbouring frames (5 frames total for Desktop)
  use_reranker: true            # Enables Cross-Encoder second-stage reranking

# Vision-Language Model (Forensic Engine)
llm:
  backend: "local"              # Options: "local" (llama-server), "openai", "gemini"
  model: "models/qwen3_vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
  base_url: "http://127.0.0.1:8080/v1"
  temperature: 0.1
  max_tokens: 1024
```

---

## 💻 Standalone CLI Tools

### 1. Re-Index Custom Video Footage
To extract keyframes, filter duplicates with dHash, compute MobileCLIP 512-D vectors, and build a new FAISS index from the terminal:
```powershell
python scripts/index.py --data "data/videos/sample_cctv.mp4" --camera CAM_01
```

### 2. Interactive CLI Forensic Query Loop
Run natural-language queries directly in the terminal with colored tables and panels:
```powershell
python scripts/query.py --query "people wearing pink color costumes"
```

---

## 🧪 Benchmark Telemetry & Test Matrix

Tested on NVIDIA GeForce RTX 4050 Laptop GPU (6.00 GB Dedicated VRAM) + Intel Core i7 (15.73 GB RAM):

| Natural Language Query | Camera | Promoted Rank #1 Timestamp | Cross-Encoder Score | FAISS Score | Total Pipeline Latency | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `people wearing pink color costumes` | `CAM_01` | **`00:07:36`** | **`+0.2998`** | `0.2505` | **2,428 ms** | 100% (Verified) |
| `white pickup truck parked near yellow tape` | `CAM_01` | **`00:08:50`** | **`+0.3421`** | `0.2612` | **2,150 ms** | 100% (Verified) |
| `police officer or uniformed security personnel` | `CAM_01` | **`00:10:14`** | **`+0.2845`** | `0.2480` | **2,380 ms** | 100% (Verified) |
| `camera mounted on a black color cart` | `CAM_01` | None (Absent) | **`-2.9319`** | `0.2010` | **1,980 ms** | 100% (Accurate Refusal) |

---

## ❓ Troubleshooting & FAQ

### Q1: `llama-server.exe` fails to start or says CUDA out of memory
- **Solution**: Switch to the **Mobile 2B CPU** profile via the UI navigation header or pass `--profile mobile` on startup.

### Q2: Port 8000 or 8080 is already occupied
- **Solution**: Terminate the previous process or specify custom ports:
```powershell
python src/videorag/server.py --port 8005
```

### Q3: Why does a query say "No visual evidence" for objects not present?
- **Answer**: This is intentional! Unlike legacy generative models that hallucinate content, Qwen3-VL and Qwen2-VL perform strict pixel verification and accurately refutes absent entities.

### Q4: How do I add live RTSP CCTV camera feeds?
- **Answer**: Open the **RTSP Live Stream Manager** tab in the Developer Hub, enter your RTSP URL (e.g. `rtsp://admin:pass@192.168.1.100:554/stream1`), assign a Camera ID (`CAM_02`), and click **Register Stream**.

---

## 📁 Project Layout

```text
VideoRAG/
├── config/
│   └── config.yaml                     # Pipeline & model configuration parameters
├── data/
│   ├── cameras_registry.json           # Live & recorded camera registry
│   ├── real_cctv_events.json           # Master CCTV surveillance events database
│   └── cameras/                        # Per-camera frame storage & events
│       ├── CAM_01/
│       │   ├── events.json
│       │   └── extracted_frames/
│       └── CAM_3000/
│           ├── events.json
│           └── extracted_frames/
├── index/
│   ├── cctv_index.faiss                # 512-D MobileCLIP FAISS vector index
│   └── cctv_index.meta.json            # Keyframe chunk metadata records
├── models/
│   ├── qwen3_vl/                       # Desktop 4B VLM GGUF weights & vision projector
│   └── qwen2_vl_2b/                    # Mobile 2B VLM GGUF weights & vision projector
├── tools/
│   └── llama/                          # llama.cpp server binaries & CUDA 12 runtimes
├── src/
│   └── videorag/
│       ├── captioning/
│       │   └── vlm_captioner.py        # Multi-frame forensic VLM client & in-memory resizing
│       ├── evaluation/
│       │   └── evaluator.py            # Precision@5, MRR, NDCG retrieval evaluator
│       ├── indexing/
│       │   ├── embedder.py             # Apple MobileCLIP-S2 multimodal embedder (512-D)
│       │   └── vector_store.py         # FAISS vector store (Cosine IP)
│       ├── ingestion/
│       │   ├── hash_filter.py          # dHash / pHash edge frame filter (<0.2ms)
│       │   └── stream_capture.py       # Multi-camera RTSP/YouTube Live stream capture
│       ├── llm/
│       │   ├── vlm_process_manager.py  # Lifecycle & clean switching for Desktop/Mobile profiles
│       │   └── prompter.py             # Security-specialized prompt templates
│       ├── monitoring/
│       │   └── hardware_telemetry.py   # NVML GPU VRAM, CPU utilization & RSS memory tracker
│       ├── retrieval/
│       │   ├── reranker.py             # MS-MARCO Cross-Encoder reranker
│       │   └── retriever.py            # Multimodal anchor + temporal episode retriever
│       └── server.py                   # FastAPI backend server & REST API
├── ui/
│   ├── app.js                          # Control center frontend logic, profile switching & telemetry
│   ├── index.html                      # Forensic surveillance web application
│   └── style.css                       # Modern surveillance CSS design
├── requirements.txt                    # Python package dependencies
└── README.md                           # Comprehensive documentation & architecture guide
```

├── requirements.txt                    # Python package dependencies
└── README.md                           # Comprehensive documentation & architecture guide
```

---

## 📄 License

MIT License. Built for high-security environments requiring real-time multimodal CCTV intelligence and zero cloud dependency.


