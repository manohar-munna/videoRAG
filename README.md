<div align="center">

# VideoRAG — CCTV Intelligence Platform
### Lazy Multi-Frame VLM & Temporal Grounding Architecture for Surveillance Video

**Instant keyframe ingestion, 512-D multimodal vector search, and on-demand multi-frame forensic video reasoning powered by Apple MobileCLIP-S2 & Local Qwen3-VL (CUDA Accelerated).**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Edge Filter](https://img.shields.io/badge/Edge%20Filter-dHash%20%7C%20pHash-38bdf8)]()
[![RAG Architecture](https://img.shields.io/badge/FAISS-384--D%20Inner%20Product-0284c7)]()
[![LLM Backend](https://img.shields.io/badge/VLM-Qwen3--VL%204B%20(Local%20GPU)-green)]()
[![UI](https://img.shields.io/badge/UI-Surveillance%20Control%20Center-darkred)]()

</div>

---

## 📑 Table of Contents
- [📌 Architecture: Eager VLM vs. Lazy Multi-Frame VLM](#-architecture-eager-vlm-vs-lazy-multi-frame-vlm)
- [🚀 Key Innovations & Capabilities](#-key-innovations--capabilities)
- [🖥️ UI & Control Center Walkthrough](#️-ui--control-center-walkthrough)
- [🛠️ 60-Second Quickstart](#️-60-second-quickstart)
- [🔌 Complete REST API Reference](#-complete-rest-api-reference)
- [⚙️ Configuration Reference (`config.yaml`)](#️-configuration-reference-configyaml)
- [💻 Standalone CLI Tools](#-standalone-cli-tools)
- [🧪 Benchmark Test Matrix](#-benchmark-test-matrix)
- [❓ Troubleshooting & FAQ](#-troubleshooting--faq)
- [📁 Project Layout](#-project-layout)
- [📄 License](#-license)

---

## 📂 Complete Project Directory Structure

If starting from scratch or after cleaning the repository, this is the exact layout the application expects:

```text
VideoRAG-main/
│
├── config/
│   └── config.yaml                     # Primary configuration (LLM backend, retrieval top-k, FAISS paths)
│
├── models/
│   └── qwen3_vl/                       # Local Qwen3-VL 4B GGUF Model & Vision Projector
│       ├── Qwen3VL-4B-Instruct-Q4_K_M.gguf          # ~2.38 GB (4-bit quantized VLM weights)
│       └── mmproj-Qwen3VL-4B-Instruct-F16.gguf      # ~797 MB (Multimodal vision projector)
│
├── tools/
│   └── llama/                          # llama.cpp server binaries & CUDA 12 dependencies
│       ├── llama-server.exe            # Local OpenAI-compatible high-speed inference server
│       ├── cublas64_12.dll             # NVIDIA cuBLAS runtime
│       ├── cudart64_12.dll             # NVIDIA CUDA runtime
│       ├── ggml-cuda.dll               # GGML CUDA backend
│       └── (additional ggml/llama DLLs)
│
├── data/
│   ├── real_cctv_events.json           # Master CCTV surveillance events database
│   ├── cameras_registry.json           # Registered cameras (RTSP, YouTube Live, MP4 feeds)
│   ├── videos/                         # Stored MP4 recordings (e.g. sample_cctv.mp4)
│   └── cameras/                        # Per-camera extracted keyframes and isolated event logs
│       ├── CAM_01/
│       │   ├── events.json
│       │   └── extracted_frames/       # Sampled keyframes (e.g. CAM_01_00_01_15_451.jpg)
│       └── CAM_3000/
│           ├── events.json
│           └── extracted_frames/
│
├── index/
│   ├── cctv_index.faiss                # Dense vector index (384-D FAISS Flat)
│   └── cctv_index.meta.json            # Chunk metadata mapping vector IDs to timestamps/cameras
│
├── src/
│   └── videorag/
│       ├── captioning/                 # VLM frame captioning (VLMCaptioner)
│       ├── evaluation/                 # Groundedness & relevance evaluation metrics
│       ├── indexing/                   # TextEmbedder, Chunker, FAISSVectorStore
│       ├── ingestion/                  # RTSP stream capture, dHash/pHash EdgeFrameFilter
│       ├── llm/                        # Prompt builder & LLMClient (local llama-server / Gemini)
│       ├── retrieval/                  # CCTVRetriever & CrossEncoderReranker
│       └── server.py                   # FastAPI application & background auto-indexer worker
│
├── ui/
│   ├── index.html                      # CCTV Surveillance Command Center UI
│   ├── app.js                          # Real-time DVR seeking, search, and live stream telemetry
│   └── style.css                       # Control room interface styling
│
├── scripts/
│   ├── download_models.py              # Automated downloader for Qwen3-VL GGUF weights
│   ├── index.py                        # Indexing script to compile FAISS database
│   ├── reindex_real.py                 # Quick helper to re-embed data/real_cctv_events.json
│   └── query.py                        # CLI test query utility
│
├── .env.example                        # Template for API keys (optional fallback)
├── .gitignore                          # Git ignore rules (*.gguf, temporary files)
└── requirements.txt                    # Python dependencies
```

---

## ⚡ Setup & Installation From Scratch

Follow these steps to restore and run the complete system even after deleting model/data caches:

### 1. Environment Setup

Clone the repository and install required Python packages:

### 2. 👁️ Apple MobileCLIP-S2 Multimodal Embedder (512-D)
- Fast, state-of-the-art zero-shot multimodal vision-language model.
- Unit-normalized ($\|v\| = 1.0000$) vector representations optimized for FAISS Inner Product cosine ranking.

### 3. 🧠 Local Multi-Frame Qwen3-VL Forensic Reasoning (GPU)
- Runs completely on-premise / offline using `llama-server.exe` with CUDA GPU acceleration.
- Takes ordered base64 image sequences and applies structured **Chain-of-Thought (CoT)** reasoning:
  1. *Frame-by-Frame Evaluation*: Details presence or absence of target query with exact timestamps.
  2. *Forensic Summary Synthesis*: Generates non-contradictory security intelligence reports.

### 4. 🕒 Timezone-Invariant Synchronized DVR & Video Player
- Calculates client-to-server clock drift (`serverTimeOffsetMs`) via `/api/health` roundtrip telemetry.
- Jump seeks effortlessly across local MP4 recordings, RTSP surveillance feeds, and high-resolution archived evidence snapshots.

---

## 🖥️ UI & Control Center Walkthrough

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ [VideoRAG DEFENCE]   Model: Qwen3-VL   Index: 179 Vectors   Status: ONLINE (GPU)  [Dev Mode]│
├────────────────────────────────────────────────────────────────────────────────────────┤
│  🔍 [ Search Video: "white pickup truck parked near yellow tape"                     ] │
│  Camera: [All Feeds] [CAM_01]  Quick: [Vehicles & Trucks] [Caution Tape] [Pedestrians] │
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│  📹 SURVEILLANCE FEED MONITOR            │  🤖 LOCAL QWEN3-VL FORENSIC ANALYSIS        │
│  ┌────────────────────────────────────┐  │  🎬 Forensic Storyboard (5 Frames)          │
│  │                                    │  │  [00:02:33] [00:02:39] [00:02:42]* [00:02:48]│
│  │   1080p CCTV Live Player / DVR     │  │  ───────────────────────────────────────────│
│  │   • Frame-Accurate Jump Seeking    │  │  ▪ 00:02:33 (Frame 1): Crowd gathered near  │
│  │   • HUD: CAM_01 @ 00:02:42         │  │    white truck and yellow caution tape.     │
│  └────────────────────────────────────┘  │  ▪ 00:02:42 (Frame 3): Uniformed security.  │
│  Feed Switcher: [ALL FEEDS] [CAM_01]     │  ───────────────────────────────────────────│
│                                          │  Precision@5: 1.00 | MRR: 1.00 | NDCG: 1.00 │
├──────────────────────────────────────────┴─────────────────────────────────────────────┤
│  ⚡ DEVELOPER HUB (Live 4-Stage Operating Stepper & 512-D Vector Inspector)            │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐         │
│  │ STAGE 1: Done │──>│ STAGE 2: Done │──>│ STAGE 3: Done │──>│ STAGE 4: Done │         │
│  │ Query Embed   │   │ FAISS Search  │   │ Time Slicer   │   │ Qwen3-VL      │         │
│  │ ~33 ms        │   │ Cosine: 0.098 │   │ ±15s–30s      │   │ GPU Vision    │         │
│  └───────────────┘   └───────────────┘   └───────────────┘   └───────────────┘         │
│  Tabs: [⚡ Lazy VLM Pipeline] [Vector & Execution Trace] [Visual Keyframes Dataset (179)]│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ 60-Second Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/manohar-munna/videoRAG.git
cd videoRAG
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download LLM & Vision Projector Models

Run the included automated model downloader:

```bash
python scripts/download_models.py
```

This places the following files into `models/qwen3_vl/`:
1. `Qwen3VL-4B-Instruct-Q4_K_M.gguf` (~2.38 GB)
2. `mmproj-Qwen3VL-4B-Instruct-F16.gguf` (~797 MB)

*(Alternatively, you can manually place compatible Qwen2.5-VL / Qwen3-VL 4B GGUF weights into `models/qwen3_vl/`)*.

### 3. Ensure Llama Server Binaries are Present

Place `llama-server.exe` and its supporting DLLs in `tools/llama/`.
- VideoRAG uses `llama-server.exe` with CUDA 12 GPU acceleration.
- When the application starts, `server.py` automatically detects and launches `llama-server.exe` on port `8080` with `-ngl 99` (full GPU layer offloading).

### 4. Build or Rebuild the FAISS Vector Index

Compile the surveillance vector store from `data/real_cctv_events.json`:

```bash
python scripts/reindex_real.py
```

This generates `index/cctv_index.faiss` and `index/cctv_index.meta.json`.

---

## 🚀 Running the Web Server & Control Center

Start the unified VideoRAG backend:

```
videoRAG/
├── models/
│   ├── mobileclip_s2/
│   │   ├── open_clip_model.safetensors        # Apple MobileCLIP-S2 weights
│   │   └── open_clip_config.json              # Model configuration
│   └── qwen3_vl/
│       ├── Qwen3VL-4B-Instruct-Q4_K_M.gguf    # Quantized Qwen3-VL 4B Instruct
│       └── mmproj-Qwen3VL-4B-Instruct-F16.gguf # Multimodal vision projector
├── data/
│   └── videos/
│       └── sample_cctv.mp4                    # Sample 24/7 CCTV surveillance footage
└── tools/llama/
    └── llama-server.exe                       # llama.cpp server binary
```

### 4. Launch the Server
```powershell
$env:PYTHONUTF8=1
python src/videorag/server.py --port 8000
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
  "camera_filter": "CAM_01"
}
```
- **Response Structure**:
```json
{
  "query": "people wearing pink cloths",
  "answer": "--- Frame 1 (00:08:33) ---\nObservation: ...\n--- Frame 2 (00:08:39) ---\nObservation: Two individuals in pink tops visible.",
  "storyboard": [
    {
      "camera": "CAM_01",
      "timestamp": "00:08:33",
      "seconds": 513.0,
      "image_path": "/data/extracted_frames/CAM_01_00_08_33_15390.jpg",
      "is_anchor": false,
      "score": 0.0
    },
    {
      "camera": "CAM_01",
      "timestamp": "00:08:42",
      "seconds": 522.0,
      "image_path": "/data/extracted_frames/CAM_01_00_08_42_15660.jpg",
      "is_anchor": true,
      "score": 0.0993
    }
  ],
  "evaluation": {
    "retrieval": { "precision_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0 },
    "answer": { "context_utilization": 1.0, "has_timestamp": true, "has_camera": true }
  },
  "debug_trace": {
    "query_vector_dim": 512,
    "query_vector_norm": 1.0,
    "timings_ms": {
      "query_embedding_ms": 32.5,
      "faiss_retrieval_ms": 0.4,
      "temporal_expansion_ms": 0.5,
      "vlm_reasoning_ms": 14200.0,
      "total_ms": 14233.4
    }
  }
}
```

### 2. System Health & System Info
- **Endpoint**: `GET /api/health`
- **Response**:
```json
{
  "status": "online",
  "vector_count": 179,
  "vector_dimension": 512,
  "embedder_model": "MobileCLIP-S2",
  "llm_backend": "local",
  "server_time": 1771440000.123
}
```

### 3. Keyframe Vectors List (Lazy VLM Grounding)
- **Endpoint**: `GET /api/lazy_vlm/vectors?limit=179`
- **Response**: List of 512-D unit vector samples, thumbnail image paths, and timestamp metadata.

### 4. Visual Keyframes Dataset Metadata
- **Endpoint**: `GET /api/events?detailed=true`
- **Response**: Returns total count (`179 Keyframes`), camera breakdown, and complete metadata payload.

### 5. Camera Feeds & RTSP Registry
- **Endpoint**: `GET /api/cameras/feeds` / `GET /api/cameras`
- **Response**: Active camera list and stream configurations.

---

## ⚙️ Configuration Reference (`config.yaml`)

Edit [`config/config.yaml`](file:///C:/Users/manoh/Downloads/git%20repos/VideoRAG-main/config/config.yaml) to customize pipeline parameters:

```yaml
# Data & Video Paths
data:
  mock_path: "data/mock_cctv.json"
  sample_video: "Video Footage/sample_cctv.mp4"

# Edge Frame Extraction & dHash Filtering
edge_filter:
  enabled: true
  hash_method: "dhash"          # Options: "dhash", "phash", "average_hash"
  hamming_threshold: 12         # Min bitwise difference to keep frame
  min_motion_percent: 1.5

# Multimodal Indexing & Embedding
indexing:
  model_name: "MobileCLIP-S2"   # Apple MobileCLIP-S2 (512-D)
  checkpoint_path: "models/mobileclip_s2/open_clip_model.safetensors"
  dimension: 512
  index_type: "flat_ip"         # FAISS IndexFlatIP for normalized cosine similarity
  index_save_path: "index/cctv_index.faiss"
  meta_save_path: "index/cctv_index.meta.json"

# Retrieval & Temporal Episode Bundling
retrieval:
  top_k: 10
  context_window: 2             # ±2 neighbouring frames (5 frames total per episode)
  use_reranker: false

# Vision-Language Model (Forensic Engine)
llm:
  backend: "local"              # Options: "local" (llama-server), "openai", "gemini"
  model: "models/qwen3_vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
  mmproj: "models/qwen3_vl/mmproj-Qwen3VL-4B-Instruct-F16.gguf"
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
python scripts/query.py --query "white pickup truck parked near yellow tape"
```

---

## 🧪 Benchmark Test Matrix

| Benchmark Scenario | Natural Language Query | Ground-Truth Video Timeline | Expected Forensic Output |
| :--- | :--- | :--- | :--- |
| **🚗 Vehicle Inspection** | `white pickup truck parked near yellow tape` | `00:02:33` | Identifies parked white pickup truck and crowd behind tape. |
| **👮 Security Personnel** | `police officer or uniformed security personnel` | `00:02:42` | Cites uniformed officer / security member at anchor timestamp. |
| **👥 Crowd Gathering** | `crowd gathered in front of venice hotel` | `00:02:33 → 00:02:48` | Details crowd dynamics near Venice V Hotel signage. |
| **📹 Anomaly Detection** | `camera abruptly pans or shifts to the ground` | `00:02:42` | Captures blurred ground-level camera anomaly (Frame 3). |
| **👗 Specific Subjects** | `people wearing pink cloths` | `00:08:39 → 00:08:42` | Accurately identifies 2 individuals wearing pink tops. |

---

## ❓ Troubleshooting & FAQ

### Q1: `llama-server.exe` fails to start or says CUDA out of memory
- **Solution**: In `src/videorag/llm/prompter.py`, adjust the `-ngl` (number of GPU layers) parameter. Default is `-ngl 99` (all layers in VRAM). Set `-ngl 24` or `-ngl 16` if your GPU has 4GB–6GB VRAM.

### Q2: Port 8000 or 8080 is already occupied
- **Solution**: Terminate the previous process or specify custom ports:
```powershell
python src/videorag/server.py --port 8005
```

### Q3: Why does a query say "No visual evidence" for objects not present?
- **Answer**: This is intentional! Unlike legacy generative models that hallucinate content, Qwen3-VL performs strict pixel verification and accurately refutes absent entities.

### Q4: How do I add live RTSP CCTV camera feeds?
- **Answer**: Open the **RTSP Live Stream Manager** tab in the Developer Hub, enter your RTSP URL (e.g. `rtsp://admin:pass@192.168.1.100:554/stream1`), assign a Camera ID (`CAM_02`), and click **Register Stream**.

---

## 📁 Project Layout

```
VideoRAG/
├── config/
│   └── config.yaml                     # Pipeline & model configuration parameters
├── data/
│   ├── cameras/                        # Per-camera frame storage & events
│   ├── cameras_registry.json           # Live & recorded camera registry
│   └── extracted_frames/               # Extracted visual keyframes
├── index/
│   ├── cctv_index.faiss                # 512-D MobileCLIP FAISS vector index
│   └── cctv_index.meta.json            # 179 keyframe metadata records
├── models/
│   └── mobileclip_s2/                  # Apple MobileCLIP-S2 weights (safetensors)
├── src/
│   └── videorag/
│       ├── captioning/
│       │   └── vlm_captioner.py        # Multi-frame forensic VLM client (Qwen3-VL)
│       ├── evaluation/
│       │   └── evaluator.py            # Multimodal retrieval & answer evaluator
│       ├── indexing/
│       │   ├── embedder.py             # MobileCLIP-S2 multimodal embedder (512-D)
│       │   └── vector_store.py         # FAISS vector store (Cosine IP)
│       ├── ingestion/
│       │   ├── hash_filter.py          # dHash / pHash edge frame filter
│       │   ├── stream_capture.py       # Multi-camera stream manager
│       │   └── video_processor.py      # Frame extractor with FPS downsampling
│       ├── retrieval/
│       │   ├── reranker.py             # Cross-encoder reranker
│       │   └── retriever.py            # Multimodal anchor + temporal episode retriever
│       └── server.py                   # FastAPI backend server & REST API
├── ui/
│   ├── app.js                          # Control center frontend logic & telemetry
│   ├── index.html                      # Forensic surveillance web application
│   └── style.css                       # Modern dark/light surveillance CSS design
├── requirements.txt                    # Python package dependencies
└── README.md                           # Comprehensive documentation & architecture guide
```

---

## 🧠 System Architecture & GPU VRAM Allocation

- **VLM Inference Engine**: Qwen3-VL 4B running locally on GPU via `llama-server.exe` at `http://127.0.0.1:8080/v1`.
- **GPU Memory Pre-Allocation (~95% VRAM)**:
  - VideoRAG pre-allocates model weights and unified Key-Value (KV) cache slots directly in VRAM.
  - Keeping weights resident in VRAM prevents SSD read bottlenecks, delivering sub-millisecond query retrieval and fast image frame captioning.
  - When idle, **GPU Compute Utilization is 0%**, drawing minimal power.
- **RAG Pipeline**:
  - **Dense Embedding**: `sentence-transformers/all-MiniLM-L6-v2` (384-D).
  - **Cross-Encoder Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  - **Vector Retrieval**: FAISS Inner-Product similarity search.
- **Edge Compute Filter**: Bitwise XOR Hamming Distance (< 0.2ms) drops 80–90% of redundant CCTV frames before they reach the VLM.

---

## 📄 License

MIT License. Built for high-security environments requiring local edge video intelligence and zero-cloud dependency.

