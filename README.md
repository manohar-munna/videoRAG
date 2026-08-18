<div align="center">

# VideoRAG — CCTV Intelligence Platform
### Lazy Multi-Frame VLM & Temporal Grounding Architecture for Surveillance Video

**Instant keyframe ingestion, 512-D multimodal vector search, and on-demand multi-frame forensic video reasoning powered by Apple MobileCLIP-S2 & Local Qwen3-VL (CUDA Accelerated).**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Edge Filter](https://img.shields.io/badge/Edge%20Gate-dHash%20%7C%20pHash%20(<0.2ms)-38bdf8)]()
[![Multimodal Embedder](https://img.shields.io/badge/Vision%20Embedder-Apple%20MobileCLIP--S2%20(512--D)-6366f1)]()
[![Vector Store](https://img.shields.io/badge/FAISS-512--D%20Cosine%20IP-0284c7)]()
[![Forensic VLM](https://img.shields.io/badge/Forensic%20VLM-Local%20Qwen3--VL%204B%20(GPU)-10b981)]()
[![UI](https://img.shields.io/badge/UI-Surveillance%20Control%20Center-darkred)]()

</div>

---

## 📌 Architecture Overview: The Lazy VLM Approach

Traditional video RAG architectures run heavy Vision-Language Models (VLMs) on **every extracted frame during ingestion** to generate static text descriptions. This creates major bottlenecks:
- **Massive Ingestion Latency**: 5–10 seconds of LLM wait per keyframe.
- **Lost Visual Details**: Low-level forensic details (clothing patterns, exact positions, vehicle models) are lost in text summaries.
- **Temporal Blindness**: Frames are summarized in isolation without chronological context.

**VideoRAG solves this with the Lazy Multi-Frame VLM Architecture:**

```
[ INGESTION PIPELINE — Instant (~30ms per frame, 0.0s LLM Wait) ]
Raw CCTV Stream ──> dHash/pHash Edge Gate ──> MobileCLIP-S2 (512-D) ──> FAISS Vector Store + Metadata
                    (Drops static frames)     (Direct image embedding)

[ ON-DEMAND QUERY & RETRIEVAL PIPELINE ]
User Query ──> MobileCLIP-S2 Text Embedder (512-D)
                           │
                           ▼
               FAISS Multimodal Search (Cosine IP)
                           │
                           ▼ (Anchor Moment)
             Temporal Context Window Expander
             (Pulls ±15s–30s chronological frames)
                           │
                           ▼
          Local Multi-Frame Forensic VLM (Qwen3-VL 4B GPU)
          (Inspects raw pixels across the episode sequence)
                           │
                           ▼
           Factually Grounded Security Analysis
```

---

## 🚀 Key Features

### ⚡ 1. Instant Edge Ingestion & Multimodal Vector Store
- **Edge dHash / pHash Gate**: Computes 64-bit perceptual fingerprints in `< 0.2ms` and calculates Hamming Distance against previous frames to discard static duplicate footage, saving **80–90% of storage and compute**.
- **Apple MobileCLIP-S2 (512-D)**: Keyframes are directly encoded into unit-normalized 512-dimensional float32 vector embeddings on GPU.
- **Zero Ingestion LLM Delay**: No heavy captions are generated at ingestion time. Ingestion is near instantaneous (~30ms per keyframe).

### 🎬 2. On-Demand Multi-Frame Forensic Reasoning (Qwen3-VL)
- **Temporal Context Expander**: Once an anchor frame is retrieved, the pipeline automatically extracts surrounding chronological frames ($\pm 15\text{s} - 30\text{s}$) from the same camera timeline.
- **Pixel-Level Forensic Reasoning**: **Qwen3-VL 4B Instruct** inspects the multi-frame visual progression simultaneously, identifying movements, subject appearances, vehicle entries, and anomalies with exact time-stamped citations.
- **Zero-Hallucination Visual Grounding**: Factually refutes absent queries (e.g. *"no evidence of person in pink clothing"*) while confirming verified positive matches with frame-by-frame evidence.

### 🎛️ 3. Developer Hub & Real-Time Pipeline Debugger
- **4-Stage Horizontal Live Operating Pipeline**: Visual stepper displaying live status (`RUNNING` $\rightarrow$ `COMPLETE`) and exact latency across:
  - `Stage 1`: Text Query Embedding (*MobileCLIP-S2 512-D*)
  - `Stage 2`: Multimodal FAISS Search (*Cosine Inner Product*)
  - `Stage 3`: Temporal Window Expander (*±15s–30s Context Slicer*)
  - `Stage 4`: Qwen3-VL Forensic Reasoning (*GPU Vision-Language Engine*)
- **512-D Vector & Execution Tracing**: Inspect raw float32 vector arrays, verify unit norm ($\|v\| = 1.0000$), inspect full timing breakdowns, and copy raw prompt payloads.
- **Visual Keyframes Dataset Explorer**: Explore all indexed keyframes with visual thumbnail cards or toggle to raw JSON metadata with one-click clipboard copying.
- **Collapsible Keyframe Grounding Explorer**: Expand/collapse the 179 keyframe thumbnail cards with an animated toggle arrow.

### 🕒 4. Timezone-Invariant & Synchronized DVR Seeking
- **Clock Synchronization Engine**: Client-server clock offset tracking (`serverTimeOffsetMs`) calculated via `/api/health` latency telemetry, eliminating drift caused by client browser timezones or container virtualization.
- **Interactive Forensic Storyboard**: Clickable frame-by-frame storyboard thumbnails with exact timestamp jump seeking.
- **Live Stream DVR & Snapshot Modes**: Seamlessly switches between local MP4 video file playback, live RTSP streams, and high-resolution archived evidence snapshots.

---

## 🏗️ System Architecture

```
                               ┌──────────────────────────────────────────────┐
                               │             VideoRAG Control Center          │
                               │  (FastAPI Server + Modern Surveillance UI)   │
                               └──────────────────────┬───────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
         ┌───────────────────────────┐                                 ┌───────────────────────────┐
         │     Ingestion Engine      │                                 │      Retrieval Engine     │
         │  • MultiCameraStreamMgr   │                                 │  • CCTVRetriever          │
         │  • VideoFrameExtractor    │                                 │  • Temporal Context Slicer│
         │  • EdgeFrameFilter(dHash) │                                 │  • Evaluation Engine      │
         └─────────────┬─────────────┘                                 └─────────────┬─────────────┘
                       │                                                             │
                       ▼                                                             ▼
         ┌───────────────────────────┐                                 ┌───────────────────────────┐
         │     Embedding Engine      │                                 │       VLM Reasoning       │
         │  • Apple MobileCLIP-S2    │                                 │  • Local llama-server     │
         │  • 512-D Unit Vectors     │                                 │  • Qwen3-VL 4B Instruct   │
         │  • FAISS Vector Store     │                                 │  • CUDA GPU Acceleration  │
         └───────────────────────────┘                                 └───────────────────────────┘
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python**: 3.10 or higher
- **NVIDIA GPU**: CUDA-capable GPU (recommended for local Qwen3-VL acceleration)
- **Git**

### 2. Clone and Install Dependencies
```bash
git clone https://github.com/manohar-munna/videoRAG.git
cd videoRAG
pip install -r requirements.txt
```

### 3. Model Files Setup

Ensure the model files are placed in their respective project directories:

1. **Apple MobileCLIP-S2 Checkpoint**:
   - `models/mobileclip_s2/open_clip_model.safetensors`
   - `models/mobileclip_s2/open_clip_config.json`
2. **Local Qwen3-VL 4B GGUF Model & Vision Projector**:
   - `Local LLM 3VL 4Q/Qwen3VL-4B-Instruct-Q4_K_M.gguf`
   - `Local LLM 3VL 4Q/mmproj-Qwen3VL-4B-Instruct-F16.gguf`
3. **llama-server Binary**:
   - `tools/llama/llama-server.exe`

### 4. Running the Platform

Launch the VideoRAG FastAPI server (which automatically initializes `llama-server.exe` on GPU port 8080):

```powershell
$env:PYTHONUTF8=1
python src/videorag/server.py --port 8000
```

Open your browser to:  
👉 **`http://127.0.0.1:8000/`**

---

## 🧪 Benchmark Test Scenarios

Try running the built-in benchmark queries in the search bar:

| Scenario | Query | Expected Forensic Output |
| :--- | :--- | :--- |
| **Vehicle Inspection** | `white pickup truck parked near yellow tape` | Identifies parked white pickup truck and crowd behind tape at `00:02:33`. |
| **Security Personnel** | `police officer or uniformed security personnel` | Detects uniformed security / officer at `00:02:42`. |
| **Object Detection** | `people wearing pink cloths` | Highlights two individuals in pink tops at `00:08:39` and `00:08:42`. |
| **Anomaly Detection** | `camera abruptly pans or shifts to the ground` | Discovers blurred ground shot anomaly at `00:02:42` (Frame 3). |

---

## 📁 Repository Structure

```
VideoRAG/
├── config/
│   └── config.yaml                     # Pipeline & model configurations
├── data/
│   ├── cameras/                        # Per-camera extracted frames & events
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
└── README.md                           # Documentation & architecture guide
```

---

## 📄 License

MIT License. Built for high-security environments requiring local edge video intelligence and zero-cloud dependency.
