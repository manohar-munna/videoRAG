<div align="center">

# VideoRAG — CCTV Intelligence Platform

**Semantic video retrieval, real-time multi-camera ingestion, and timestamp-precise live stream DVR search powered by Local Vision-Language RAG & Edge dHash/pHash Compute Optimization.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Edge Filter](https://img.shields.io/badge/Edge%20Filter-dHash%20%7C%20pHash-38bdf8)]()
[![RAG Architecture](https://img.shields.io/badge/FAISS-384--D%20Inner%20Product-0284c7)]()
[![LLM Backend](https://img.shields.io/badge/VLM-Qwen3--VL%204B%20(Local%20GPU)-green)]()
[![UI](https://img.shields.io/badge/UI-Surveillance%20Control%20Center-darkred)]()

</div>

---

## 📌 Overview

**VideoRAG** is a defence-grade CCTV intelligence platform that brings Retrieval-Augmented Generation (RAG) to real-time video surveillance and multi-camera stream monitoring.

Instead of sending every raw frame to heavy Vision-Language Models (VLMs), VideoRAG converts CCTV frames into 64-bit perceptual fingerprints. By measuring the **Hamming Distance** against previous keyframes in real time (< 0.2ms), static duplicate frames are filtered at the edge, saving **80–90% of VLM compute** while indexing timestamped surveillance events into a high-dimensional vector store (FAISS).

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

```powershell
$env:PYTHONUTF8=1
python -u src/videorag/server.py --port 8000
```

Open your browser to:  
👉 **`http://127.0.0.1:8000/`**

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

MIT License. Built for high-security environments requiring local edge video intelligence.
