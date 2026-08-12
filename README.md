<div align="center">

# VideoRAG — CCTV Intelligence Platform

**Semantic video retrieval, indexing, and timestamp-precise search for defence-grade surveillance systems.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![UI](https://img.shields.io/badge/UI-Minimal%20Glassmorphic-0284c7)]()
[![Status](https://img.shields.io/badge/status-active%20development-brightgreen)]()
[![Defence](https://img.shields.io/badge/domain-defence%20%26%20security-red)]()

</div>

---

## Overview

**VideoRAG** is a defence-grade CCTV intelligence platform that brings Retrieval-Augmented Generation (RAG) to video surveillance. It features a **minimalist glassmorphic Web UI** that enables security operators and analysts to semantically query CCTV video footage in natural language and **instantly jump to the exact video timestamp** with a single click — no manual scrubbing required.

Instead of reviewing footage frame-by-frame, simply ask:

> *"Show me the moment a person in a red jacket entered the north gate."*
> *"Find all instances of unattended baggage near Gate 3 after 22:00."*
> *"What vehicles or trucks are visible in the CCTV video and at what timestamp?"*

VideoRAG indexes, understands, and retrieves — giving analysts time back for what matters.

---

## 🎨 Minimal Glassmorphic Web UI

The Web UI is built with a deep slate canvas, frosted glass containers (`backdrop-filter: blur(16px)`), and **electric blue theme selection highlights (`#38bdf8`)**:

- 📺 **Interactive Video Player**: Embedded CCTV player (`sample_cctv.mp4`).
- ⏱️ **Click-to-Seek Timestamps**: Clicking any timestamp (`00:09:30`) in the AI analysis or evidence list immediately seeks the video player to that exact second and plays!
- 🔍 **Natural Language Search Box**: Real-time semantic query input with glowing blue focus state.
- 🏷️ **Camera Filter Pills**: Filter search results by specific camera feeds (`CAM_01`, `CAM_02`, `CAM_03`...) with electric blue selection pills.
- ⚡ **Local Qwen3-VL AI Security Analysis Card**: Live CUDA-accelerated LLM reasoning output with highlighted camera tags and timestamp badges.
- 📊 **Live Evaluation Metrics**: Displays Precision@5, MRR, NDCG@5, and context utilization.

---

## Key Features

| Feature | Description |
|---|---|
| 🖥️ **Minimal Web UI** | Glassmorphic Web Interface at `http://127.0.0.1:8000/` |
| 🔍 **Semantic Search** | Query real video footage in natural language |
| 🕐 **Click-to-Seek Timestamps** | Direct video seeking on timestamp click |
| 📹 **Local Video Ingestion** | Extract, sample, & caption MP4/CCTV video feeds |
| 🧠 **Local VLM Vision Engine** | **Qwen3-VL 4B + mmproj** on local GPU for visual frame understanding |
| 🛡️ **Defence-Grade & Air-Gapped** | 100% local processing — no cloud APIs, zero video leakage |
| 🔄 **Vector Search + Reranking** | FAISS dense retrieval + CrossEncoder reranking |

---

## Architecture

```
CCTV MP4 Video Footage
        |
   [Frame Extractor]    OpenCV — sample every N seconds
        |                  -> data/extracted_frames/
        |
   [Local VLM Vision]   Qwen3-VL 4B (GGUF) + mmproj (CUDA GPU)
        |                  -> data/real_cctv_events.json
        |
   [Chunker]            Temporal sequence sliding window
        |
   [Embedder]           sentence-transformers (all-MiniLM-L6-v2)
        |
   [FAISS Index]        IndexFlatIP — cosine similarity vector search
        |
        |  <--- Natural language query via Web UI (http://127.0.0.1:8000/)
        |
   [FastAPI Backend]    src/videorag/server.py
        |
   [Retriever + Reranker] Top-K dense retrieval + CrossEncoder reranking
        |
   [Local LLM Answer]   Qwen3-VL 4B (Local GGUF on CUDA GPU)
        |
   Glassmorphic Web UI  Clickable Timestamps · Video Seek · Blue Selection Highlights
```

---

## Project Structure

```
videorag/
├── ui/                         # Minimal Glassmorphic Web Interface
│   ├── index.html              # HTML structure & glass layout
│   ├── style.css               # Glassmorphic CSS & electric blue text highlights
│   └── app.js                  # Interactive REST API search & video seek logic
├── Video Footage/
│   └── sample_cctv.mp4         # Target CCTV video file
├── Local LLM 3VL 4Q/
│   ├── Qwen3VL-4B-Instruct-Q4_K_M.gguf      # Local LLM text weights
│   └── mmproj-Qwen3VL-4B-Instruct-F16.gguf  # Local VLM vision projector
├── tools/
│   └── llama/                  # Embedded CUDA llama-server binary
├── src/videorag/
│   ├── server.py               # FastAPI backend serving REST API & Web UI
│   ├── ingestion/
│   │   ├── loader.py           # Document dataset loader
│   │   └── video_processor.py  # OpenCV frame sampling & timestamping
│   ├── captioning/
│   │   └── vlm_captioner.py    # Local Qwen3-VL frame vision captioner
│   ├── indexing/
│   │   ├── chunker.py          # Sliding-window document chunking
│   │   ├── embedder.py         # MiniLM text embedder
│   │   └── vector_store.py     # FAISS cosine vector store
│   ├── retrieval/
│   │   ├── retriever.py        # Semantic retriever
│   │   └── reranker.py         # CrossEncoder reranker
│   └── llm/
│       └── prompter.py         # Prompt builder & local Qwen3-VL LLM client
├── scripts/
│   ├── process_video.py        # CLI: video -> frame extraction -> VLM caption -> FAISS index
│   ├── index.py                # CLI: JSON dataset indexing
│   ├── query.py                # CLI: query interface against indexed video
│   └── test_rag.py             # Debug: full step-by-step pipeline tracer
├── config/
│   └── config.yaml             # Pipeline settings
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU (tested on RTX 4050 6GB VRAM)

### Installation

```bash
git clone https://github.com/manohar-munna/videoRAG.git
cd videoRAG
pip install -r requirements.txt
```

---

## Running the Web UI

Launch the VideoRAG FastAPI server:

```powershell
$env:PYTHONUTF8=1
python src/videorag/server.py --port 8000
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:8000/`**

---

## Live Real Video Test Example

**Video File:** `Video Footage/sample_cctv.mp4` (13.5 minutes, 24,338 frames)  
**Vision Engine:** Local `Qwen3-VL 4B` (`Q4_K_M`) + `mmproj-Qwen3VL-4B-Instruct-F16.gguf` on **RTX 4050 GPU**

**Operator Query:** `"What vehicles or trucks are visible in the CCTV video and at what timestamp?"`

### Local VLM Analysis Answer

> **Vehicles & trucks detected in CCTV video:**
>
> - **White pickup truck** — visible at **00:00:00**, **00:01:30**, **00:09:00**, **00:09:30**, and **00:10:00**.
> - **White “MOTION PICTURE” production truck** — visible at **00:00:00**.
> - **Large camera rig on a flatbed truck** — visible at **00:09:30**.
> - **Large camera rig on a trailer** — visible at **00:10:00**.
> - **Two bicycles** — visible at **00:01:30**.

### Retrieval Evaluation

| Metric | Score |
|---|---|
| Precision@5 | **1.0** |
| Recall Estimate | **1.0** |
| MRR | **1.0** |
| NDCG@5 | **1.0** |

---

## License

MIT License. Built for high-security environments requiring local video intelligence.
