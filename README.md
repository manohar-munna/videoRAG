<div align="center">

# VideoRAG — CCTV Intelligence Platform

**Semantic video retrieval, indexing, and timestamp-precise search for defence-grade surveillance systems.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active%20development-brightgreen)]()
[![Defence](https://img.shields.io/badge/domain-defence%20%26%20security-red)]()

</div>

---

## Overview

**VideoRAG** is a defence-grade CCTV intelligence platform that brings Retrieval-Augmented Generation (RAG) to video surveillance. It enables operators and analysts to semantically query hours of CCTV footage in natural language and instantly navigate to the exact timestamp of interest — no manual scrubbing required.

Instead of reviewing footage frame-by-frame, simply ask:

> *"Show me the moment a person in a red jacket entered the north gate."*
> *"Find all instances of unattended baggage near Gate 3 after 22:00."*
> *"What vehicles or trucks are visible in the CCTV video and at what timestamp?"*

VideoRAG indexes, understands, and retrieves — giving analysts time back for what matters.

---

## Key Features

| Feature | Description |
|---|---|
| 🔍 **Semantic Search** | Query real video footage in natural language |
| 🕐 **Timestamp-Precise Retrieval** | Direct timestamp links to exact video moments |
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
        |  <--- Natural language operator query
        |
   [Retriever]          Top-K dense retrieval + camera filter
        |
   [Reranker]           CrossEncoder (ms-marco-MiniLM-L-6-v2)
        |
   [Local LLM Answer]   Qwen3-VL 4B (Local GGUF on CUDA GPU)
        |
     Timestamped Video Search Answer
```

---

## Project Structure

```
videorag/
├── Video Footage/
│   └── sample_cctv.mp4         # Target CCTV video file
├── Local LLM 3VL 4Q/
│   ├── Qwen3VL-4B-Instruct-Q4_K_M.gguf      # Local LLM text weights
│   └── mmproj-Qwen3VL-4B-Instruct-F16.gguf  # Local VLM vision projector
├── tools/
│   └── llama/                  # Embedded CUDA llama-server binary
├── src/videorag/
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

## How to Test Real Video Footage

### Step 1: Place Your Video Footage

Place any MP4 surveillance video in the `Video Footage/` directory:

```bash
Video Footage/sample_cctv.mp4
```

### Step 2: Run End-to-End Video Processing & Indexing

Run `process_video.py` to extract frame images, analyze them using your local **Qwen3-VL GPU Vision model**, and build the vector search index:

```bash
python scripts/process_video.py --video "Video Footage/sample_cctv.mp4" --camera-id CAM_01 --interval 15 --backend local
```

*What this does automatically:*
1. Samples frames every 15 seconds from the video.
2. Sends frame pixels to local Qwen3-VL VLM for vision captioning.
3. Generates `data/real_cctv_events.json`.
4. Embeds text descriptions and builds `index/cctv_index.faiss`.

### Step 3: Query Your Video in Natural Language

Query the indexed video using natural language:

```bash
python scripts/query.py --query "What vehicles or trucks are visible in the CCTV video and at what timestamp?"
```

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

## Roadmap

- [x] Synthetic CCTV data generation & baseline RAG pipeline
- [x] FAISS vector indexing with timestamp metadata
- [x] Semantic retrieval & CrossEncoder reranking
- [x] **Local LLM integration (Qwen3-VL-4B GGUF on CUDA)**
- [x] **Local VLM Vision Engine (`mmproj` frame captioning on GPU)**
- [x] **End-to-end video processing pipeline (`process_video.py`)**
- [x] **Real video footage testing & timestamp retrieval**
- [ ] Multi-camera cross-search & video player UI

---

## License

MIT License. Built for high-security environments requiring local video intelligence.
