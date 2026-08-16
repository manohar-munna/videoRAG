<div align="center">

# VideoRAG — CCTV Intelligence Platform

**Semantic video retrieval, real-time multi-camera ingestion, and timestamp-precise live stream DVR search powered by Local Vision-Language RAG & Edge dHash/pHash Compute Optimization.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Edge Filter](https://img.shields.io/badge/Edge%20Filter-dHash%20%7C%20pHash-38bdf8)]()
[![RAG Architecture](https://img.shields.io/badge/FAISS-384--D%20Inner%20Product-0284c7)]()
[![LLM Backend](https://img.shields.io/badge/VLM-Qwen3--VL%204B-green)]()
[![UI](https://img.shields.io/badge/UI-Surveillance%20Control%20Center-darkred)]()

</div>

---

## 📌 Overview

**VideoRAG** is a defence-grade CCTV intelligence platform that brings Retrieval-Augmented Generation (RAG) to real-time video surveillance and multi-camera stream monitoring.

Instead of sending every raw frame to heavy Vision-Language Models (VLMs), VideoRAG converts CCTV frames into 64-bit perceptual fingerprints. By measuring the **Hamming Distance** against previous keyframes in real time (< 0.2ms), static duplicate frames are filtered at the edge, saving **80–90% of VLM compute** while indexing timestamped surveillance events into a high-dimensional vector store (FAISS).

---

## 🚀 Key Features

### 🎥 Multi-Camera Stream Ingestion & Classification
- **Supported Feed Types**:
  - `📹 RECORDED MP4`: Local CCTV recordings with seekable timeline controls.
  - `▶️ YT VIDEO`: Static uploaded YouTube video clips with `start=sec` deep-linking.
  - `🔴 24/7 LIVE STREAM`: Continuous YouTube Live & RTSP streams with real-time frame extraction.
- **Fair Round-Robin Auto-Indexer**: Bounded per-camera queue processing prevents high-frequency streams from starving newly added cameras. Instant queue clearing when streams are paused or removed.
- **YouTube-Style Progress Bar**: Interactive extraction scrubbers displaying percentage processed, keyframes indexed, and compute savings telemetry.

### 🕒 Timezone-Invariant & Synchronized DVR Seeking
- **Clock Synchronization Engine**: Client-server clock offset tracking (`serverTimeOffsetMs`) calculated via `/api/health` latency telemetry, eliminating drift caused by client browser timezones or container virtualization.
- **Absolute Epoch Metadata**: Preserves Unix `epoch_time` and `seconds` across FAISS index compilation and search retrieval.
- **Interactive Dual-Mode Player**:
  - **`▶️ Rewound Live DVR`**: Seeks the live stream playhead backward by the exact time delta ($\Delta t = \text{Adjusted Now} - \text{Event Time}$).
  - **`📸 Snapshot Evidence`**: Instant high-definition keyframe snapshot display for verified evidence viewing or out-of-bounds historical moments (> 12 hours old).

### 🔍 RAG Architecture & Vector Debugger
- **Hybrid Retrieval & Reranking**: Dense FAISS vector retrieval (`all-MiniLM-L6-v2`) combined with cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`).
- **Interactive Timestamp Deep Links**: LLM security analysis output automatically parses camera mentions and timestamps into clickable video jump buttons.
- **Vector Debugger & JSON Explorer**: Live trace inspection of query embeddings, execution timings (embedding, FAISS, rerank, LLM), and full per-camera event dataset filtering.

---

## ⚡ Edge dHash / pHash Compute Pipeline

```
   Raw CCTV Video Stream (30 FPS)
                 │
                 ▼
     ┌───────────────────────┐
     │ 1. Motion & pHash/    │  Compute 64-Bit dHash / pHash Fingerprints (< 0.2ms)
     │    dHash Filter       │  Calculate Hamming Distance (Bitwise XOR)
     └───────────┬───────────┘
                 │ Hamming Distance >= Threshold (Significant Scene Shift)
                 ▼
     ┌───────────────────────┐  [DROP STATIC FRAMES]
     │ 2. Keyframe           │  If Hamming Dist < Threshold -> Discard Frame
     │    Extraction         │  Zero VLM / LLM Compute Wasted!
     └───────────┬───────────┘
                 │ Keyframes Only
                 ▼
     ┌───────────────────────┐
     │ 3. Local VLM Engine   │  Qwen3-VL 4B (Local llama-server / OpenAI API)
     │    & FAISS Vector RAG │  Generate descriptions & index keyframe events
     └───────────────────────┘
```

---

## 🛠️ Getting Started

### 1. Installation

```bash
git clone https://github.com/manohar-munna/videoRAG.git
cd videoRAG
pip install -r requirements.txt
```

### 2. Running the Web Server & Control Center

```powershell
$env:PYTHONUTF8=1
python -u src/videorag/server.py --port 8000
```

Open your browser to:  
👉 **`http://127.0.0.1:8000/`**

---

## 📄 License

MIT License. Built for high-security environments requiring local edge video intelligence.
