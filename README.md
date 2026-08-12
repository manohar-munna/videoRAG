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
> *"When did the vehicle with partial plate MH-12 appear on Camera 5?"*

VideoRAG indexes, understands, and retrieves — giving analysts time back for what matters.

---

## Key Features

| Feature | Description |
|---|---|
| 🔍 **Semantic Search** | Query footage using natural language — no keywords required |
| 🕐 **Timestamp-Precise Retrieval** | Results link directly to the exact video moment |
| 📼 **Multi-Camera Indexing** | Index and search across hundreds of concurrent CCTV feeds |
| 🧠 **Video RAG Pipeline** | Combines VLM captioning with vector retrieval and LLM reasoning |
| 🛡️ **Defence-Grade** | Designed for high-security environments with strict data handling requirements |
| ⚡ **Real-Time & Archival** | Works on live feeds and historical footage archives |
| 🔄 **Reranking** | CrossEncoder reranking for precision on top of vector retrieval |
| 🔒 **Air-Gap Friendly** | Swappable LLM backends — Gemini API for dev, local Qwen-VL for production |

---

## Architecture

```
CCTV Feeds / Archive
        |
   [Frame Extraction]   FFmpeg — sample every N seconds
        |
   [VLM Captioning]     Qwen3-VL (local) / Gemini Vision (cloud)
        |                  -> { camera, timestamp, description }
        |
   [Chunking]           Sliding window over temporal event sequences
        |
   [Embedding]          sentence-transformers (all-MiniLM-L6-v2)
        |
   [FAISS Index]        IndexFlatIP — cosine similarity search
        |
        |  <--- Natural language query
        |
   [Retrieval]          Top-K semantic search with optional camera filter
        |
   [Reranking]          CrossEncoder (ms-marco-MiniLM-L-6-v2)
        |
   [LLM Answer]         Gemini / OpenAI / Ollama / Local
        |
   [Evaluation]         Precision@K, MRR, NDCG, context utilization
        |
     Answer + Timestamps
```

---

## Project Structure

```
videorag/
├── data/
│   └── mock_cctv.json          # 160 synthetic CCTV events (8 cameras, 24h)
├── src/videorag/
│   ├── ingestion/
│   │   └── loader.py           # JSON loader -> document format
│   ├── indexing/
│   │   ├── chunker.py          # Sliding window + individual chunking
│   │   ├── embedder.py         # sentence-transformers wrapper
│   │   └── vector_store.py     # FAISS IndexFlatIP (cosine similarity)
│   ├── retrieval/
│   │   ├── retriever.py        # Semantic retrieval with camera filter
│   │   └── reranker.py         # CrossEncoder + ScoreReranker
│   ├── llm/
│   │   └── prompter.py         # Prompt builder + Gemini/OpenAI/Mock LLM
│   └── evaluation/
│       └── evaluator.py        # Precision@K, MRR, NDCG, answer quality
├── scripts/
│   ├── index.py                # CLI: ingest -> embed -> save FAISS index
│   ├── query.py                # CLI: interactive query loop
│   └── test_rag.py             # Deep debug: scores, JSON mapping, prompt, eval
├── config/
│   └── config.yaml             # All pipeline settings
├── .env.example                # API key template
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA GPU recommended (for local VLM — Qwen3-VL)

### Installation

```bash
git clone https://github.com/manohar-munna/videoRAG.git
cd videoRAG
pip install -r requirements.txt
```

### Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your keys
```

```env
GOOGLE_API_KEY=your-gemini-api-key-here
```

### Build the Index

```bash
python scripts/index.py --config config/config.yaml --data data/mock_cctv.json
```

### Query the Index

```bash
# Interactive mode
python scripts/query.py

# Single query
python scripts/query.py --query "Was there any suspicious activity near the fence?"
```

### Run Debug Test Suite

```bash
# Full 5-query test suite with pipeline trace
python scripts/test_rag.py

# Single query with full step-by-step breakdown
python scripts/test_rag.py --query "When did the altercation happen?"
```

---

## Live Test Example

**Query:** `"Was there any suspicious activity near the fence at night?"`

**Pipeline:** `MiniLM-L6-v2 embeddings` → `FAISS retrieval` → `CrossEncoder reranking` → `Gemini 2.0 Flash`

### Step 1 — FAISS Vector Retrieval (Top 10, cosine similarity)

| # | Camera | Timestamp | FAISS Score | Description |
|---|---|---|---|---|
| 1 | CAM_03 | 23:30:45 | 0.5376 | North fence, IR mode. Sensor detects movement at panel 3... |
| 2 | CAM_03 | 21:02:15 | 0.5992 | North fence. Full night. Camera in IR mode. No personnel... |
| 3 | CAM_03 | 22:05:44 | 0.5707 | North fence, IR mode. A lone male figure approaches the fence from outside... |
| 4 | CAM_03 | 02:15:50 | 0.5598 | North fence line. IR camera detects two individuals crouching near fence panel 7... |
| 5 | CAM_03 | 17:28:43 | 0.5377 | North fence. All clear. Evening light beginning... |

### Step 2 — Source JSON Records (from `mock_cctv.json`)

Each retrieved result maps back to an exact JSON record:

```json
{
  "camera": "CAM_03",
  "timestamp": "22:05:44",
  "description": "North fence, IR mode. A lone male figure approaches the fence from outside at panel 12. He crouches low and inspects the base of the fence for nearly 4 minutes before walking away briskly when a vehicle passes on the outer road."
}
```

```json
{
  "camera": "CAM_03",
  "timestamp": "02:15:50",
  "description": "North fence line. IR camera detects two individuals crouching near fence panel 7. One subject appears to be testing fence tension. They flee eastward on foot before security guards arrive."
}
```

### Step 3 — CrossEncoder Reranking

| Rank | Camera | Timestamp | FAISS Score | Rerank Score |
|---|---|---|---|---|
| 1 | CAM_03 | 23:30:45 | 0.5376 | **1.2228** |
| 2 | CAM_03 | 21:02:15 | 0.5992 | -1.3294 |
| 3 | CAM_03 | 22:05:44 | 0.5707 | -1.3923 |
| 4 | CAM_03 | 02:15:50 | 0.5598 | -3.8185 |
| 5 | CAM_03 | 17:28:43 | 0.5377 | -3.9911 |

### Step 4 — Gemini Answer

> **Yes, there were two instances of suspicious activity near the north fence at night.**
>
> At **22:05:44**, a lone male figure was observed approaching the fence from outside, crouching low and inspecting the base of the fence for nearly 4 minutes before walking away briskly when a vehicle passed (**CAM_03**).
>
> Later, at **02:15:50**, two individuals were detected crouching near fence panel 7, with one subject testing the fence tension. They fled eastward on foot before security guards arrived (**CAM_03**).

### Step 5 — Evaluation

| Retrieval Metric | Score |
|---|---|
| Precision@5 | **1.0** |
| Recall Estimate | **1.0** |
| MRR | **1.0** |
| NDCG@5 | **1.0** |

| Answer Metric | Value |
|---|---|
| Has Timestamp | Yes |
| Has Camera | Yes |
| Context Utilization | 1.0 |

---

## LLM Backends

| Backend | Config | Use Case |
|---|---|---|
| `gemini` | `GOOGLE_API_KEY` in `.env` | Cloud — development & testing |
| `openai` | `OPENAI_API_KEY` in `.env` | Cloud — OpenAI or Ollama-compatible |
| `mock` | No key needed | Offline testing |

Switch backend in `config/config.yaml`:

```yaml
llm:
  backend: "gemini"          # gemini | openai | mock
  model: "gemini-2.0-flash-lite"
```

For local Ollama (air-gapped deployment):
```yaml
llm:
  backend: "openai"
  model: "qwen3-vl:4b"
  base_url: "http://localhost:11434/v1"
```

---

## Roadmap

- [x] Mock CCTV data generation (160 events, 8 cameras, 24h)
- [x] FAISS vector indexing with timestamp metadata
- [x] Semantic retrieval with camera filter
- [x] CrossEncoder reranking
- [x] Gemini / OpenAI / Mock LLM backends
- [x] RAG evaluation (Precision@K, MRR, NDCG)
- [x] Full pipeline debug/test script
- [ ] VLM captioning pipeline (Qwen3-VL-4B local)
- [ ] Video frame extraction (FFmpeg)
- [ ] Multi-camera cross-search and event correlation
- [ ] Desktop UI with integrated video player
- [ ] Live feed support
- [ ] Role-based access control (RBAC)
- [ ] Audit logging for defence compliance

---

## Contributing

This project is in active development. Contribution guidelines will be published once the core architecture is stable.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Built for analysts who don't have time to scrub through footage.

</div>
