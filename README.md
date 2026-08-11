<div align="center">

<img src="assets/logo.png" alt="VideoRAG Logo" width="120" />

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
| 🧠 **Video RAG Pipeline** | Combines vision-language models with vector retrieval for deep video understanding |
| 🛡️ **Defence-Grade** | Designed for high-security environments with strict data handling requirements |
| ⚡ **Real-Time & Archival** | Works on live feeds and historical footage archives |
| 🗂️ **Structured Index** | Scene-level, event-level, and object-level indexing for fine-grained retrieval |
| 🔒 **Air-Gap Friendly** | Deployable on isolated, on-premise infrastructure with local LLMs |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CCTV Feeds / Archive                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Ingestion  │  Frame sampling · ASR · Scene detection
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Captioning │  Vision-Language Model (VLM)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Embedding  │  Multimodal vector embeddings
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Vector Store│  FAISS / hnswlib index
                    └──────┬──────┘
                           │
          ┌────────────────▼────────────────┐
          │         RAG Query Engine        │
          │  Query → Retrieve → LLM → Answer│
          └────────────────┬────────────────┘
                           │
                    ┌──────▼──────┐
                    │  UI / API   │  Timestamp links · Video player · Search UI
                    └─────────────┘
```

---

## Use Cases

- **Perimeter Security** — Detect and retrieve footage of unauthorised access attempts
- **Incident Investigation** — Rapidly locate events without manual review
- **Pattern Analysis** — Find recurring behaviours across days or weeks of footage
- **Object & Person Tracking** — Trace individuals or vehicles across multiple camera feeds
- **Threat Detection** — Semantic search for abandoned objects, crowd anomalies, or restricted-zone breaches

---

## Project Structure

```
videorag/
├── ingestion/          # Video ingestion, frame sampling, scene splitting
├── captioning/         # VLM-based frame and scene captioning
├── embedding/          # Multimodal embedding generation
├── indexing/           # Vector store management and indexing pipeline
├── retrieval/          # RAG query engine and ranking
├── api/                # REST API layer
├── ui/                 # Desktop / web interface
├── config/             # Configuration files
└── docs/               # Documentation
```

---

## Getting Started

> **Note:** Full setup documentation is in progress. This section will be updated as the project develops.

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for VLM inference)
- FFmpeg

### Installation

```bash
git clone https://github.com/manohar-munna/videoRAG.git
cd videoRAG
pip install -r requirements.txt
```

### Quick Start

```bash
# Index a folder of CCTV footage
python -m videorag index --source /path/to/footage --camera-id CAM-01

# Query your indexed footage
python -m videorag query "person carrying a bag near the entrance after midnight"
```

---

## Roadmap

- [ ] Core ingestion and captioning pipeline
- [ ] Vector indexing with timestamp metadata
- [ ] Natural language query API
- [ ] Multi-camera cross-search
- [ ] Desktop UI with integrated video player
- [ ] Live feed support
- [ ] On-premise LLM integration (Ollama / vLLM)
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
