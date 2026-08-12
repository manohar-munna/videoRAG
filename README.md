<div align="center">

# VideoRAG — CCTV Intelligence Platform

**Semantic video retrieval, indexing, and timestamp-precise search with Edge dHash/pHash LLM Compute Optimization for defence-grade surveillance systems.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Edge Filter](https://img.shields.io/badge/Edge%20Filter-dHash%20%7C%20pHash-38bdf8)]()
[![UI](https://img.shields.io/badge/UI-Minimal%20Glassmorphic-0284c7)]()
[![Status](https://img.shields.io/badge/status-active%20development-brightgreen)]()
[![Defence](https://img.shields.io/badge/domain-defence%20%26%20security-red)]()

</div>

---

## Overview

**VideoRAG** is a defence-grade CCTV intelligence platform that brings Retrieval-Augmented Generation (RAG) to video surveillance. It features a **minimalist glassmorphic Web UI** with an **Edge Frame Hashing & Motion Optimizer** (`dHash` / `pHash`).

Instead of sending every raw frame to heavy Vision-Language Models (VLMs), VideoRAG converts CCTV frames into 64-bit binary perceptual fingerprints. By measuring the **Hamming Distance** against previous keyframes, static duplicate frames are filtered out at the edge **in < 0.2ms**, saving **80–90% of VLM compute** and battery resources on edge CCTV devices.

---

## ⚡ Edge dHash / pHash Compute Optimization

```
   Raw CCTV Video Stream (30 FPS)
                 │
                 ▼
     ┌───────────────────────┐
     │ 1. Motion & pHash/    │  Compute 64-Bit dHash / pHash Fingerprints (<0.2ms)
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
     │ 3. Local VLM Engine   │  Qwen3-VL 4B (CUDA GPU)
     │    & FAISS Vector RAG │  Generate descriptions & index keyframe events
     └───────────────────────┘
```

### Hash Algorithms Supported
- **dHash (Difference Hash)**: Compares horizontal pixel gradients across a 9x8 grid. Fast and ideal for edge CCTV CPUs.
- **pHash (Perceptual Hash)**: Uses 2D Discrete Cosine Transform (DCT) low-frequency components. Robust against illumination & noise shifts.
- **aHash (Average Hash)**: Compares pixel intensities against frame mean.

---

## 🎨 Developer Mode & Edge Inspector UI

Click **`Dev Mode: ON`** in the Web UI header to inspect abstract edge compute metrics live:

- 🎛️ **Live Config Controls**: Switch between `dHash`, `pHash`, and `aHash`, or adjust Hamming Distance Threshold sliders (0–32).
- 📊 **Real-Time KPI Badges**:
  - `Total Frames Sampled`
  - `Keyframes Kept (VLM)`
  - `Static Frames Skipped`
  - **`LLM Compute Saved (%)`**
- 📜 **Frame-by-Frame Hash Audit Trail Table**: Displays 64-bit Hex Fingerprints (`0xd99159b3636bd332`), Hamming Distances, Motion %, and decision status pills (`KEYFRAME (KEEP)` vs `STATIC (SKIP)`).

---

## Getting Started

### Installation

```bash
git clone https://github.com/manohar-munna/videoRAG.git
cd videoRAG
pip install -r requirements.txt
```

### Running the Web Server & UI

```powershell
$env:PYTHONUTF8=1
python src/videorag/server.py --port 8000
```

Open your browser to:  
👉 **`http://127.0.0.1:8000/`**

### Running Video Processing with dHash Smart Filtering (CLI)

```powershell
$env:PYTHONUTF8=1
python scripts/process_video.py --video "Video Footage/sample_cctv.mp4" --camera-id CAM_01 --interval 15 --backend local --enable-hash-filter --hash-method dhash --threshold 10
```

---

## License

MIT License. Built for high-security environments requiring local edge video intelligence.
