"""
scripts/reindex_real.py
-----------------------
Quick indexing script for real CCTV events without TTY blocking.
"""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from videorag.indexing.embedder import TextEmbedder
from videorag.indexing.vector_store import FAISSVectorStore

def main():
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "real_cctv_events.json"
    index_path = root / "index" / "cctv_index"

    with open(data_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    print(f"Indexing {len(events)} real CCTV events from {data_path}...")
    embedder = TextEmbedder("all-MiniLM-L6-v2")

    texts = [
        f"Camera: {e.get('camera')} | Time: {e.get('timestamp')} | Event: {e.get('description')}"
        for e in events
    ]
    embeddings = embedder.embed(texts)

    store = FAISSVectorStore(dim=384, index_type="flat")
    store.add(embeddings, metadata=events)
    store.save(str(index_path))
    print(f"FAISS index saved successfully with {store.size} vectors to {index_path}.faiss")

if __name__ == "__main__":
    main()
