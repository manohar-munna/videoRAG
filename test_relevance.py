import sys
from pathlib import Path

# Add src folder to python path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import yaml
from videorag.indexing.vector_store import FAISSVectorStore
from videorag.indexing.embedder import MultimodalEmbedder
from videorag.retrieval.retriever import CCTVRetriever
from videorag.retrieval.retriever import _expand_query

# 1. Print Query Expansions to verify color/noun preservation
print("--- TESTING EXPANSIONS ---")
test_queries = [
    "people wearing pink color costumes",
    "total number of red cars in the video",
    "unattended blue backpack"
]
for q in test_queries:
    print(f"\nQuery: '{q}'")
    print(f"Expansions: {_expand_query(q)}")

# 2. Validate Retrieval Score Integrity
print("\n--- RETRIEVAL ENGINE CHECK ---")
try:
    embedder = MultimodalEmbedder(model_name="MobileCLIP-S2")
    store = FAISSVectorStore(dim=512)
    store.load("index/cctv_index")
    retriever = CCTVRetriever(store, embedder)
    
    for q in test_queries:
        results = retriever.retrieve(q, top_k=3)
        print(f"\nTop Hits for '{q}':")
        for idx, r in enumerate(results):
            meta = r["metadata"]
            print(f"  {idx+1}. Score: {r['score']:.4f} | TS: {meta.get('timestamp')} | Region: {meta.get('best_crop_region')}")
except Exception as e:
    print(f"Index load skipped: {e}")
