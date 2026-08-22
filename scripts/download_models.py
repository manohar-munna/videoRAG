"""
scripts/download_models.py
---------------------------
Helper script to download Qwen3-VL 4B Instruct GGUF model weights
and mmproj vision projector for offline local inference.

Usage:
    python scripts/download_models.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = _PROJECT_ROOT / "models" / "qwen3_vl"

MODEL_FILENAME = "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
MMPROJ_FILENAME = "mmproj-Qwen3VL-4B-Instruct-F16.gguf"


def download_weights():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = MODELS_DIR / MODEL_FILENAME
    mmproj_path = MODELS_DIR / MMPROJ_FILENAME

    print("=" * 60)
    print("VideoRAG — Model Weights Downloader")
    print(f"Destination: {MODELS_DIR}")
    print("=" * 60)

    if model_path.exists() and mmproj_path.exists():
        print(f"[OK] Model already present: {model_path} ({model_path.stat().st_size / (1024**2):.1f} MB)")
        print(f"[OK] Vision projector already present: {mmproj_path} ({mmproj_path.stat().st_size / (1024**2):.1f} MB)")
        print("\nAll model weights are ready for offline inference!")
        return

    try:
        from huggingface_hub import hf_hub_download
        print("\nDownloading required weights from Hugging Face...")
        if not model_path.exists():
            print(f"Downloading {MODEL_FILENAME} (~2.5 GB)...")
            hf_hub_download(
                repo_id="Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
                filename=MODEL_FILENAME,
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
            )
        if not mmproj_path.exists():
            print(f"Downloading {MMPROJ_FILENAME} (~836 MB)...")
            hf_hub_download(
                repo_id="Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
                filename=MMPROJ_FILENAME,
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
            )
        print("\n[SUCCESS] Model weights downloaded successfully!")
    except Exception as e:
        print(f"\n[NOTE] Automatic download failed: {e}")
        print("You can download the GGUF model files manually into 'models/qwen3_vl/':")
        print(f"  1. {MODEL_FILENAME}")
        print(f"  2. {MMPROJ_FILENAME}")
        print("Alternatively, you can set GOOGLE_API_KEY in .env to use Gemini API.")


if __name__ == "__main__":
    download_weights()
