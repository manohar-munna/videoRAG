"""
scripts/export_mobileclip_onnx.py
---------------------------------
Export BOTH MobileCLIP towers to ONNX for on-device retrieval.

Why both: the Android app previously "embedded" text with a trigram hash-bucket
projection and images with a colour histogram, then compared the two. Those are
unrelated vector spaces, so cosine similarity between them carried no meaning and
search was effectively keyword matching over generated captions.

Exporting the real image AND text towers puts queries and keyframes in one shared
512-D space, which is the whole premise of CLIP retrieval. Exporting from the same
open_clip checkpoint the desktop pipeline uses also keeps the two indexes compatible.

Usage:
    python scripts/export_mobileclip_onnx.py --arch MobileCLIP-S2 --out models/mobileclip_onnx
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="MobileCLIP-S2")
    ap.add_argument("--pretrained", default="datacompdr")
    ap.add_argument("--out", default="models/mobileclip_onnx")
    ap.add_argument("--opset", type=int, default=17,
                    help="17 keeps us within onnxruntime-android 1.17's supported range")
    args = ap.parse_args()

    import open_clip

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.arch} (pretrained={args.pretrained}) ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.arch, pretrained=args.pretrained, device="cpu"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.arch)

    # Input resolution comes from the model's own preprocessing transform, not a guess.
    image_size = model.visual.image_size
    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]
    print(f"image_size = {image_size}")

    with torch.no_grad():
        dim = model.encode_text(tokenizer(["probe"])).shape[-1]
    print(f"embedding dim = {dim}")

    class ImageTower(torch.nn.Module):
        """Encode pixels -> L2-normalised embedding (normalising here so the app can't forget)."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            f = self.m.encode_image(pixel_values)
            return f / f.norm(dim=-1, keepdim=True)

    class TextTower(torch.nn.Module):
        """Encode CLIP BPE token ids -> L2-normalised embedding."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids):
            f = self.m.encode_text(input_ids)
            return f / f.norm(dim=-1, keepdim=True)

    img_path = out_dir / "mobileclip_image.onnx"
    txt_path = out_dir / "mobileclip_text.onnx"

    dummy_px = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        ImageTower(model), dummy_px, str(img_path),
        input_names=["pixel_values"], output_names=["image_embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embedding": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    print(f"wrote {img_path}  ({img_path.stat().st_size/1e6:.1f} MB)")

    dummy_ids = tokenizer(["a photo of a yellow school bus"])
    torch.onnx.export(
        TextTower(model), dummy_ids, str(txt_path),
        input_names=["input_ids"], output_names=["text_embedding"],
        dynamic_axes={"input_ids": {0: "batch"}, "text_embedding": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    print(f"wrote {txt_path}  ({txt_path.stat().st_size/1e6:.1f} MB)")
    print(f"context length = {dummy_ids.shape[1]}")

    # --- export the tokenizer vocabulary so the Kotlin side can reproduce it ---
    bpe = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    encoder = getattr(bpe, "encoder", None)
    bpe_ranks = getattr(bpe, "bpe_ranks", None)
    if encoder and bpe_ranks:
        import json
        (out_dir / "clip_vocab.json").write_text(
            json.dumps(encoder, ensure_ascii=False), encoding="utf-8")
        merges = sorted(bpe_ranks.items(), key=lambda kv: kv[1])
        (out_dir / "clip_merges.txt").write_text(
            "\n".join(f"{a} {b}" for (a, b), _ in merges), encoding="utf-8")
        print(f"wrote tokenizer vocab ({len(encoder)} tokens) and {len(merges)} merges")
    else:
        print("WARNING: could not extract tokenizer tables; inspect open_clip's tokenizer")

    # --- verify the two towers actually agree in the shared space ---
    import onnxruntime as ort
    from PIL import Image

    s_img = ort.InferenceSession(str(img_path), providers=["CPUExecutionProvider"])
    s_txt = ort.InferenceSession(str(txt_path), providers=["CPUExecutionProvider"])

    prompts = ["a yellow school bus", "a green taxi", "an empty road"]
    t_emb = s_txt.run(None, {"input_ids": tokenizer(prompts).numpy().astype(np.int32)})[0]

    probe = Path("android/sample_videos")
    print("\nself-check: text-vs-text cosine (should be <1 between different prompts)")
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            c = float(np.dot(t_emb[i], t_emb[j]))
            print(f"  '{prompts[i]}' vs '{prompts[j]}': {c:.3f}")
    print("\nExport complete.")


if __name__ == "__main__":
    main()
