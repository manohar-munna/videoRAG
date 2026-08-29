"""
scripts/verify_clip_retrieval.py
--------------------------------
Pre-flight for the Android Phase 2 work: does a real dual-tower CLIP rank the
keyframe containing the yellow bus above the frames that do not?

The app currently answers "there is no yellow bus" because its fake embedder ranked
00:00:00 and 00:00:13 top for that query while the bus is at 00:00:29. That is a
retrieval failure, not a model failure. This scores the same candidate frames with
real CLIP embeddings so we know the port is worth doing before writing the Kotlin
BPE tokenizer.

Also quantises both towers to int8, since 398 MB of fp32 is impractical on device,
and re-scores to confirm quantisation does not destroy the ranking.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CANDIDATE_SECONDS = [0, 13, 18, 24, 29, 34, 39, 44]   # what the dHash gate kept
BUS_SECONDS = 29                                       # the one containing the yellow bus


def frames_from_video(video: Path, seconds, size: int):
    import cv2
    cap = cv2.VideoCapture(str(video))
    out = {}
    for s in seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
        ok, fr = cap.read()
        if ok:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            out[s] = Image.fromarray(fr)
    cap.release()
    return out


def score(sess_img, sess_txt, preprocess, tokenizer, frames, queries):
    import numpy as np
    px = np.stack([preprocess(im).numpy() for im in frames.values()]).astype(np.float32)
    img_emb = sess_img.run(None, {"pixel_values": px})[0]
    ids = tokenizer(queries).numpy().astype(np.int64)      # the towers expect int64
    txt_emb = sess_txt.run(None, {"input_ids": ids})[0]
    return img_emb @ txt_emb.T                              # already L2-normalised in-graph


def report(title, sims, secs, queries):
    print(f"\n=== {title} ===")
    for qi, q in enumerate(queries):
        order = np.argsort(-sims[:, qi])
        ranked = [(secs[i], float(sims[i, qi])) for i in order]
        top = ", ".join(f"{s}s={v:.3f}" for s, v in ranked[:4])
        rank_of_bus = [s for s, _ in ranked].index(BUS_SECONDS) + 1
        flag = "OK" if rank_of_bus == 1 else f"bus at rank {rank_of_bus}"
        print(f"  '{q}'\n      {top}\n      -> {flag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/mobileclip_onnx")
    ap.add_argument("--video",
                    default="android/sample_videos/Sysvideo 4K 8 Megapixel  IP Camera Demo traffic car(720P_HD).mp4")
    ap.add_argument("--arch", default="MobileCLIP-S2")
    args = ap.parse_args()

    import open_clip
    import onnxruntime as ort

    d = Path(args.onnx)
    img_f, txt_f = d / "mobileclip_image.onnx", d / "mobileclip_text.onnx"

    _, _, preprocess = open_clip.create_model_and_transforms(
        args.arch, pretrained=None, device="cpu")
    tokenizer = open_clip.get_tokenizer(args.arch)

    frames = frames_from_video(Path(args.video), CANDIDATE_SECONDS, 256)
    secs = list(frames.keys())
    print(f"scored frames (s): {secs}   bus is at {BUS_SECONDS}s")

    queries = ["a yellow school bus", "yellow bus", "a green taxi", "an empty road with no vehicles"]

    s_img = ort.InferenceSession(str(img_f), providers=["CPUExecutionProvider"])
    s_txt = ort.InferenceSession(str(txt_f), providers=["CPUExecutionProvider"])
    sims = score(s_img, s_txt, preprocess, tokenizer, frames, queries)
    report("fp32", sims, secs, queries)

    # ---- int8 dynamic quantisation: 398 MB of fp32 is not shippable ----
    from onnxruntime.quantization import quantize_dynamic, QuantType
    qi, qt = d / "mobileclip_image.int8.onnx", d / "mobileclip_text.int8.onnx"
    for src, dst in ((img_f, qi), (txt_f, qt)):
        if not dst.exists():
            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        print(f"{dst.name}: {src.stat().st_size/1e6:.0f} MB -> {dst.stat().st_size/1e6:.0f} MB")

    s_img_q = ort.InferenceSession(str(qi), providers=["CPUExecutionProvider"])
    s_txt_q = ort.InferenceSession(str(qt), providers=["CPUExecutionProvider"])
    sims_q = score(s_img_q, s_txt_q, preprocess, tokenizer, frames, queries)
    report("int8 quantised", sims_q, secs, queries)


if __name__ == "__main__":
    main()
