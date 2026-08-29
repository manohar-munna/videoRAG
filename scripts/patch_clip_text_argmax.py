"""
scripts/patch_clip_text_argmax.py
---------------------------------
Make the exported CLIP text tower loadable by onnxruntime-android.

CLIP pools the text sequence at the end-of-text token, which open_clip expresses as
`text.argmax(dim=-1)` over the int64 token ids. onnxruntime-android registers ArgMax
for float and int32 but not int64, so the session fails to open:

    ORT_NOT_IMPLEMENTED - Could not find an implementation for ArgMax(13)
    node with name '/text/ArgMax'

Rather than re-export and risk changing anything else, insert a single
Cast(int64 -> int32) in front of that ArgMax. ArgMax returns indices either way, so
the pooled embedding is bit-identical; this only changes which kernel is selected.

Verified by comparing embeddings before and after the patch.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/mobileclip_onnx/mobileclip_text.onnx")
    ap.add_argument("--out", default="models/mobileclip_onnx/mobileclip_text.patched.onnx")
    args = ap.parse_args()

    src, dst = Path(args.model), Path(args.out)
    m = onnx.load(str(src))

    targets = [n for n in m.graph.node if n.op_type == "ArgMax"]
    if not targets:
        print("no ArgMax nodes; nothing to patch")
        return
    print(f"found {len(targets)} ArgMax node(s)")

    for i, node in enumerate(targets):
        original_input = node.input[0]
        cast_out = f"{original_input}_as_int32_{i}"
        cast = helper.make_node(
            "Cast", inputs=[original_input], outputs=[cast_out],
            to=TensorProto.INT32, name=f"ArgMaxInputCast_{i}",
        )
        node.input[0] = cast_out
        # insert the Cast immediately before the ArgMax so topological order holds
        idx = list(m.graph.node).index(node)
        m.graph.node.insert(idx, cast)
        print(f"  patched '{node.name}': {original_input} -> Cast(int32) -> ArgMax")

    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    print(f"wrote {dst} ({dst.stat().st_size/1e6:.1f} MB)")

    # --- prove the patch did not change the output ---
    import onnxruntime as ort
    import open_clip

    tok = open_clip.get_tokenizer("MobileCLIP-S2")
    prompts = ["a yellow school bus", "an empty road", "a green taxi"]
    ids = tok(prompts).numpy().astype(np.int64)

    a = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    ea = a.run(None, {"input_ids": ids})[0]
    eb = b.run(None, {"input_ids": ids})[0]

    delta = float(np.abs(ea - eb).max())
    cos = [float(np.dot(ea[i], eb[i])) for i in range(len(prompts))]
    print(f"\nmax abs difference : {delta:.3e}")
    print(f"cosine per prompt  : {[round(c,6) for c in cos]}")
    print("PASS - embeddings unchanged" if delta < 1e-5 else "FAIL - patch altered outputs")


if __name__ == "__main__":
    main()
