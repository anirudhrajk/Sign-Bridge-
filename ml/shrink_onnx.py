"""Halve a model's download size: store large weights as float16 and convert them back to float32
inside the graph (one Cast per weight), so every layer still computes in float32 on every browser.

  python ml/shrink_onnx.py static/models/asl/model.onnx            # writes model.onnx in place
  python ml/shrink_onnx.py in.onnx --out out.onnx --check data/cache_v2/asl.npz
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
import sys
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def shrink(model, min_size=1024):
    g = model.graph
    used_as_input = {i.name for i in g.input}
    new_inits, casts = [], []
    for init in g.initializer:
        if init.data_type != TensorProto.FLOAT or init.name in used_as_input:
            new_inits.append(init)
            continue
        arr = numpy_helper.to_array(init)
        if arr.size < min_size or np.abs(arr).max() > 60000:
            new_inits.append(init)
            continue
        half = numpy_helper.from_array(arr.astype(np.float16), init.name + "__f16")
        new_inits.append(half)
        casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT,
                                      name=init.name + "__to_f32"))
    del g.initializer[:]
    g.initializer.extend(new_inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", default="")
    ap.add_argument("--check", default="", help="cache .npz: compare answers before and after on 2000 recordings")
    args = ap.parse_args()
    src = Path(args.model)
    out = Path(args.out) if args.out else src
    m = shrink(onnx.load(str(src)))
    onnx.checker.check_model(m)
    before = src.stat().st_size
    if args.check:
        import onnxruntime as ort
        a = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
        b = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        T = a.get_inputs()[0].shape[1]
        from ml.calibrate import npz_memmap
        X = npz_memmap(args.check, "X")
        idx = np.floor(np.linspace(0, X.shape[1] - 1, T) + 0.5).astype(int)
        diff, agree = 0.0, 0
        pick = np.linspace(0, len(X) - 1, 2000).astype(int)
        for i in pick:
            s = X[i][idx].astype(np.float32)[None]
            mk = np.isfinite(s[..., 0]).astype(np.float32)
            s = np.nan_to_num(s)
            la = a.run(None, {"points": s, "mask": mk})[0][0]
            lb = b.run(None, {"points": s, "mask": mk})[0][0]
            pa = np.exp(la - la.max()); pa /= pa.sum()
            pb = np.exp(lb - lb.max()); pb /= pb.sum()
            diff = max(diff, float(np.abs(pa - pb).max()))
            agree += la.argmax() == lb.argmax()
        print(f"same top answer on {agree}/{len(pick)} recordings; largest probability change {diff:.4f}")
    onnx.save(m, str(out))
    print(f"{src.name}: {before / 1e6:.1f} MB -> {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
