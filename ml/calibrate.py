"""Make a model's confidence honest: fit one temperature on held-out recordings so that
"70% sure" means about 70% right, then build it into the ONNX file (logits / T).

The sign page adds a word by itself only at >= 70% confidence, so a model that is right but
timid (strong label smoothing / mixup) would otherwise add too few words, and an overconfident
one too many wrong words.

  python ml/calibrate.py data/models_v2/asl_v4 --lang asl            # fit on the held-out split
  python ml/calibrate.py data/models_v2/asl_final --temperature 0.62 # apply a known temperature

Writes model.onnx (and model.json "temperature") in place; the original is kept as model_raw.onnx.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.train import split  # noqa: E402


def npz_memmap(path, key):
    """Read one array of an uncompressed .npz straight from disk, without loading it into memory."""
    import zipfile
    from numpy.lib import format as npf
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(key + ".npy")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        head = f.read(30)
        start = info.header_offset + 30 + int.from_bytes(head[26:28], "little") + int.from_bytes(head[28:30], "little")
        f.seek(start)
        version = npf.read_magic(f)
        shape, fortran, dtype = npf._read_array_header(f, version)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order="F" if fortran else "C")


def held_out_logits(folder, lang, cache="", limit=6000):
    m = json.loads((Path(folder) / "model.json").read_text())
    raw = Path(folder) / "model_raw.onnx"
    sess = ort.InferenceSession(str(raw if raw.exists() else Path(folder) / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    d = np.load(ROOT / "data" / "cache_v2" / f"{cache or lang}.npz", allow_pickle=True)
    labels, signers = d["labels"], d["signers"]
    if "split" in d.files:
        va = np.flatnonzero(d["split"] == 1)
    else:
        _, va = split(labels, 0.2, 7, signers if (signers >= 0).all() else None)
    va = va[np.linspace(0, len(va) - 1, min(limit, len(va))).astype(int)]
    idx = np.floor(np.linspace(0, d["X"].shape[1] - 1, m["frames"]) + 0.5).astype(int)
    X = npz_memmap(ROOT / "data" / "cache_v2" / f"{cache or lang}.npz", "X")
    L = []
    for i in va:
        s = X[i][idx].astype(np.float32)[None]
        mk = np.isfinite(s[..., 0]).astype(np.float32)
        L.append(sess.run(None, {"points": np.nan_to_num(s), "mask": mk})[0][0])
    return np.array(L), labels[va]


def nll(L, y, t):
    z = L / t
    z = z - z.max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return -lp[np.arange(len(y)), y].mean()


def fit(L, y):
    ts = np.exp(np.linspace(np.log(0.2), np.log(5), 400))
    return float(ts[np.argmin([nll(L, y, t) for t in ts])])


def report(L, y, t, tag):
    z = L / t
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    conf, right = p.max(1), p.argmax(1) == y
    k = conf >= 0.7
    print(f"  {tag}: T={t:.3f}  >=70% sure on {k.mean():.0%} of signs, right {right[k].mean():.1%} of those; "
          f"mean confidence {conf.mean():.2f} vs accuracy {right.mean():.2f}")


def apply(folder, t):
    folder = Path(folder)
    raw = folder / "model_raw.onnx"
    if not raw.exists():
        shutil.copy(folder / "model.onnx", raw)
    m = onnx.load(str(raw))
    g = m.graph
    out = next(o for o in g.output if o.name == "logits")
    for n in g.node:  # rename the producer of "logits" and divide it by T
        n.output[:] = ["logits_raw" if o == "logits" else o for o in n.output]
    g.initializer.append(numpy_helper.from_array(np.array(t, dtype=np.float32), "temperature"))
    g.node.append(helper.make_node("Div", ["logits_raw", "temperature"], ["logits"], name="calibrate"))
    onnx.checker.check_model(m)
    onnx.save(m, str(folder / "model.onnx"))
    mj = json.loads((folder / "model.json").read_text())
    mj["temperature"] = round(t, 4)
    (folder / "model.json").write_text(json.dumps(mj))
    print(f"wrote {folder / 'model.onnx'} with temperature {t:.3f} (original kept as model_raw.onnx)")
    del out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--lang", default="asl")
    ap.add_argument("--cache", default="")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    t = args.temperature
    if not t:
        L, y = held_out_logits(args.folder, args.lang, args.cache)
        report(L, y, 1.0, "before")
        t = fit(L, y)
        report(L, y, t, "after ")
    apply(args.folder, t)


if __name__ == "__main__":
    main()
