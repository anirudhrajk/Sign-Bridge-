"""Join several trained v2 models into one ONNX file that averages their answers.

  python ml/ensemble_v2.py --lang asl --members data/models_v2/asl_A data/models_v2/asl_B --out data/models_v2/asl_AB
  python ml/ensemble_v2.py --lang asl --members data/models_v2/asl_A_all data/models_v2/asl_B_all \
      --reported data/models_v2/asl_AB/model.json --out data/models_v2/asl_final

The file has the same inputs and outputs as a single model (points, mask -> logits, embedding),
so the browser code does not change: logits are log(mean of the members' probabilities) and the
embedding is the members' unit embeddings joined and rescaled to unit length.
Without --reported the ensemble is scored on the held-out signers; with it (members trained on all
data) the scores of the tested ensemble are kept.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.train import split  # noqa: E402
from ml.train_v2 import P, T, T_IN, Exported, SignNetV2  # noqa: E402


def load_member(folder):
    ck = torch.load(Path(folder) / "model.pt", map_location="cpu")
    a = ck["args"]
    net = SignNetV2(len(ck["classes"]), d=a["dim"], repeats=a["repeats"],
                    drop=a.get("drop", 0.2), head_drop=a.get("head_drop", 0.4))
    net.load_state_dict(ck["state"])
    return net.eval(), ck


class Ensemble(nn.Module):
    def __init__(self, nets):
        super().__init__()
        self.members = nn.ModuleList(Exported(n) for n in nets)

    def forward(self, x, m):
        outs = [e(x, m) for e in self.members]
        prob = torch.stack([o[0].softmax(-1) for o in outs]).mean(0)
        emb = torch.cat([o[1] for o in outs], -1) / len(outs) ** 0.5
        return prob.clamp(min=1e-9).log(), emb


def load_exported(folder):
    """The module the browser runs, for a single model or an ensemble folder."""
    ck = torch.load(Path(folder) / "model.pt", map_location="cpu")
    if "members" in ck:
        return Ensemble([load_member(ROOT / f)[0] for f in ck["members"]]).eval()
    return Exported(load_member(folder)[0]).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="asl")
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reported", default="")
    args = ap.parse_args()
    nets, cks = zip(*(load_member(f) for f in args.members))
    if any(c["classes"] != cks[0]["classes"] for c in cks):
        sys.exit("members were trained on different sign lists")
    ens = Ensemble(nets).eval()
    first = json.loads((Path(args.members[0]) / "model.json").read_text())

    if args.reported:
        metrics = {**json.loads(Path(args.reported).read_text())["metrics"], "trained_on_all_data": True}
    else:
        d = np.load(ROOT / "data" / "cache_v2" / f"{args.lang}.npz", allow_pickle=True)
        labels, signers = d["labels"], d["signers"]
        _, va = split(labels, 0.2, 7, signers if (signers >= 0).all() else None)
        idx = np.floor(np.linspace(0, T_IN - 1, T) + 0.5).astype(int)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ens.to(dev)
        t1 = t5 = 0
        with torch.no_grad():
            for i in range(0, len(va), 512):
                s = d["X"][va[i:i + 512]][:, idx].astype(np.float32)
                mk = np.isfinite(s[..., 0])
                x = torch.as_tensor(np.nan_to_num(s) * mk[..., None], device=dev)
                lp = ens(x, torch.as_tensor(mk, dtype=torch.float32, device=dev))[0]
                top = lp.topk(5, -1).indices.cpu().numpy()
                yy = labels[va[i:i + 512]]
                t1 += (top[:, 0] == yy).sum()
                t5 += (top == yy[:, None]).any(1).sum()
        ens.cpu()
        metrics = {"val_top1": round(t1 / len(va), 4), "val_top5": round(t5 / len(va), 4), "val_samples": len(va),
                   "train_samples": first["metrics"]["train_samples"],
                   "val_new_signers": bool((signers >= 0).all())}
    metrics["ensemble_of"] = len(nets)
    print("metrics", metrics)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"members": [str(Path(f).resolve().relative_to(ROOT)) for f in args.members],
                "classes": cks[0]["classes"]}, out / "model.pt")
    dummy = (torch.zeros(1, T, P, 2), torch.ones(1, T, P))
    torch.onnx.export(ens, dummy, out / "model.onnx", input_names=["points", "mask"],
                      output_names=["logits", "embedding"], opset_version=17, dynamo=False)
    (out / "model.json").write_text(json.dumps({**first, "version": time.strftime("%Y%m%d-%H%M"), "metrics": metrics}))
    print("exported", out / "model.onnx", f"{(out / 'model.onnx').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
