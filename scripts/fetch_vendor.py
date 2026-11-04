"""Download the two large browser libraries during a deploy build (instead of storing them in git).

MediaPipe Holistic (hand and body tracking) and ONNX Runtime Web (runs the sign models) come
from the official npm registry; each archive is checked against the registry's sha512 hash.

  python scripts/fetch_vendor.py
"""
import base64
import hashlib
import io
import tarfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = [
    ("@mediapipe/holistic", "0.5.1675471629", ROOT / "static" / "vendor" / "holistic", "package/", [
        "holistic.js", "holistic.binarypb", "holistic_solution_packed_assets.data",
        "holistic_solution_packed_assets_loader.js", "holistic_solution_simd_wasm_bin.data",
        "holistic_solution_simd_wasm_bin.js", "holistic_solution_simd_wasm_bin.wasm",
        "holistic_solution_wasm_bin.js", "holistic_solution_wasm_bin.wasm", "pose_landmark_lite.tflite",
        "pose_landmark_full.tflite"]),
    ("onnxruntime-web", "1.30.0", ROOT / "static" / "vendor" / "onnxruntime", "package/dist/", [
        "ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]),
]


def get(url, timeout):
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))


def main():
    for name, version, dest, prefix, files in PACKAGES:
        if all((dest / f).exists() for f in files):
            print(f"{name}: already present")
            continue
        meta = get(f"https://registry.npmjs.org/{name}/{version}", 60).json()
        blob = get(meta["dist"]["tarball"], 600).content
        algo, digest = meta["dist"]["integrity"].split("-", 1)
        if base64.b64encode(hashlib.new(algo, blob).digest()).decode() != digest:
            raise SystemExit(f"{name}: integrity check failed")
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            for f in files:
                member = tar.getmember(prefix + f)
                (dest / f).write_bytes(tar.extractfile(member).read())
        print(f"{name} {version}: {len(files)} files, integrity ok")


if __name__ == "__main__":
    main()
