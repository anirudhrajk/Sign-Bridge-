# Sign recognition model

SignBridge reads signs **on the patient's device**. MediaPipe Holistic tracks the body, hands
and face in the browser, and a neural network turns about two seconds of movement into a word.
The network runs with ONNX Runtime Web in a background worker (`static/js/sign-model-v2.js`).
No video is uploaded, and it keeps working offline.

## Current models

127 points per frame: 9 body points, 21 per hand, 40 lip points, 16 per eye and 4 nose points
(`ml/landmarks_v2.py`). The network computes its own features from the raw points and a presence
mask, so the browser only gathers points. Architecture: depthwise 1-D convolutions plus a
transformer, repeated twice (`ml/train_v2.py`).

### ASL: how well it works on people it has never seen

| Test (none of these signers are in training) | Top-1 | Top-5 | Words the app adds by itself (>=70% sure) |
|---|---|---|---|
| **Webcam**: ASL Citizen, 11 Deaf/hard-of-hearing signers, 2,089 videos | **75.8%** | **90.3%** | 69% of signs, 91.5% of them right |
| **Phone**: PopSign, 3 held-out signers, 1,316 videos (tested recipe) | 90.7% | 97.2% | 86% of signs, 97.2% right |
| Kaggle/Google ISLR, 4 held-out signers (tested recipe, sample of 4,075) | 68.2% | | |

Progress on the webcam test: old v1 model not measured, first v2 pair 73.0%, now 75.8%.
On the Kaggle test the old v1 model scored 55.8% and the first v2 pair 64.2%.
The browser pipeline was checked against Python on 199 of the webcam videos: 75.9% vs 76.5%.
Every webcam signer scores 71-81%. A slow phone (10 fps) loses about 3.5 points.

What the shipped ASL model is:
- **Data**: Kaggle/Google ISLR (94,477 recordings, 21 signers) plus PopSign ASL v1.0
  (11,896 recordings from its 8 test-set signers, CC BY 4.0), 106,373 recordings from 29 signers.
- **Recipe** (from the top Kaggle solutions, each kept only because it helped on unseen signers):
  hand-shape distance features (`--geo`), the face swapped in from another recording of the same
  sign (`--face-swap`), manifold mixup, label smoothing 0.3, weight averaging (EMA), plus the
  webcam, mirror, camera and tracker augmentations.
- **Two networks** (192 and 256 wide) averaged in one file (`ml/ensemble_v2.py`).
- **Calibrated** (`ml/calibrate.py`, temperature 0.868 fitted on held-out signers), so "70% sure"
  means what it says.
- **Half-size weights** (`ml/shrink_onnx.py`): 10.2 MB instead of 20 MB, same answers.

Tried and not used (measured, not guessed): averaging mirrored/re-timed copies at test time
(+0.1 points), 48 frames instead of 32 (no gain), extra smoothing of the points (-1.4 points), the
newer MediaPipe Tasks tracker (same speed on a laptop: 51 vs 55 ms per frame).

### ISL

| Data | Signs | Top-1 | Top-5 | Test set |
|---|---|---|---|---|
| INCLUDE, 4,284 recordings | 262 | 98.1% | 99.2% | random 20% (signers may repeat) |

Every simulated webcam condition is at least 95.8%. The ASL recipe scored 97.1% here, so ISL keeps
its model (now stored at half size, 3.6 MB, same answers).

## Test sets

- `ml/asl_citizen.py` + `ml/eval_citizen.py`: ASL Citizen (Microsoft Research) is licensed for
  non-commercial research only and may not be shared, so it is used **only to measure** the
  models. Nothing from it is trained in or stored in the repo; delete `data/asl_citizen*` when done.
- `ml/popsign.py`: PopSign ASL v1.0 (Georgia Tech / NTID, CC BY 4.0), read video by video from the
  Kaggle uploads. Cite: Starner et al., PopSign ASL v1.0, NeurIPS 2023 Datasets and Benchmarks.
- `ml/holistic_extract.py`: videos -> 127 points with MediaPipe Holistic, the same settings as
  the app (needs `data/venv_mp` with `mediapipe==0.10.14 opencv-python-headless`).

## Retrain ASL

```bash
py -3.11 ml/download_asl.py --v2 --part 1/3                    # Kaggle data (run parts 1/3..3/3)
py -3.11 ml/popsign.py --set test --per-sign-signer 8           # PopSign videos
data/venv_mp/Scripts/python ml/holistic_extract.py data/popsign/test --parts 10 --step 2
python ml/prepare_v2.py --lang asl --extra data/popsign/test --hold 8031,8032,8033 --name asl_plus
R="--lang asl --cache asl_plus --geo --smooth 0.3 --mixup 0.5 --face-swap 0.25 --ema 0.999"
python ml/train_v2.py $R --out data/models_v2/asl_v5a                                   # tested runs
python ml/train_v2.py $R --dim 256 --drop 0.3 --head-drop 0.5 --out data/models_v2/asl_v5b
python ml/ensemble_v2.py --lang asl --cache asl_plus --members data/models_v2/asl_v5a data/models_v2/asl_v5b --out data/models_v2/asl_v5ab
python ml/calibrate.py data/models_v2/asl_v5ab --cache asl_plus                         # fits the temperature
python ml/train_v2.py $R --epochs 120 --all-data --reported data/models_v2/asl_v5a/model.json --out data/models_v2/asl_v5a_all
python ml/train_v2.py $R --epochs 120 --dim 256 --drop 0.3 --head-drop 0.5 --all-data --reported data/models_v2/asl_v5b/model.json --out data/models_v2/asl_v5b_all
python ml/ensemble_v2.py --lang asl --cache asl_plus --members data/models_v2/asl_v5a_all data/models_v2/asl_v5b_all --reported data/models_v2/asl_v5ab/model.json --out data/models_v2/asl_final2
python ml/calibrate.py data/models_v2/asl_final2 --temperature 0.868
python ml/shrink_onnx.py data/models_v2/asl_final2/model.onnx --out static/models/asl/model.onnx --check data/cache_v2/asl_plus.npz
python ml/eval_citizen.py static/models/asl                                              # webcam test
```

The final models train on every recording, so their scores are those of the tested runs (the
webcam test still has no overlap with them). `ml/check_v2.py --stress` simulates webcam conditions.

## Retrain ISL

```bash
python ml/prepare_v2.py --lang isl
python ml/train_v2.py --lang isl --epochs 200
python ml/check_v2.py --lang isl --stress
python ml/train_v2.py --lang isl --epochs 200 --all-data --reported data/models_v2/isl/model.json --out data/models_v2/isl_final
```

## Older models (v1)

`ml/prepare_data.py`, `ml/train.py` and `ml/check_export.py` build the 49-point v1 format
(`model.json` + `weights.bin`). `static/js/sign-model.js` still loads either format.

## Adding new signs (for example, medical words)

1. Record each new sign about 15 to 20 times, ideally with several different signers,
   lighting setups and cameras.
2. Extract the points with `ml/holistic_extract.py` (or MediaPipe Holistic into the parquet
   layout under `data/keypoints/<category>/<word>/`).
3. Retrain as above. Folder names become the class names, and `ml/labels.py` turns joined words
   into readable English.

Patients can also teach their own signs in the app (`docs/teach-a-sign.md`).

## Honest limits

- On a new person's webcam, about 1 sign in 4 is misread or left unsure. The app only adds a
  word by itself at 70% confidence (right about 9 times in 10) and otherwise asks.
- A pause in the middle of a sign can split it into two words (1 in 30 webcam signs in the page test).
- The 250 ASL signs come from a children's vocabulary (PopSign); ISL has 262 everyday signs.
  Neither has many medical words. Recording medical signs is the most valuable next step.
- The models read one isolated word at a time, not continuous signing or grammar.
