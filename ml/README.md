# Sign recognition model

SignBridge reads signs **on the patient's device**. MediaPipe Holistic tracks the body, hands
and face in the browser, and a neural network turns about two seconds of movement into a word.
The network runs with ONNX Runtime Web (`static/js/sign-model-v2.js`). No video is uploaded,
and it keeps working offline.

## Current models (v2)

127 points per frame: 9 body points, 21 per hand, 40 lip points, 16 per eye and 4 nose points
(`ml/landmarks_v2.py`). The network computes its own features from the raw points and a presence
mask, so the browser only gathers points. Architecture: depthwise 1-D convolutions plus a
transformer, repeated twice (`ml/train_v2.py`). Training runs on the GPU, augmentation included:
webcam zoom with hands leaving the frame, a left-handed mirror, camera tilt, scale and shear,
speed changes and cropping, tracker dropouts, and the face hidden.

| Language | Recordings | Signs | Top-1 | Top-5 | Test set |
|---|---|---|---|---|---|
| ISL (INCLUDE) | 4,284 | 262 | 98.1% | 99.2% | random 20% (signers may repeat) |
| ASL (Google ISLR) | 94,477 | 250 | 64.2% | 82.3% | 4 signers never seen in training |

The ASL model averages two networks (192 and 256 wide, `ml/ensemble_v2.py`) in one ONNX file.
On the same 4 held-out signers the old v1 ASL model scored 55.8%. The sign page only adds a
word when the model is at least 70% sure. For ASL that happens for 59% of signs, and those words
are right 85% of the time; for the other signs the page shows its top guesses and asks. The best Kaggle score on this data (89.3%) used a
different test split and much larger ensembles, so the numbers aren't directly comparable.

Stress test (`python ml/check_v2.py --lang asl --stress`), ASL top-1 on held-out signers:
clean 65.7%, noisy tracking 64.2%, 10 fps device 64.2%, left-handed 65.6%, tilted camera 64.9%,
1.5x faster 65.3%, face not tracked 61.5%, all webcam effects 64.1%.
ISL: every condition is at least 95.8%.

### Retrain

```bash
python ml/prepare_v2.py --lang isl                      # data/keypoints/**.parquet -> data/cache_v2/isl.npz
python ml/train_v2.py --lang isl --epochs 200           # tested run (held-out split)
python ml/check_v2.py --lang isl --stress               # ONNX vs PyTorch + stress conditions
python ml/train_v2.py --lang isl --epochs 200 --all-data --reported data/models_v2/isl/model.json --out data/models_v2/isl_final
```

ASL (needs your own Kaggle account with the competition rules accepted, and a token in
`%USERPROFILE%\.kaggle\access_token`):

```bash
py -3.11 ml/download_asl.py --v2 --part 1/3             # run parts 1/3, 2/3, 3/3 in parallel
python ml/prepare_v2.py --lang asl                      # 2.3 GB cache
python ml/train_v2.py --lang asl --out data/models_v2/asl_A
python ml/train_v2.py --lang asl --dim 256 --drop 0.3 --head-drop 0.5 --out data/models_v2/asl_B
python ml/ensemble_v2.py --lang asl --members data/models_v2/asl_A data/models_v2/asl_B --out data/models_v2/asl_AB
python ml/check_v2.py --lang asl --dir data/models_v2/asl_AB --stress
```

Then copy `model.onnx` and `model.json` from the chosen folder into `static/models/<lang>/`.
To ship models trained on every recording, train each member with `--all-data --reported <tested
member>/model.json`, then join them with `ml/ensemble_v2.py ... --reported data/models_v2/asl_AB/model.json`.
The scores stay those of the tested run.

## Older models (v1)

`ml/prepare_data.py`, `ml/train.py` and `ml/check_export.py` build the 49-point v1 format
(`model.json` + `weights.bin`). `static/js/sign-model.js` still loads either format.

## Adding new signs (for example, medical words)

1. Record each new sign about 15 to 20 times, ideally with several different signers,
   lighting setups and cameras.
2. Extract MediaPipe Holistic landmarks into the same parquet layout as the existing files
   (`frame, row_id, type, landmark_index, x, y, z`) and save them under
   `data/keypoints/<category>/<word>/`.
3. Run the ISL commands above. Folder names become the class names, and `ml/labels.py`
   turns joined words into readable English.

Patients can also teach their own signs in the app (`docs/teach-a-sign.md`).

## Honest limits

- ISL has no signer IDs, so its test set probably includes people seen in training. Accuracy
  for new people will be lower than 98%.
- ASL at 64% means about 1 in 3 signs from a new person is misread or left unsure. The app
  asks for confirmation instead of guessing.
- The models read one isolated word at a time. They don't yet understand continuous signing
  or sign language grammar.
- Both datasets have very few medical words. Recording medical signs is the most valuable next step.
