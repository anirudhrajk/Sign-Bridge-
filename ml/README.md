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
| **Webcam**: ASL Citizen, 11 Deaf/hard-of-hearing signers, 2,089 videos | **77.5%** | **90.8%** | 73% of signs, 92.2% of them right |
| **Phone**: PopSign, 3 held-out signers, 1,316 videos (tested pair) | 90.5% | 97.7% | |
| Kaggle/Google ISLR, 4 held-out signers, 16,299 recordings (tested pair) | 69.6% | | |

Progress on the webcam test: old v1 model not measured, first v2 pair 73.0%, then 75.8%, 76.1% with
MediaPipe's full tracker, which the app now uses (a device under 10 tracked fps drops to lite by itself),
and 77.5% with three networks and 40 signers.
On the Kaggle test the old v1 model scored 55.8% and the first v2 pair 64.2%.
The browser pipeline was checked against Python on 199 of the webcam videos: 75.9% vs 76.5%.
Every webcam signer scores 73-82%. A slow phone (10 fps) loses about 3.5 points.

### Clinic words

`ml/clinic_words.py` lists words a patient is likely to sign (82 of ASL's 250, 85 of ISL's 264), picked
by meaning. The sign page counts each one 8 times more likely than other words. This is a boost, not a
hard list: another word still wins when the sign is clearly that word, and the patient can switch it
off. The boost was chosen on held-out Kaggle and phone signers. A higher boost stops the app adding
other words by itself: on Kaggle, 56% of them at 8 but only 32% at 16.

| ASL, 11 unseen webcam signers | No boost | Boost 8 |
|---|---|---|
| Clinic words (706 videos, 58 of the 82 words), right first time | 75.2% | **80.6%** |
| Clinic words, in the top 4 | 89.2% | **92.5%** |
| Clinic words added by the app itself | 71%, 89.8% right | **82%, 91.3% right** |
| Other words (1,383 videos), right first time | 77.0% | 72.6% |
| Other words added by the app itself | 72%, 93.3% right | 61%, 90.4% right |

Teach a sign on held-out signers: recognised 67.5% to 71.5%, false fires 2.5% to 3.6%.
ISL showed almost no change (98.2% to 98.6%), but that test does not use new signers.

What the shipped ASL model is:
- **Data**: Kaggle/Google ISLR (94,477 recordings, 21 signers) plus PopSign ASL v1.0 (CC BY 4.0):
  8 test-set signers and 11 training signers not in the Kaggle data, found by testing. 113,372 recordings
  from about 40 signers.
- **Recipe** (from the top Kaggle solutions, each kept only because it helped on unseen signers):
  hand-shape distance features (`--geo`), the face swapped in from another recording of the same
  sign (`--face-swap`), manifold mixup, label smoothing 0.3, weight averaging (EMA), plus the
  webcam, mirror, camera and tracker augmentations.
- **Three networks** (192, 256 and a second 192 wide) averaged in one file (`ml/ensemble_v2.py`).
- **Calibrated** (`ml/calibrate.py`, temperature 0.834 fitted on held-out signers), so "70% sure"
  means what it says.
- **Full precision** (32-bit weights, 27.4 MB). `ml/shrink_onnx.py` can store them at half size
  (10.2 MB, same answers in tests) but the shipped files are kept full size.

- **Teach a sign** uses `teach_bias` 17 for ASL (`model.json`): on held-out signers, 3 examples each, taught signs
  are recognised 67.5% of the time with 2.5% false fires (56.2% and 0.6% at the old setting).

Tried and not used (measured, not guessed): AWP (Kaggle +1.3, phone +0.6, webcam -0.4, at twice the training
time), averaging mirrored/re-timed copies at test time
(+0.1 points), 48 frames instead of 32 (no gain), extra smoothing of the points (-1.4 points), the
newer MediaPipe Tasks tracker (same speed on a laptop: 51 vs 55 ms per frame).

Camera size and tracker on unseen signers: full tracker 76.1% vs lite 75.8% on webcams; on 400 phone
videos 89.5-90.5% across 640 to 1920 px, lite or full, all within noise. Tracking takes 25-28 ms a frame
on the lite tracker and 32-35 ms on the full one at any of those sizes, on a laptop CPU.

### ISL

| Data | Signs | Top-1 | Top-5 | Test set |
|---|---|---|---|---|
| INCLUDE, 4,284 recordings | 262 | 97.7% | | random 20% (signers may repeat); new model, was 97.8% on the same rows |
| Fever and injury, 1 new person | 2 | 20 of 20 | | only 5 real clips per word in each of 2 forms, the other 40 are tilted copies; not deaf signers |

Every simulated webcam condition is at least 95.8%. The ASL recipe scored 97.1% here, so ISL keeps
its model (7.1 MB, full precision).

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
cp data/models_v2/asl_final2/model.onnx data/models_v2/asl_final2/model.json static/models/asl/
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

### Fever and injury

From the Indian Sign Language Video Dataset (Kaggle, prasadshet, marked MIT): 60 clips per word from
four contributors, 640x720, many of them tilted copies. Hearing contributors copied the signs from the
government ISL channel, so a deaf signer should check them. Other words from the same people are mixed in
so the model cannot learn the camera instead of the sign.

```bash
data/venv_mp/Scripts/python ml/holistic_extract.py data/isl_medical/all --parts 6 --complexity 1
python ml/prepare_v2.py --lang isl --extra data/isl_medical/all --only k_ --new-words fever,injury   --x-scale 0.5 --rest-copy --hold 20230926,20230927 --name isl_plus
python ml/train_v2.py --lang isl --cache isl_plus --epochs 200 --out data/models_v2/isl_m1
python ml/train_v2.py --lang isl --cache isl_plus --epochs 200 --all-data   --reported data/models_v2/isl_m1/model.json --out data/models_v2/isl_m1_all
data/venv_mp/Scripts/python ml/add_sign_clip.py track video.mp4 fever.npz --x-scale 0.5
python ml/add_sign_clip.py add fever.npz --lang isl --name fever --label fever
```

On 52 words ISL does not know, it was sure enough to add fever or injury on 2-3% of clips.
The ISL emergency-word videos (Mendeley 2vfdm42337) were tried and dropped. They were filmed behind a
board with no face or body, and the tracker found a hand in only 1-18% of frames.

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
- The 250 ASL signs come from a children's vocabulary (PopSign); ISL has 264 signs. Most medical words are
  still missing from both. The app's symptom screen covers them by tapping until there is data.
  Neither has many medical words. Recording medical signs is the most valuable next step.
- The models read one isolated word at a time, not continuous signing or grammar.
