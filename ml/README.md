# Sign recognition model

SignBridge reads signs **on the patient's device**. MediaPipe Holistic tracks the body and
hands in the browser, and a small neural network (`static/js/sign-model.js`) turns about two
seconds of movement into a word. No video is uploaded, and it keeps working without a server
round trip.

## Retrain

```bash
python ml/prepare_data.py   # reads data/keypoints/**.parquet -> data/cache/landmarks.npz (about 15 s)
python ml/train.py          # trains and exports static/models/isl/ (about 5 min on a laptop CPU)
python ml/check_export.py   # re-checks the exported model's accuracy
```

Restart the app after retraining. The browser picks up the new `model.json` and `weights.bin`.

## American Sign Language

The ASL data is Google's "Isolated Sign Language Recognition" Kaggle competition: 94,477
recordings of 250 signs by 21 signers. You need your own Kaggle account with the competition
rules accepted, and a token saved in `%USERPROFILE%\.kaggle\access_token`.

```bash
py -3.11 ml/download_asl.py                  # streams the 40 GB archive, keeps about 230 MB
python ml/prepare_data.py --lang asl
python ml/train.py --lang asl --epochs 40 --eval-every 4 --width 192
python ml/build_sign_clips.py --lang asl     # lists the recordings the avatar needs
py -3.11 ml/download_asl.py --clips          # fetches only those 250 full recordings
python ml/build_sign_clips.py --lang asl     # builds static/signs/asl/
```

ASL accuracy is measured on 4 signers held out of training, so it reflects how the model does
with new people. ISL has no signer IDs, so its test set may include people seen in training.

## Adding new signs (for example, medical words)

1. Record each new sign about 15 to 20 times, ideally with several different signers,
   lighting setups and cameras.
2. Extract MediaPipe Holistic landmarks into the same parquet layout as the existing files
   (`frame, row_id, type, landmark_index, x, y, z`) and save them under
   `data/keypoints/<category>/<word>/`.
3. Run the three commands above. Folder names become the class names, and `ml/labels.py`
   turns joined words into readable English.

## How features work

`ml/features.py` and `sign-model.js` must stay identical. Each frame uses the shoulders,
elbows, wrists and nose, plus 21 points per hand. Positions are measured relative to the
shoulders, so where someone sits doesn't matter. Each hand's shape is also measured relative
to its own wrist.

## Honest limits

- The held-out test videos probably include the same signers as the training videos, so
  accuracy for new people will be lower than the reported figure.
- The model reads one isolated word at a time. It doesn't yet understand continuous signing
  or ISL grammar.
- The dataset has very few medical words. Recording medical signs is the most valuable next step.
