# Emotion Detector (Multimodal)

This repository contains a multimodal emotion detection demo (face + voice) and a lightweight, explainable lie-detection layer that fuses the emotion signals.

Short summary
- Use `main_multimodal.py` as the primary entrypoint (webcam or video). `main.py` is an earlier/demo script and is considered obsolete for the main multimodal flow.
- Face emotion detection uses `detectors/face_emotion.py` (FER + MTCNN preprocessing).
- Voice emotion detection uses `detectors/voice_emotion.py` (Hugging Face wav2vec2 / optional finetuned model).
- A small rule-based, explainable `detectors/lie_detector.py` computes a `lie_prob` and `lie_label` from face+voice+fused signals. It includes a stateful `LieDetector` with temporal smoothing.

Contents (high level)
- `main_multimodal.py` — primary app: webcam or video mode, fuses face & voice, logs results, overlays UI.
- `detectors/face_emotion.py` — face emotion detector wrapper (FER + preprocessing).
- `detectors/voice_emotion.py` — speech emotion recognition using Hugging Face models (inference + optional finetune helpers).
- `detectors/lie_detector.py` — explainable lie probability generator (stateless `evaluate_lie` + stateful `LieDetector`).
- `logger/emotion_logger.py` — logging helpers, smoothing, and AdaptiveBaseline.
- `data/` — local dataset directory (ignored by git). Large and kept out of repo.
- `pretrained_models/` — local models (ignored by git). Use this to store downloaded or finetuned models locally.

Quick setup (Windows / PowerShell)

1. Create a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- Recommended Python: 3.10 or 3.11 (ensure compatibility with the listed `torch` version).
- Installing `torch` may require platform-specific wheel commands — consult https://pytorch.org if you run into issues.

Run the app

- Run webcam multimodal (preferred):

```powershell
python main_multimodal.py --source webcam
```

- Run multimodal over a video file (one-shot voice prediction from full audio + per-frame face detection):

```powershell
python main_multimodal.py --source video --video_path path\to\your\video.mp4
```

Notes about `main.py` vs `main_multimodal.py`
- `main_multimodal.py` is the recommended entrypoint — it handles real-time fusion between face and voice, threading for mic inference, and the lie-detector overlay and logging.
- `main.py` is an older example focused on face-only demo flows. Prefer `main_multimodal.py` for full functionality.

Logging, outputs & artifacts
- Emotion/time-series logs are saved via `logger/emotion_logger.py` as CSV. The default path used by the app is `data/emotion_logs_multimodal.csv`.
- The lie detector adds `lie_prob` and `lie_label` to each logged row when run from `main_multimodal.py`.

Tests
- A tiny smoke test exists for the lie detector:

```powershell
python test_lie_detector.py
```

Configuration & tuning
- `detectors/lie_detector.py` exposes `LieDetectorConfig` with thresholds and weights. The app-level defaults are set in `main_multimodal.py` (variable `LIE_CONFIG`) so you can tune sensitivity there.
- You can also instantiate your own `LieDetector(cfg=...)` if you want different behavior per session.

Pretrained models & storage
- Large model files and audio datasets are intentionally ignored by `.gitignore` (see `pretrained_models/` and `data/`). If you want to keep some model artifacts in the repository, consider using Git LFS instead of storing binaries directly.

Pushing to GitHub (short reminder)
- Make sure `.gitignore` is present (it already is) before `git add .` so `data/` and models aren't tracked.
- Initialize git, commit, create a GitHub repository (web UI or `gh` CLI), add `origin`, and push (see the project root `README` instructions you followed earlier or the command list below).

Troubleshooting
- If the lie detector never reaches `lie` for your inputs:
  - Try tuning `LIE_CONFIG` in `main_multimodal.py` (lower `conflict_delta`, lower `lie_prob_cutoff`, increase `conflict_weight` or `temporal_boost`).
  - Inspect logs in `data/emotion_logs_multimodal.csv` which now include `lie_prob` & `lie_label` to debug decisions.
  - Ensure the voice model is loading correctly — if the voice scores are noisy/flat, the fusion will underperform.
- If `data/` or `pretrained_models/` were accidentally committed, remove them from tracking with:

```powershell
git rm -r --cached data
git rm -r --cached pretrained_models
git commit -m "Remove large data/model folders from tracking"
git push
```

Advanced notes
- To remove large files from history entirely, use BFG or `git filter-repo` (backup first). This is destructive and requires a forced push.
- If you want to keep large models in your remote but not blow up the repo, set up Git LFS:

```powershell
# install git-lfs (one-time)
git lfs install
git lfs track "pretrained_models/**"
git add .gitattributes
git commit -m "Track pretrained models with git-lfs"
```

Contact & credits
- This is a small research/demo project combining open-source tools: FER (face emotion), Hugging Face transformers (speech emotion), librosa, OpenCV, and a custom fusion layer.

If you want, I can also:
- Add a small `CONTRIBUTING.md` and `LICENSE` file.
- Create a minimal release with just code (no models) and attach instructions to download recommended pretrained weights.

Enjoy — keep your `data/` folder local and tune `LieDetectorConfig` to match your usage scenario.
