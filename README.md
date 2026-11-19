# Tactical Hand-Signal Recognition System

Bachelor's thesis, Igor Sikorsky Kyiv Polytechnic Institute, Faculty of Applied Mathematics, 2026.
Author: Anastasiia Rynkevych.

The system recognizes Ukrainian Armed Forces tactical hand signals from a live webcam stream
and converts them into voice commands in real time. It runs fully offline.

**Why:** voice commands give away a position, and radio is vulnerable to electronic warfare.
Hand signals are reliable, but the receiver has to keep looking at the sender instead of
watching their own sector. This system removes that constraint.

## Results

5-fold stratified cross-validation on 380 clips (304 train / 76 validation per fold):

| Metric | Value |
|---|---|
| Accuracy | **93.4% ± 2.5%** (95% CI: 90.4–96.5%) |
| Macro-F1 | **93.4%** |
| Inference time per clip (GPU T4) | **≈0.13 s** |
| Classes | 10 gestures |
| Dataset | 380 clips, recorded and labeled by me |

![5-fold CV](results/kfold/cv_summary.png)

![Confusion matrix](results/kfold/confusion_matrix_all_folds.png)

The hardest class is "Mine" (mean F1 0.85), most often confused with "Attack" and "Enemy position".
Full per-fold metrics and training curves are in [`results/kfold/`](results/kfold/).

Limitations: the data was recorded indoors, without camouflage, tactical gloves or a
chest-camera viewpoint. This is a prototype; field validation is the next step.

## How it works

```
webcam → ring buffer (2 s) → 8 uniformly sampled frames → TimeSformer → probabilities
       → EMA smoothing → confidence threshold → cooldown → voice command (WAV)
```

- **Model:** [TimeSformer-Base](https://huggingface.co/facebook/timesformer-base-finetuned-k400)
  pre-trained on Kinetics-400. The head is replaced with a 10-class one and only the last
  encoder blocks are fine-tuned (transfer learning).
- **Training:** AdamW with separate learning rates for head and encoder, cosine annealing,
  label smoothing, mixed precision, early stopping. Trained on Google Colab (T4).
- **Evaluation:** 5-fold stratified cross-validation, 95% CI via Student's t-distribution,
  per-class precision/recall/F1, confusion matrix.
- **Streaming post-processing:** EMA (α = 0.3), confidence threshold 0.7 and a 2 s cooldown,
  so one gesture doesn't trigger several times.
- **UI:** PyQt6 desktop app, inference runs on a separate thread.

Gesture classes come from the "Пантерівець" tactical training videos: danger, forward, come,
down, attack, encircle, disperse, enemy position, mine, ambush.

## Project structure

```
config.py              — all hyperparameters (single source of truth)
src/
  video_buffer.py      — thread-safe frame ring buffer
  preprocessing.py     — clip preprocessing for the model
  model.py             — TimeSformer wrapper, layer freezing
  postprocessing.py    — EMA, threshold, cooldown
  audio.py             — non-blocking audio playback
  inference.py         — streaming inference (CLI: video file or webcam)
  dataset.py           — PyTorch Dataset with augmentation
  train.py             — training
  train_kfold.py       — 5-fold cross-validation
  evaluate.py          — metrics and plots
gui/                   — PyQt6 desktop interface
tools/
  record_dataset.py    — webcam dataset recorder
  verify_dataset.py    — dataset integrity check
  profile_inference.py — latency benchmark (CPU / GPU)
  generate_audio.py    — offline TTS voice commands
notebooks/colab_train.ipynb — training on Colab T4
results/kfold/         — cross-validation results
CLAUDE.md              — project specification for Claude Code
ROADMAP.md             — phased work plan
```

## How I worked with AI

The code was written together with **Claude Code**. I defined the architecture, model choice,
hyperparameters and evaluation method myself; they are fixed in the math section of the thesis.

- **[CLAUDE.md](CLAUDE.md)** — specification for the agent: math parameters that must not
  change without approval, tech stack and banned libraries, project layout, coding standards,
  and rules like "never fabricate metrics".
- **[ROADMAP.md](ROADMAP.md)** — work in phases: implementation → data collection and
  training → thesis write-up.
- **computer-vision-engineer** — a CV engineer role description from GitHub, added so the
  agent follows computer vision best practices.
- I planned the pipeline in plan mode first, then gave the agent small tasks
  (~100–200 lines) and checked every module with smoke tests.

## Usage

```bash
pip install -r requirements.txt
python -m tools.generate_audio                              # generate 10 WAV commands
python -m gui.main --checkpoint checkpoints/best_model.pt   # GUI
python -m src.inference --video clip.mp4 --audio            # CLI
```

Training (on Colab with GPU):

```bash
pip install -r requirements-colab.txt
python -m src.train_kfold      # 5-fold cross-validation
python -m src.train            # final model
python -m src.evaluate         # metrics and plots
```

The dataset and model weights are not included: the dataset contains videos of people,
and the weights are several hundred MB.
