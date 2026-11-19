# Project Context for Claude Code

This file gives you (Claude Code) everything you need to assist with this project.
Read it fully before making any edits or generating code.

---

## 1. Project Overview

**Project name:** Tactical Gesture Recognition System
**Repository slug:** `tactical_gestures`
**Author:** Anastasiia Rynkevych, 4th-year student, BSc in Applied Mathematics
**Institution:** Igor Sikorsky Kyiv Polytechnic Institute, Faculty of Applied Mathematics
**Degree thesis:** *"Система розпізнавання жестових сигналів"*
**Thesis defense deadline:** 11 June 2026

### One-line description

A Python pipeline that recognizes Ukrainian Armed Forces tactical hand-signal
gestures from a live RGB video stream and converts them into audible voice
commands in real time, designed to run autonomously without internet access.

### Why it exists

In modern combat conditions, voice commands compromise position and radio is
vulnerable to electronic warfare (EW). Tactical hand signals are reliable but
require constant visual contact between the sender and receiver, forcing the
receiver to look away from their sector of responsibility. This system removes
that constraint by converting detected gestures into audio commands.

### Success criteria (from thesis Section 1)

The system must:
- process a frame sequence with model inference latency ≤ 1 second;
- classify gestures with high Accuracy and macro-F1 score;
- include a decision-logic module that filters spurious activations using a
  confidence threshold;
- map each recognized class deterministically to a pre-recorded audio file.

### What this project is NOT

- Not a research-grade SOTA system. It is an undergraduate thesis prototype.
- Not field-tested. Dataset is collected in a home environment; absence of
  camouflage, tactical gloves, and chest-camera viewpoints is a known and
  honestly documented limitation.
- Not an online service. Operation must be **fully offline**: no cloud TTS,
  no remote inference endpoints. Internet is used only during initial model
  download from HuggingFace.

---

## 2. Mathematical Specification (CRITICAL: do not change these values)

These values are fixed in the thesis Section 3 ("Математичне забезпечення").
The code MUST be consistent with them. If you think a value should change,
flag it but do not silently override.

### Input tensor

- Frame format: RGB
- Camera native frame rate: 30 FPS
- Clip length fed to model: **T = 8 frames**
- Frames are selected by **uniform (sparse) sampling** from a **2-second
  temporal window**. Step between sampled frames ≈ 7–8 raw frames.
- Spatial resolution: **H = W = 224 pixels**
- Channels: **C = 3** (RGB)
- Input tensor shape: `(B, T, C, H, W) = (B, 8, 3, 224, 224)` for PyTorch
  convention. HuggingFace `TimesformerForVideoClassification` uses
  `(B, T, C, H, W)` after `image_processor`.

### Model

- Architecture: **TimeSformer-Base** with **Divided Space-Time Attention**
- Patch size: **P = 16** → N = 196 patches per frame
- Sequence length to transformer: M = N·T + 1 = **1569 tokens** (incl. CLS)
- Latent dim: **d = 768**
- Encoder depth: **L = 12 blocks**
- Attention heads: **h = 12**
- MLP hidden dim: **d_mlp = 3072**
- Total parameters: ≈ 121·10⁶
- Pre-trained weights: **Kinetics-400** (`facebook/timesformer-base-finetuned-k400`
  on HuggingFace)
- Fine-tuning strategy: freeze first **L - L' = 8** blocks, train last
  **L' = 4** blocks + classification head
- Number of gesture classes: **K = 10**

### Classes (in fixed order; index 0..9)

The 10 classes were selected from the Ukrainian Armed Forces tactical-signal
training video series **"Пантерівець"** (YouTube channel covering small-group
tactical preparation, ZSU SOF context). Selection criteria:

1. **Camera visibility**: only large-arm gestures (forearm rotation, full
   arm movement) are included. Fine-grained finger-only signals (numbers,
   "quiet" finger-to-lips, "watch" two-fingers-to-eyes) were excluded
   because at a 5-10 m chest-camera viewing distance they fall below the
   recognizable pixel threshold.
2. **Inter-class contrast**: chosen so that pairs share as few visual
   features as possible (mix of one-handed vs two-handed, different
   forearm orientations, static vs dynamic).
3. **Tactical relevance**: balance between movement commands (4), tactical
   maneuvers (3), and threat warnings (3).

```python
GESTURES = [
    "danger",      # увага / бойова зупинка (кулак вертикально вгору)
    "forward",     # уперед (передпліччя горизонтально, долоня вперед)
    "come",        # до мене (stop-поза + рух кисті до себе)
    "down",        # лягти / присісти (долоня вниз, рух вгору-вниз)
    "attack",      # атакувати (кулак горизонтально, рух від/до себе)
    "encircle",    # оточити (дві руки в круговій траєкторії)
    "disperse",    # розсипатися (кулак→долоня на рівні живота)
    "position",    # позиція противника (дві руки на грудях, пальці вниз,
                   # розводяться у різні боки)
    "mine",        # міна (опущена рука зі зігнутою кистю "лапкою")
    "ambush",      # засідка (рука через себе за спину/шию)
]
```

### Per-class geometry (for dataset recording + audio mapping)

These descriptions are the authoritative ground truth for all dataset
contributors. The PDF recording guide (`docs/RECORDING_GUIDE.pdf`,
to be generated) renders them with timecode references to the source video.

| idx | code     | UA command          | timecode | geometry                                                               |
|-----|----------|---------------------|----------|------------------------------------------------------------------------|
| 0   | danger   | "Увага"             | 2:00     | One arm. Bent elbow, forearm vertical, **fist** at head level.         |
| 1   | forward  | "Уперед"            | 1:36     | One arm. Bent elbow, forearm horizontal, open palm pointing forward.   |
| 2   | come     | "До мене"           | 2:28     | One arm. Bent elbow, forearm vertical-ish, open palm, **beckoning**    |
|     |          |                     |          | motion of fingers toward the body. Dynamic.                            |
| 3   | down     | "Лягти"             | 2:12     | One arm. Bent elbow, forearm horizontal, **palm down**, vertical       |
|     |          |                     |          | up-down motion. Dynamic.                                               |
| 4   | attack   | "Атакувати"         | 2:40     | One arm. **Fist** horizontal at chest level, in-out motion             |
|     |          |                     |          | (away/toward body). Dynamic.                                           |
| 5   | encircle | "Оточити"           | 2:43     | **Two arms.** Hands start near body, sweep outward in a **circular**   |
|     |          |                     |          | trajectory. Wide-amplitude dynamic gesture.                            |
| 6   | disperse | "Розсипатися"       | 2:52     | One arm. Held at **waist level**, starts as fist, opens into a palm.   |
|     |          |                     |          | Unique low-position gesture.                                           |
| 7   | position | "Позиція противника"| 5:24     | **Two arms.** Hands at chest level, **forefingers pointing down**,     |
|     |          |                     |          | start together, **spread apart** sideways. Dynamic.                    |
| 8   | mine     | "Міна"              | 5:18     | One arm. Extended forward-and-down, **fingers curled** ("claw shape"). |
|     |          |                     |          | Static. Unique low-forward position.                                   |
| 9   | ambush   | "Засідка"           | 5:21     | One arm. **Reaches across the body**, hand goes behind the neck or     |
|     |          |                     |          | over the shoulder. Static "crossed-arm" silhouette. Most unique pose.  |

### Known difficult pairs (expected confusion in matrix)

The model will most likely confuse:

- **come ↔ forward** — same general arm pose (extended forward). Distinguished
  only by the beckoning vs. pointing motion of the hand. Honest disclosure
  in thesis §4.4 expected.
- **attack ↔ danger** — both use a fist. Distinguished by forearm orientation
  (horizontal vs. vertical) and by motion (in-out vs. static).

All other pairs have high visual contrast and should not confuse.

### Source citation

The class list and per-gesture geometry are derived from the YouTube channel
**Пантерівець** (https://www.youtube.com/@panterivets), a publicly available
tactical-training resource by/for Ukrainian Armed Forces personnel. The
reference video used is https://www.youtube.com/watch?v=JH7e6mztaeQ
(timecodes in the table above).

Thesis §4.4.1 cites this source explicitly. No proprietary or classified
material is used.

### Loss and optimization

- Loss: **Cross-entropy with label smoothing**, ε = 0.1
- Optimizer: **AdamW** with weight decay
- Discriminative learning rates: higher LR for classification head,
  lower LR for unfrozen encoder blocks (use ratio ~10:1, e.g.
  head_lr=3e-4, encoder_lr=3e-5)
- Mixed precision (AMP) when CUDA is available

### Temporal post-processing (formulas 3.15–3.19 in thesis)

Streaming mode applies the following on top of per-clip predictions:

1. **Sliding window**: clip at frame n is `[F_{n-T+1}, …, F_n]` with
   uniform sampling inside; window step **s = 15 frames** (= 0.5 s at 30 FPS).
2. **Exponential smoothing** of probability vector:
   `p_smoothed_n = α·p_n + (1−α)·p_smoothed_{n-s}`,
   with **α = 0.3**.
3. **Confidence threshold**: trigger only if `max_c p_smoothed_n > θ_conf`,
   with **θ_conf = 0.7**.
4. **Cooldown (dead zone)**: minimum **τ_cd = 2 seconds** between two
   consecutive triggers.

These four parameters MUST be exposed in `config.py` as named constants and
referenced symbolically in code (do not hardcode literals in business logic).

---

## 3. Technology Stack

**Hard constraints:**

- Python **3.11.1** (installed; do not require a newer Python).
- All runtime dependencies must work **offline** after initial install.
- Must run on Windows 10/11 with an integrated webcam.
- Local hardware: laptop with 8 GB RAM, CPU-only (no local CUDA).
- Training is done on Google Colab (free Tesla T4 GPU).

**Required libraries** (specify versions in `requirements.txt`):

| Library | Purpose | Notes |
|---|---|---|
| `torch` | tensors, NN | install CPU-only wheel locally; CUDA wheel on Colab |
| `torchvision` | transforms | matches torch version |
| `transformers` | TimeSformer loader | HuggingFace, version ≥ 4.40 |
| `accelerate` | training utilities | optional, used by transformers |
| `opencv-python` | webcam capture, frame resize | use `cv2.CAP_DSHOW` on Windows |
| `numpy` | arrays | pinned to a version compatible with torch |
| `Pillow` | image handling | required by torchvision |
| `decord` | fast video reading for dataset | preferred over OpenCV for clips |
| `PyQt6` | desktop GUI | not PyQt5, not PySide |
| `simpleaudio` | offline WAV playback | non-blocking |
| `pyttsx3` | offline TTS (one-time gen) | only for `tools/generate_audio.py` |
| `scikit-learn` | metrics, confusion matrix | for evaluation only |
| `matplotlib` | plots for thesis | static figures |
| `seaborn` | confusion matrix heatmap | thin wrapper over matplotlib |
| `tqdm` | progress bars | training loop |
| `pyyaml` | config files | reading `configs/*.yaml` |

**Do NOT add:**

- `gTTS` or any cloud-TTS — internet dependency conflicts with field-use
  requirement.
- `tensorflow`, `keras`, `jax` — single-framework rule (PyTorch only).
- `wandb`, `mlflow` — tracking can be done by local CSV/JSON to keep
  the project self-contained.
- `flask`, `fastapi`, `streamlit` — no web layer needed.

---

## 4. Project Structure

Create and maintain the following layout. Do not invent new top-level
directories without a written reason in your response.

```
tactical_gestures/
├── CLAUDE.md                    # this file
├── README.md                    # user-facing instructions (Ukrainian)
├── ROADMAP.md                   # phased plan, owned by Anastasiia
├── requirements.txt             # locked dependency versions
├── requirements-colab.txt       # Colab-specific overrides (CUDA torch)
├── .gitignore
│
├── config.py                    # ALL hyperparameters; single source of truth
│
├── src/
│   ├── __init__.py
│   ├── video_buffer.py          # ring buffer + uniform sampling
│   ├── preprocessing.py         # resize / normalize / to tensor
│   ├── model.py                 # TimeSformer wrapper, load/save, freeze logic
│   ├── postprocessing.py        # EMA, threshold, cooldown (formulas 3.15-3.19)
│   ├── audio.py                 # non-blocking WAV playback
│   ├── inference.py             # end-to-end streaming inference loop
│   ├── dataset.py               # PyTorch Dataset for tactical gestures
│   ├── train.py                 # training script (callable from CLI and Colab)
│   ├── evaluate.py              # metrics + confusion matrix + plots
│   └── utils.py                 # small helpers (timing, logging)
│
├── gui/
│   ├── __init__.py
│   ├── main_window.py           # PyQt6 main window
│   ├── inference_worker.py      # QThread that wraps src/inference.py
│   ├── widgets.py               # custom widgets (gesture indicator, log panel)
│   └── resources/               # icons, .qss styles
│
├── tools/
│   ├── generate_audio.py        # one-time TTS → WAV generator
│   ├── record_dataset.py        # webcam recorder for collecting clips
│   ├── verify_dataset.py        # checks lengths, FPS, class balance
│   └── profile_inference.py     # latency benchmark (CPU vs GPU)
│
├── notebooks/
│   └── colab_train.ipynb        # Colab notebook for training on T4
│
├── assets/
│   └── audio/                   # 10 pre-recorded WAVs, named 00_stop.wav etc.
│
├── data/                        # raw dataset (NOT committed to git)
│   ├── stop/
│   ├── forward/
│   └── ...                      # one folder per class
│
├── checkpoints/                 # trained model weights (NOT committed)
│   └── .gitkeep
│
└── docs/
    ├── DATA_COLLECTION.md       # instructions for recording the dataset
    └── FIGURES.md               # list of figures needed for thesis Section 4
```

---

## 5. Coding Standards

**General:**
- Strictly **PEP 8**. Run `ruff check` mentally before finalizing.
- Maximum line length: **100 characters** (not 79 — modern convention).
- All public functions and classes must have **docstrings**.
- All functions handling tensors must include the **expected tensor shape**
  in the docstring, e.g. `Args: x: tensor of shape (B, T, C, H, W)`.
- Use **type hints everywhere**. Including return types. Use `from __future__
  import annotations` at the top of modules with forward references.

**Docstring style:**
- Use **Google style** docstrings (Args / Returns / Raises sections).
- Docstrings in **English**.
- Inline `# comments` can be in **Ukrainian or English** — pick whichever
  is clearer for that specific line. Be consistent within a single file.

**Imports:**
- Group: stdlib → third-party → local (`src.*`, `gui.*`).
- Separate groups with one blank line.

**PyTorch specifics:**
- Always honor a `device` argument; default to `torch.device("cuda" if
  torch.cuda.is_available() else "cpu")`.
- Use `model.eval()` and `torch.inference_mode()` for inference paths.
- Use `torch.cuda.amp.autocast()` only inside training when CUDA is available;
  CPU-only paths must not use AMP.
- Validate tensor shapes before passing to the model. Raise `ValueError`
  with the actual vs. expected shape on mismatch.

**Error handling:**
- Use `try/except` around: webcam open, model load, file I/O, audio playback.
- Log errors with the `logging` module (not `print`), level `ERROR` for
  failures, `WARNING` for recoverable issues, `INFO` for milestones.
- Never silently swallow exceptions.

**Configuration:**
- All numeric constants from Section 2 of this file MUST live in `config.py`
  as module-level UPPER_SNAKE_CASE names. Reference them, do not duplicate.
- Example:
  ```python
  # config.py
  CLIP_LENGTH = 8           # T
  PATCH_SIZE = 16           # P
  IMAGE_SIZE = 224          # H = W
  NUM_CLASSES = 10          # K
  WINDOW_STEP_FRAMES = 15   # s
  EMA_ALPHA = 0.3           # α
  CONFIDENCE_THRESHOLD = 0.7  # θ_conf
  COOLDOWN_SECONDS = 2.0    # τ_cd
  LABEL_SMOOTHING = 0.1     # ε
  ```

---

## 6. UI/UX Language

- **Code, comments, logs, docstrings, commit messages, variable names: English.**
- **User-visible UI strings in PyQt6 GUI: Ukrainian.**
  - Use a single dict `UI_STRINGS` in `gui/main_window.py` so they are easy
    to find and review.
- **README.md, ROADMAP.md, docs/DATA_COLLECTION.md: Ukrainian.**
  - These are read by Anastasiia and possibly her advisor.
- **CLAUDE.md (this file): English.**

---

## 7. Workflow Conventions

**Branching / commits:**
- Single-developer project. Use `main` branch directly.
- Commit messages in **English**, imperative mood, ≤ 72 chars first line.
  Example: `feat(buffer): add uniform sampling with stride 7`

**When you (Claude Code) propose changes:**
1. State the file you will modify and the rationale in one sentence.
2. If modifying behavior covered by the thesis Section 3, cite the formula
   number (e.g. "implements 3.16 with α=0.3").
3. Run mental tests on tensor shapes and edge cases before writing.
4. If a dependency conflict arises (e.g. `decord` not building on Windows),
   propose a fallback (e.g. fall back to OpenCV) instead of forcing it.

**Things to ALWAYS verify before claiming "done":**
- Imports actually resolve.
- Function returns the type its annotation promises.
- No hardcoded paths (use `pathlib.Path` and `__file__` resolution).
- New constants added to `config.py` if they are reused.
- `requirements.txt` updated if a new package is introduced.

**Things to NEVER do:**
- Do not invent training results or fabricate metrics. If asked for numbers
  before training has run, state "training not yet performed; placeholder."
- Do not use libraries listed in "Do NOT add".
- Do not change values listed in Section 2 of this file without first
  flagging the inconsistency with the thesis.
- Do not push any code to GitHub. Anastasiia handles git pushes manually.
- Do not write a 500-line file in one shot. Break work into reviewable units
  of ~100-200 lines at a time.

---

## 8. Thesis Document Awareness

The full thesis draft is in the same directory (or provided separately) as
**`БКР_Ринкевич.docx`** (Ukrainian).

**Already written:** Sections 1 (Postanovka), 2 (Analysis), 3 (Math).
**To be written by the chat-Claude (not you):** Section 4 (Software).
**Your job:** generate code consistent with Sections 1-3 so that Section 4
can describe a real, working implementation.

When Anastasiia asks you to implement something, assume she also wants the
code to be **documentable in Section 4**. That means:
- Module boundaries should match the modules described in this CLAUDE.md.
- Public APIs should be small enough to summarize in a paragraph.
- Print statements during execution should produce information suitable for
  screenshots (in Ukrainian, formatted, with timestamps).

---

## 9. Phases of the Project

We are currently in **Phase A**.

- **Phase A (now):** Implement all modules. Generate audio files. Verify
  that the pipeline runs end-to-end on a synthetic 8-frame zeros tensor and
  on a real video file (no training yet).
- **Phase B (after dataset is recorded):** Run training on Colab. Collect
  metrics, confusion matrix, training curves.
- **Phase C (after metrics are in):** Polish GUI, take screenshots, finalize
  thesis Section 4. Re-write thesis Conclusions and Abstract under the
  current topic (current draft has leftover ANFIS-related text from a
  template — to be removed).

Do not jump ahead of the current phase unless explicitly asked.

---

## 10. How to Talk to Anastasiia

- She is a final-year applied math student. She knows the math well
  (TimeSformer, transformers, optimization) but is **newer to software
  engineering practice** (virtual environments, dependency pinning, git
  branching). Explain such things when relevant, do not assume them.
- When something is ambiguous, **ask one targeted question**, do not list
  five at once.
- When proposing a non-trivial change, explain **what** and **why** in
  Ukrainian if she is reading the response, or in English if it is in code
  comments.
- Default tone: concise, technical, no filler.

---

## 11. Quick Sanity Checklist (Run Before Any Major Edit)

- [ ] Are tensor shapes in your code consistent with Section 2 of this file?
- [ ] Are α, θ_conf, τ_cd, s, T, P, K, ε taken from `config.py`?
- [ ] Does your change preserve offline operation?
- [ ] Did you update `requirements.txt` if you added an import?
- [ ] Did you update docstrings if you changed a signature?
- [ ] Did you keep file size under ~300 lines?

---

End of CLAUDE.md.
