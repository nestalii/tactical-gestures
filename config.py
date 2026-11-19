"""Single source of truth for all hyperparameters.

Every numeric value here is fixed in thesis Section 3 ("Математичне
забезпечення") and must not be changed silently. If a value here needs to
move, the thesis text moves with it.
"""

from __future__ import annotations

from pathlib import Path

# --- Input tensor (thesis 3.1–3.4) ---
CLIP_LENGTH = 8                     # T: frames per clip fed to the model
IMAGE_SIZE = 224                    # H = W: spatial resolution
NUM_CHANNELS = 3                    # C: RGB
NATIVE_FPS = 30                     # camera frame rate
TEMPORAL_WINDOW_S = 2.0             # 2-second window for uniform sampling

# --- Model (thesis 3.5–3.12) ---
MODEL_NAME = "facebook/timesformer-base-finetuned-k400"
PATCH_SIZE = 16                     # P: patch size in pixels
NUM_CLASSES = 10                    # K: tactical gestures
ENCODER_DEPTH = 12                  # L: transformer blocks
UNFROZEN_BLOCKS = 3                 # L': last blocks to fine-tune
LATENT_DIM = 768                    # d
NUM_HEADS = 12                      # h
MLP_HIDDEN_DIM = 3072               # d_mlp

# --- Classes (CLAUDE.md §2 "Classes" — authoritative; fixed index order 0..9).
# Selected from the Ukrainian Armed Forces tactical-signal training video
# series "Пантерівець"; see CLAUDE.md for per-class geometry and timecodes.
GESTURES: list[str] = [
    "danger",       # увага / бойова зупинка
    "forward",      # уперед
    "come",         # до мене
    "down",         # лягти
    "attack",       # атакувати
    "encircle",     # оточити
    "disperse",     # розсипатися
    "position",     # позиція противника
    "mine",         # міна
    "ambush",       # засідка
]

# --- Training (thesis 3.13–3.14) ---
LABEL_SMOOTHING = 0.1               # ε
HEAD_LR = 3e-4                      # classification head LR
ENCODER_LR = 3e-5                   # unfrozen encoder LR (≈ head_lr / 10)
WEIGHT_DECAY = 0.1

# --- Streaming post-processing (thesis 3.15–3.19) ---
WINDOW_STEP_FRAMES = 15             # s: sliding step in frames (= 0.5 s @ 30 FPS)
EMA_ALPHA = 0.3                     # α: exponential smoothing factor
CONFIDENCE_THRESHOLD = 0.7          # θ_conf: trigger threshold on smoothed probs
COOLDOWN_SECONDS = 2.0              # τ_cd: minimum gap between two triggers

# --- Paths (resolved relative to this file, never to CWD) ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
AUDIO_DIR: Path = PROJECT_ROOT / "assets" / "audio"
