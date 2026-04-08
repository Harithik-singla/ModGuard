# src/config.py
import os
import torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = BASE_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
MODEL_DIR      = BASE_DIR / "models"
# If CHECKPOINT_DIR is provided (e.g. in Colab), write checkpoints there.
# Example:
#   export CHECKPOINT_DIR="/content/drive/MyDrive/ModGuard/checkpoints"
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", str(MODEL_DIR / "checkpoints")))
LOG_DIR        = BASE_DIR / "logs"

for d in [CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Labels ─────────────────────────────────────────────
LABELS     = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
NUM_LABELS = len(LABELS)

# ── Model ──────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 128

# ── Training (improved settings) ───────────────────────
BATCH_SIZE          = 32
EPOCHS              = 6       # was 4
LEARNING_RATE       = 5e-6   # was 2e-5 — slower, more careful
WEIGHT_DECAY        = 0.01
WARMUP_RATIO        = 0.1
MAX_GRAD_NORM       = 1.0
EARLY_STOP_PATIENCE = 3       # was 2 — more patience
THRESHOLD           = 0.5
SEED                = 42

# ── Device ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"