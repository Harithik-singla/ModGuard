# src/config.py
from pathlib import Path

BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = BASE_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
MODEL_DIR      = BASE_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
LOG_DIR        = BASE_DIR / "logs"

MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 128
NUM_LABELS   = 6

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

BATCH_SIZE    = 32
EPOCHS        = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
WARMUP_STEPS  = 200
SEED          = 42

THRESHOLD = 0.5     # confidence threshold for positive label

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"