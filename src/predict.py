# src/predict.py
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import torch
torch.set_grad_enabled(False)   # ← add this line

import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from src.config import (
    CHECKPOINT_DIR, LABELS, MODEL_NAME,
    MAX_LEN, DEVICE, THRESHOLD
)
from src.model import ToxicClassifier


# ── Decision thresholds ────────────────────────────────
# Above REMOVE_THRESHOLD  → auto remove
# Above FLAG_THRESHOLD    → flag for human review
# Below FLAG_THRESHOLD    → approved
REMOVE_THRESHOLD = 0.85
FLAG_THRESHOLD   = 0.50


class ContentModerator:
    def __init__(self, checkpoint_path=None, threshold_path=None):
        self.device    = DEVICE
        self.tokenizer = None
        self.model     = None
        self.thresholds = self._load_thresholds(threshold_path)
        self._load_model(checkpoint_path)

    # ── Load per-label thresholds ──────────────────────
    def _load_thresholds(self, threshold_path):
        default = {label: THRESHOLD for label in LABELS}
        if threshold_path is None:
            threshold_path = CHECKPOINT_DIR / "thresholds.json"
        if Path(threshold_path).exists():
            with open(threshold_path) as f:
                loaded = json.load(f)
            print(f"Loaded tuned thresholds from {threshold_path}")
            return loaded
        print("Using default threshold 0.5 for all labels")
        return default

    # ── Load model from checkpoint ─────────────────────
    def _load_model(self, checkpoint_path=None):
        load_dir = Path(checkpoint_path) if checkpoint_path \
                   else CHECKPOINT_DIR / "best_model"

        if not load_dir.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {load_dir}. "
                f"Run training first."
            )

        print(f"Loading model from: {load_dir}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_dir / "bert_backbone"
        )

        # Load model
        self.model = ToxicClassifier().to(self.device)
        self.model.bert = AutoModel.from_pretrained(
            load_dir / "bert_backbone"
        ).to(self.device)
        self.model.classifier.load_state_dict(
            torch.load(
                load_dir / "classifier_head.pt",
                map_location=self.device
            )
        )
        self.model.eval()
        print(f"Model loaded on {self.device}")

    # ── Preprocess text ────────────────────────────────
    def _preprocess(self, text: str):
        import re, html
        text = str(text)
        text = html.unescape(text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\n|\r|\t', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    # ── Run inference on a single text ────────────────
    def predict_one(self, text: str) -> dict:
        start_time = time.time()

        clean  = self._preprocess(text)
        inputs = self.tokenizer(
            clean,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids      = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs  = torch.sigmoid(logits).cpu().numpy()[0]

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        # Build per-label results
        label_results = {}
        any_removed   = False
        any_flagged   = False

        for i, label in enumerate(LABELS):
            score    = float(probs[i])
            thresh   = self.thresholds.get(label, THRESHOLD)
            flagged  = score >= thresh
            removed  = score >= REMOVE_THRESHOLD

            label_results[label] = {
                "score":   round(score, 4),
                "flagged": flagged,
                "removed": removed
            }

            if removed:
                any_removed = True
            elif flagged:
                any_flagged = True

        # Overall decision
        if any_removed:
            decision = "REMOVED"
        elif any_flagged:
            decision = "FLAGGED"
        else:
            decision = "APPROVED"

        return {
            "text":               text,
            "clean_text":         clean,
            "decision":           decision,
            "labels":             label_results,
            "processing_time_ms": elapsed_ms
        }

    # ── Run inference on a batch of texts ─────────────
    def predict_batch(self, texts: list[str]) -> list[dict]:
        start_time = time.time()

        cleaned = [self._preprocess(t) for t in texts]
        inputs  = self.tokenizer(
            cleaned,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids      = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs  = torch.sigmoid(logits).cpu().numpy()

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        results    = []

        for j, text in enumerate(texts):
            label_results = {}
            any_removed   = False
            any_flagged   = False

            for i, label in enumerate(LABELS):
                score   = float(probs[j][i])
                thresh  = self.thresholds.get(label, THRESHOLD)
                flagged = score >= thresh
                removed = score >= REMOVE_THRESHOLD

                label_results[label] = {
                    "score":   round(score, 4),
                    "flagged": flagged,
                    "removed": removed
                }
                if removed:
                    any_removed = True
                elif flagged:
                    any_flagged = True

            decision = "REMOVED" if any_removed else \
                       "FLAGGED" if any_flagged else "APPROVED"

            results.append({
                "text":     text,
                "decision": decision,
                "labels":   label_results,
            })

        # Add total batch time to last result
        results[-1]["batch_processing_time_ms"] = elapsed_ms
        return results


# ── Quick test ─────────────────────────────────────────
if __name__ == "__main__":
    moderator = ContentModerator()

    test_cases = [
        "I love this community, everyone is so helpful!",
        "You are so stupid, I hate you",
        "I will find you and hurt you badly",
        "This movie was absolutely terrible",
        "Kill yourself you worthless piece of garbage",
    ]

    print("\n" + "="*60)
    print("  Content Moderator — Test Run")
    print("="*60)

    for text in test_cases:
        result = moderator.predict_one(text)
        print(f"\n  Text     : {text}")
        print(f"  Decision : {result['decision']} ({result['processing_time_ms']}ms)")
        flagged = {k: f"{v['score']:.2f}" for k, v in result['labels'].items()
                   if v['flagged']}
        if flagged:
            print(f"  Flagged  : {flagged}")