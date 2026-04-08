# src/train.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import (
    PROCESSED_DIR, CHECKPOINT_DIR, LABELS, MODEL_NAME,
    DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    WARMUP_RATIO, MAX_GRAD_NORM, EARLY_STOP_PATIENCE,
    THRESHOLD, SEED
)
from src.model import ToxicClassifier
from src.dataset import get_dataloaders
from src.checkpoint import save_checkpoint, list_checkpoints


# ── Reproducibility ────────────────────────────────────
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed()


# ── Class weights ──────────────────────────────────────
def compute_class_weights(csv_path):
    df      = pd.read_csv(csv_path)
    total   = len(df)
    weights = []
    print("\nClass weights:")
    for label in LABELS:
        pos    = df[label].sum()
        # Smoothed weight — less aggressive than before
        weight = (total - pos) / (pos + 1e-6)
        weight = min(weight, 50.0)   # cap at 50 to avoid extremes
        weights.append(weight)
        print(f"  {label:<20} {weight:.2f}")
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)


# ── Metrics ────────────────────────────────────────────
def compute_metrics(all_labels, all_preds, threshold=THRESHOLD):
    preds_bin = (all_preds >= threshold).astype(int)
    metrics   = {}
    for i, label in enumerate(LABELS):
        metrics[label] = {
            'f1':        f1_score(all_labels[:, i],        preds_bin[:, i], zero_division=0),
            'precision': precision_score(all_labels[:, i], preds_bin[:, i], zero_division=0),
            'recall':    recall_score(all_labels[:, i],    preds_bin[:, i], zero_division=0),
        }
    metrics['macro_f1'] = f1_score(all_labels, preds_bin,
                                   average='macro', zero_division=0)
    return metrics


def print_metrics(metrics):
    print(f"\n  {'Label':<20} {'F1':>6} {'Precision':>10} {'Recall':>8}")
    print("  " + "-" * 48)
    for label in LABELS:
        m = metrics[label]
        print(f"  {label:<20} {m['f1']:>6.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}")
    print(f"\n  Macro F1: {metrics['macro_f1']:.4f}")


# ── Validation ─────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds  = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Validating", leave=False):
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)

            logits     = model(input_ids, attention_mask)
            loss       = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    avg_loss   = total_loss / len(loader)
    metrics    = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_preds, all_labels


# ── Main training loop ─────────────────────────────────
def train(resume_from=None):
    print(f"\nDevice : {DEVICE}")
    print(f"GPU    : {torch.cuda.get_device_name(0)}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, val_loader, _ = get_dataloaders(tokenizer)

    model         = ToxicClassifier().to(DEVICE)
    class_weights = compute_class_weights(PROCESSED_DIR / "train.csv")
    criterion     = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(),       'lr': LEARNING_RATE},
        {'params': model.classifier.parameters(), 'lr': LEARNING_RATE * 5}
    ], weight_decay=WEIGHT_DECAY)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ── Resume from checkpoint if provided ────────────
    start_epoch = 1
    if resume_from:
        from src.checkpoint import load_checkpoint
        last_epoch, _ = load_checkpoint(model, optimizer, scheduler,
                                        checkpoint_path=resume_from)
        if last_epoch:
            start_epoch = last_epoch + 1
            print(f"Resuming from epoch {start_epoch}")

    # ── Training state ─────────────────────────────────
    best_val_loss    = float('inf')
    patience_counter = 0
    best_epoch       = 0
    history          = []

    print(f"\nTraining {EPOCHS} epochs | LR: {LEARNING_RATE} | Patience: {EARLY_STOP_PATIENCE}")
    print(f"Total steps: {total_steps} | Warmup: {warmup_steps}\n")

    for epoch in range(start_epoch, EPOCHS + 1):

        # ── Train ──────────────────────────────────────
        model.train()
        total_train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")

        for batch in loop:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # ── Validate ───────────────────────────────────
        val_loss, metrics, _, _ = evaluate(model, val_loader, criterion)

        print(f"\nEpoch {epoch} — Train loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch} — Val loss:   {val_loss:.4f}")
        print_metrics(metrics)

        history.append({
            'epoch':      epoch,
            'train_loss': avg_train_loss,
            'val_loss':   val_loss,
            'macro_f1':   metrics['macro_f1']
        })

        # ── Save every epoch checkpoint ────────────────
        is_best = val_loss < best_val_loss
        save_checkpoint(
            model, tokenizer, optimizer, scheduler,
            epoch, val_loss, metrics, is_best=is_best
        )

        # ── Early stopping ─────────────────────────────
        if is_best:
            best_val_loss    = val_loss
            best_epoch       = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"\n  No improvement. Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}. Best was epoch {best_epoch}.")
                break

    # ── Training summary ───────────────────────────────
    print("\n" + "="*50)
    print("  Training Complete")
    print("="*50)
    print(f"  Best epoch    : {best_epoch}")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print("\n  Epoch history:")
    print(f"  {'Epoch':<8} {'Train Loss':>12} {'Val Loss':>10} {'Macro F1':>10}")
    print("  " + "-"*44)
    for h in history:
        marker = " ←" if h['epoch'] == best_epoch else ""
        print(f"  {h['epoch']:<8} {h['train_loss']:>12.4f} {h['val_loss']:>10.4f} "
              f"{h['macro_f1']:>10.4f}{marker}")

    print("\nAll checkpoints:")
    list_checkpoints()

    return best_epoch, best_val_loss


if __name__ == "__main__":
    # To train fresh:
    train()

    # To resume from a specific checkpoint:
    # train(resume_from="models/checkpoints/epoch2_20240501_valloss0.2100")