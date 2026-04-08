# src/checkpoint.py
import os
import sys
import json
import shutil
import torch
from pathlib import Path
from datetime import datetime
from src.config import CHECKPOINT_DIR, LABELS, DEVICE


# ── Save checkpoint ────────────────────────────────────
def save_checkpoint(model, tokenizer, optimizer, scheduler,
                    epoch, val_loss, metrics, is_best=False):
    """
    Saves a full checkpoint — model weights, optimizer state,
    scheduler state, metrics, and config. Can resume from exactly
    this point if training is interrupted.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name      = f"epoch{epoch}_{timestamp}_valloss{val_loss:.4f}"
    save_dir  = CHECKPOINT_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save BERT backbone + tokenizer (HuggingFace format)
    model.bert.save_pretrained(save_dir / "bert_backbone")
    tokenizer.save_pretrained(save_dir / "bert_backbone")

    # 2. Save classifier head weights
    torch.save(model.classifier.state_dict(), save_dir / "classifier_head.pt")

    # 3. Save training state (optimizer, scheduler, epoch)
    torch.save({
        'epoch':            epoch,
        'val_loss':         val_loss,
        'optimizer_state':  optimizer.state_dict(),
        'scheduler_state':  scheduler.state_dict(),
    }, save_dir / "training_state.pt")

    # 4. Save metrics as JSON (human readable)
    metrics_to_save = {
        'epoch':    epoch,
        'val_loss': val_loss,
        'macro_f1': metrics['macro_f1'],
        'per_label': {
            label: {
                'f1':        metrics[label]['f1'],
                'precision': metrics[label]['precision'],
                'recall':    metrics[label]['recall'],
            } for label in LABELS
        },
        'timestamp': timestamp
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    print(f"\n  Checkpoint saved → {save_dir.name}")

    # 5. If this is the best model, copy to /best_model
    if is_best:
        best_dir = CHECKPOINT_DIR / "best_model"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(save_dir, best_dir)
        print(f"  Best model updated → checkpoints/best_model/")

    return save_dir


# ── Load checkpoint ────────────────────────────────────
def load_checkpoint(model, optimizer=None, scheduler=None,
                    checkpoint_path=None, load_best=True):
    """
    Loads a checkpoint. If checkpoint_path is None, loads best_model.
    Returns the epoch and val_loss the checkpoint was saved at.
    """
    if checkpoint_path is None:
        load_dir = CHECKPOINT_DIR / "best_model" if load_best \
                   else _get_latest_checkpoint()
    else:
        load_dir = Path(checkpoint_path)

    if not load_dir.exists():
        print(f"No checkpoint found at {load_dir}")
        return None, None

    print(f"\nLoading checkpoint from: {load_dir.name}")

    # 1. Load BERT backbone
    from transformers import AutoModel
    model.bert = AutoModel.from_pretrained(load_dir / "bert_backbone").to(DEVICE)

    # 2. Load classifier head
    model.classifier.load_state_dict(
        torch.load(load_dir / "classifier_head.pt", map_location=DEVICE)
    )

    # 3. Load training state if resuming
    epoch, val_loss = None, None
    state_path = load_dir / "training_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location=DEVICE)
        epoch    = state['epoch']
        val_loss = state['val_loss']
        if optimizer:
            optimizer.load_state_dict(state['optimizer_state'])
        if scheduler:
            scheduler.load_state_dict(state['scheduler_state'])
        print(f"  Resumed from epoch {epoch} | val loss: {val_loss:.4f}")

    # 4. Print saved metrics
    metrics_path = load_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        print(f"  Macro F1 at checkpoint: {m['macro_f1']:.4f}")

    return epoch, val_loss


# ── List all checkpoints ───────────────────────────────
def list_checkpoints():
    checkpoints = [
        d for d in CHECKPOINT_DIR.iterdir()
        if d.is_dir() and d.name != "best_model"
    ]
    if not checkpoints:
        print("No checkpoints found.")
        return

    print(f"\n{'#':<4} {'Name':<45} {'Val Loss':>10} {'Macro F1':>10}")
    print("-" * 72)

    for i, ckpt in enumerate(sorted(checkpoints)):
        metrics_path = ckpt / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            print(f"  {i+1:<3} {ckpt.name:<45} {m['val_loss']:>10.4f} {m['macro_f1']:>10.4f}")
        else:
            print(f"  {i+1:<3} {ckpt.name:<45} {'N/A':>10} {'N/A':>10}")


# ── Get latest checkpoint ──────────────────────────────
def _get_latest_checkpoint():
    checkpoints = sorted([
        d for d in CHECKPOINT_DIR.iterdir()
        if d.is_dir() and d.name != "best_model"
    ])
    return checkpoints[-1] if checkpoints else None


# ── Sync to Google Drive ───────────────────────────────
def save_to_drive(drive_path: str, checkpoint_name: str = "best_model"):
    """
    Copies a checkpoint to Google Drive.
    drive_path example: '/content/drive/MyDrive/content-moderator'
    """
    src  = CHECKPOINT_DIR / checkpoint_name
    dest = Path(drive_path) / "checkpoints" / checkpoint_name

    if not src.exists():
        print(f"Checkpoint '{checkpoint_name}' not found locally.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        shutil.rmtree(dest)

    shutil.copytree(src, dest)
    print(f"\nSaved to Drive → {dest}")
    _print_folder_size(dest)


def load_from_drive(drive_path: str, checkpoint_name: str = "best_model"):
    """
    Copies a checkpoint from Google Drive to local checkpoints folder.
    """
    src  = Path(drive_path) / "checkpoints" / checkpoint_name
    dest = CHECKPOINT_DIR / checkpoint_name

    if not src.exists():
        print(f"Checkpoint not found in Drive at {src}")
        return

    if dest.exists():
        shutil.rmtree(dest)

    shutil.copytree(src, dest)
    print(f"\nLoaded from Drive → {dest}")
    _print_folder_size(dest)


def _print_folder_size(path):
    total = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
    print(f"  Size: {total / 1e6:.1f} MB")