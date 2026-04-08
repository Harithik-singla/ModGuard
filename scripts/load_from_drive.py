# scripts/load_from_drive.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.checkpoint import load_from_drive

# ← Change this to your actual Drive path
DRIVE_PATH = "/mnt/drive/content-moderator"

# Load best model from Drive back to local
load_from_drive(DRIVE_PATH, checkpoint_name="best_model")
print("Model ready to use locally.")