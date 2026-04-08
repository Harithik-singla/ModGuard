# scripts/save_to_drive.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.checkpoint import save_to_drive, list_checkpoints

# ← Change this to your actual Drive path
DRIVE_PATH = "/mnt/drive/content-moderator"

print("Current checkpoints:")
list_checkpoints()

# Save best model to Drive
save_to_drive(DRIVE_PATH, checkpoint_name="best_model")

# Save a specific epoch checkpoint
# save_to_drive(DRIVE_PATH, checkpoint_name="epoch2_20240501_valloss0.2100")