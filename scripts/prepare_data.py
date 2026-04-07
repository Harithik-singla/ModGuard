# scripts/prepare_data.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import html
import pandas as pd
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# ── Paths (inline so no config import needed) ──────────
BASE_DIR      = Path(__file__).resolve().parent.parent
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# ── Step 1: Load ───────────────────────────────────────
print("Loading raw data...")
df = pd.read_csv(RAW_DIR / "train.csv")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# ── Step 2: EDA summary ────────────────────────────────
print("\n--- Label Distribution ---")
for label in LABELS:
    count = df[label].sum()
    pct   = count / len(df) * 100
    print(f"  {label:<20} {count:>6}  ({pct:.2f}%)")

clean_count = (df[LABELS].sum(axis=1) == 0).sum()
print(f"\n  Clean (no label)     {clean_count:>6}  ({clean_count/len(df)*100:.1f}%)")

df['text_len'] = df['comment_text'].apply(lambda x: len(str(x).split()))
print(f"\n  Avg comment length   : {df['text_len'].mean():.0f} words")
print(f"  95th pct length      : {df['text_len'].quantile(0.95):.0f} words")
print(f"  99th pct length      : {df['text_len'].quantile(0.99):.0f} words")

# ── Step 3: Clean ──────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\n|\r|\t', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

print("\nCleaning text...")
before = len(df)
df['clean_text'] = df['comment_text'].apply(clean_text)
df = df[df['clean_text'].str.len() > 5].reset_index(drop=True)
print(f"Dropped {before - len(df)} empty rows. Remaining: {len(df)}")

df.to_csv(PROCESSED_DIR / "cleaned.csv", index=False)
print(f"Saved → data/processed/cleaned.csv")

# ── Step 4: Split ──────────────────────────────────────
print("\nSplitting into train / val / test...")
X = df['clean_text'].values
y = df[LABELS].values

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, temp_idx in msss.split(X, y):
    train_df = df.iloc[train_idx]
    temp_df  = df.iloc[temp_idx]

msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_idx, test_idx in msss2.split(temp_df['clean_text'].values, temp_df[LABELS].values):
    val_df  = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
val_df.to_csv(PROCESSED_DIR   / "val.csv",   index=False)
test_df.to_csv(PROCESSED_DIR  / "test.csv",  index=False)

print(f"\nSplit complete:")
print(f"  Train : {len(train_df)} rows")
print(f"  Val   : {len(val_df)} rows")
print(f"  Test  : {len(test_df)} rows")

# ── Step 5: Verify label balance preserved ─────────────
print("\n--- Label % across splits ---")
for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
    rates = split[LABELS].mean() * 100
    print(f"\n  {name} ({len(split)} rows)")
    for label, pct in rates.items():
        print(f"    {label:<20} {pct:.2f}%")

print("\nPhase 1 complete! Ready for Phase 2.")