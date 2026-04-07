import re
import html

def clean_text(text: str) -> str:
    text = str(text)
    text = html.unescape(text)                        # decode HTML entities (&amp; → &)
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # remove URLs
    text = re.sub(r'\n|\r|\t', ' ', text)             # remove newlines/tabs
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)       # remove non-ASCII (optional — keeps English)
    text = re.sub(r'\s+', ' ', text).strip()          # collapse whitespace
    return text.lower()

import pandas as pd
from src.config import RAW_DIR, PROCESSED_DIR, LABELS

df = pd.read_csv(RAW_DIR / "train.csv")

# Clean
df['clean_text'] = df['comment_text'].apply(clean_text)

# Drop empty rows after cleaning
before = len(df)
df = df[df['clean_text'].str.len() > 5]
print(f"Dropped {before - len(df)} empty rows")

# Verify
print(df[['comment_text', 'clean_text']].head(3))

# Save
df.to_csv(PROCESSED_DIR / "cleaned.csv", index=False)
print(f"Saved {len(df)} rows to processed/cleaned.csv")