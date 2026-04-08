# src/dataset.py
import re
import html
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from src.config import LABELS, MODEL_NAME, MAX_LEN, BATCH_SIZE, PROCESSED_DIR


def clean_text(text: str) -> str:
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\n|\r|\t', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


class ToxicDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len=MAX_LEN):
        self.df        = pd.read_csv(filepath)
        self.texts     = self.df['clean_text'].tolist()
        self.labels    = self.df[LABELS].values.astype('float32')
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx])
        }


def get_dataloaders(tokenizer):
    train_ds = ToxicDataset(PROCESSED_DIR / "train.csv", tokenizer)
    val_ds   = ToxicDataset(PROCESSED_DIR / "val.csv",   tokenizer)
    test_ds  = ToxicDataset(PROCESSED_DIR / "test.csv",  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader