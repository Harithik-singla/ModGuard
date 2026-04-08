# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel
from src.config import MODEL_NAME, NUM_LABELS


class ToxicClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_labels=NUM_LABELS, dropout=0.3):
        super().__init__()

        # Pretrained DistilBERT backbone
        self.bert = AutoModel.from_pretrained(model_name)

        # Classification head
        hidden_size = self.bert.config.hidden_size   # 768 for distilbert
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
            # No sigmoid here — BCEWithLogitsLoss handles it internally
            # which is numerically more stable
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_output)
        return logits