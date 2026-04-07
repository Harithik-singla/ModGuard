# Run as: python -c "import verify" OR save as verify.py and run it

import torch
from transformers import AutoTokenizer

print("=== Environment Check ===")
print(f"PyTorch version   : {torch.__version__}")
print(f"CUDA available    : {torch.cuda.is_available()}")
print(f"Device            : {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
sample = tokenizer("This is a test sentence.", return_tensors="pt")
print(f"Tokenizer working : True — token count: {sample['input_ids'].shape[1]}")
