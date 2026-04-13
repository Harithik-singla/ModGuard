# scripts/test_api.py
import requests
import json

BASE_URL = "http://localhost:8000"

def print_result(result):
    print(f"\n  Text     : {result['text'][:60]}")
    print(f"  Decision : {result['decision']}")
    for label, info in result['labels'].items():
        if info['flagged']:
            print(f"  {label:<20} score: {info['score']:.4f}  FLAGGED"
                  + (" → REMOVED" if info['removed'] else ""))
    if 'processing_time_ms' in result:
        print(f"  Time     : {result['processing_time_ms']}ms")

# ── Health check ───────────────────────────────────────
print("1. Health check")
r = requests.get(f"{BASE_URL}/health")
print(f"   Status: {r.json()}")

# ── Single prediction ──────────────────────────────────
print("\n2. Single predictions")
test_texts = [
    "I love this community, everyone is so helpful!",
    "You are so stupid, I hate you",
    "I will find you and hurt you very badly",
    "Kill yourself you worthless garbage",
    "This movie was absolutely terrible and boring",
]

for text in test_texts:
    r = requests.post(f"{BASE_URL}/moderate", json={"text": text})
    print_result(r.json())

# ── Batch prediction ───────────────────────────────────
print("\n3. Batch prediction")
r = requests.post(f"{BASE_URL}/moderate/batch", json={"texts": test_texts})
data = r.json()
print(f"   Processed {data['count']} texts")
for result in data['results']:
    print(f"   {result['decision']:<10} {result['text'][:50]}")