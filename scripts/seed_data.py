# scripts/seed_data.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import time

BASE_URL = "http://localhost:8000"

texts = [
    "I love this community!",
    "You are so stupid and useless",
    "Great product, works perfectly",
    "Kill yourself you worthless idiot",
    "Thanks for all the help everyone",
    "I hate you so much",
    "Beautiful day today!",
    "You should be ashamed of yourself",
    "This is absolutely amazing work",
    "Go die in a hole you moron",
    "Really helpful tutorial, thank you",
    "I will destroy you and everything you love",
    "Wonderful experience overall",
    "You are disgusting garbage",
    "Best purchase I ever made",
    "Shut up you stupid idiot",
    "Highly recommend this to everyone",
    "You are the worst person alive",
    "This helped me so much, thank you",
    "I want to hurt you badly",
]

print(f"Seeding {len(texts)} requests...")
for i, text in enumerate(texts):
    r      = requests.post(f"{BASE_URL}/moderate", json={"text": text})
    result = r.json()
    print(f"  {i+1:>2}. {result['decision']:<10} {text[:40]}")
    time.sleep(0.2)

print("\nDone! Open the dashboard to see the data.")