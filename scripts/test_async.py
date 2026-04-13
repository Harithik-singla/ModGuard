# scripts/test_async.py
import requests
import time

BASE_URL = "http://localhost:8000"

texts = [
    "I love this product so much!",
    "You are absolutely disgusting",
    "I will destroy everything you love",
    "Great weather today isn't it",
    "Go kill yourself you worthless idiot",
]

print("Submitting jobs asynchronously...\n")
job_ids = []

# Submit all at once — all return instantly
for text in texts:
    r = requests.post(f"{BASE_URL}/moderate/async", json={"text": text})
    data = r.json()
    job_ids.append(data["job_id"])
    print(f"  Submitted → job_id: {data['job_id'][:8]}...")

print(f"\nAll {len(job_ids)} jobs submitted. Polling for results...\n")

# Poll until all done
results = {}
while len(results) < len(job_ids):
    for job_id in job_ids:
        if job_id in results:
            continue
        r    = requests.get(f"{BASE_URL}/result/{job_id}")
        data = r.json()
        if data["status"] == "done":
            results[job_id] = data["result"]
            print(f"  {job_id[:8]}... → {data['result']['decision']:<10} "
                  f"{data['result']['text'][:45]}")
        elif data["status"] == "failed":
            results[job_id] = {"error": data.get("error")}
            print(f"  {job_id[:8]}... → FAILED")
    time.sleep(0.2)

# Queue stats
r = requests.get(f"{BASE_URL}/queue/stats")
print(f"\nQueue stats: {r.json()}")