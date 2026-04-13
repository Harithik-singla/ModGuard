# scripts/test_single_async.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import time

BASE_URL = "http://localhost:8000"

# Submit one job
r      = requests.post(f"{BASE_URL}/moderate/async", json={"text": "I hate you so much"})
data   = r.json()
job_id = data["job_id"]
print(f"Submitted → job_id: {job_id[:8]}...")
print(f"Status   : {data['status']}")

# Poll until done
print("\nPolling for result...")
for i in range(20):
    time.sleep(0.5)
    r    = requests.get(f"{BASE_URL}/result/{job_id}")
    data = r.json()
    print(f"  Poll {i+1:>2} — status: {data['status']}")

    if data["status"] == "done":
        result = data["result"]
        print(f"\nDecision : {result['decision']}")
        print(f"Scores   :")
        for label, info in result["labels"].items():
            flag = "FLAGGED" if info["flagged"] else ""
            print(f"  {label:<20} {info['score']:.4f}  {flag}")
        break

    elif data["status"] == "failed":
        print(f"Job failed: {data.get('error')}")
        break
else:
    print("Timed out — worker may not be running")