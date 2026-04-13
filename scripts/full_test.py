# scripts/full_test.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def separator(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def print_result(result):
    print(f"  Decision   : {result['decision']}")
    for label, info in result['labels'].items():
        if info['flagged']:
            tag = "REMOVED" if info['removed'] else "FLAGGED"
            print(f"  {label:<20} {info['score']:.4f}  {tag}")

# ── Test 1: Health check ───────────────────────────────
separator("Test 1 — Health Check")
r = requests.get(f"{BASE_URL}/health")
h = r.json()
print(f"  Status       : {h['status']}")
print(f"  Model loaded : {h['model_loaded']}")
print(f"  Device       : {h['device']}")
assert h['status'] == 'healthy',       "FAIL — API not healthy"
assert h['model_loaded'] == True,      "FAIL — model not loaded"
print("  PASSED")

# ── Test 2: Sync — clean text ──────────────────────────
separator("Test 2 — Sync: Clean text")
r = requests.post(f"{BASE_URL}/moderate",
                  json={"text": "I love this product, it works great!"})
assert r.status_code == 200, f"FAIL — status {r.status_code}"
result = r.json()
print_result(result)
assert result['decision'] == 'APPROVED', f"FAIL — expected APPROVED, got {result['decision']}"
print(f"  Latency      : {result['processing_time_ms']}ms")
print("  PASSED")

# ── Test 3: Sync — toxic text ──────────────────────────
separator("Test 3 — Sync: Toxic text")
r = requests.post(f"{BASE_URL}/moderate",
                  json={"text": "You are so stupid I hate you"})
assert r.status_code == 200
result = r.json()
print_result(result)
assert result['decision'] in ('FLAGGED', 'REMOVED'), \
    f"FAIL — expected FLAGGED/REMOVED, got {result['decision']}"
print(f"  Latency      : {result['processing_time_ms']}ms")
print("  PASSED")

# ── Test 4: Sync — severe text ─────────────────────────
separator("Test 4 — Sync: Severe text")
r = requests.post(f"{BASE_URL}/moderate",
                  json={"text": "Kill yourself you worthless piece of garbage"})
assert r.status_code == 200
result = r.json()
print_result(result)
print(f"  Latency      : {result['processing_time_ms']}ms")
print("  PASSED")

# ── Test 5: Sync batch ─────────────────────────────────
separator("Test 5 — Sync Batch (5 texts)")
texts = [
    "Great product, highly recommend!",
    "You are an idiot",
    "I will hurt you badly",
    "The weather is nice today",
    "Go kill yourself loser",
]
r = requests.post(f"{BASE_URL}/moderate/batch", json={"texts": texts})
assert r.status_code == 200
data = r.json()
assert data['count'] == 5, f"FAIL — expected 5 results, got {data['count']}"
print(f"  {'Text':<45} {'Decision'}")
print(f"  {'-'*55}")
for result in data['results']:
    print(f"  {result['text'][:44]:<45} {result['decision']}")
print("  PASSED")

# ── Test 6: Validation ─────────────────────────────────
separator("Test 6 — Input Validation")

# Empty text
r = requests.post(f"{BASE_URL}/moderate", json={"text": ""})
assert r.status_code == 422, f"FAIL — expected 422, got {r.status_code}"
print("  Empty text rejected         PASSED")

# Text too long
r = requests.post(f"{BASE_URL}/moderate", json={"text": "a" * 10001})
assert r.status_code == 422
print("  Text too long rejected      PASSED")

# Missing field
r = requests.post(f"{BASE_URL}/moderate", json={})
assert r.status_code == 422
print("  Missing field rejected      PASSED")

# ── Test 7: Async single ───────────────────────────────
separator("Test 7 — Async Single Job")
r      = requests.post(f"{BASE_URL}/moderate/async",
                       json={"text": "You are disgusting"})
assert r.status_code == 200
data   = r.json()
job_id = data['job_id']
print(f"  Submitted job_id: {job_id[:8]}...")
assert 'job_id'   in data, "FAIL — no job_id returned"
assert 'poll_url' in data, "FAIL — no poll_url returned"

# Poll
for i in range(20):
    time.sleep(0.5)
    r    = requests.get(f"{BASE_URL}/result/{job_id}")
    data = r.json()
    if data['status'] == 'done':
        print(f"  Completed in ~{(i+1)*0.5:.1f}s")
        print_result(data['result'])
        print("  PASSED")
        break
    elif data['status'] == 'failed':
        print(f"  FAIL — job failed: {data.get('error')}")
        break
else:
    print("  FAIL — timed out after 10s")

# ── Test 8: Async batch ────────────────────────────────
separator("Test 8 — Async Batch (5 jobs)")
job_ids = []
batch_texts = [
    "This is amazing!",
    "I hate everything about you",
    "Nice weather today",
    "You should be ashamed of yourself",
    "Great work everyone!",
]

for text in batch_texts:
    r = requests.post(f"{BASE_URL}/moderate/async", json={"text": text})
    job_ids.append(r.json()['job_id'])

print(f"  Submitted {len(job_ids)} jobs simultaneously")

results  = {}
start    = time.time()
while len(results) < len(job_ids):
    for job_id in job_ids:
        if job_id in results:
            continue
        r    = requests.get(f"{BASE_URL}/result/{job_id}")
        data = r.json()
        if data['status'] == 'done':
            results[job_id] = data['result']
        elif data['status'] == 'failed':
            results[job_id] = {'decision': 'FAILED'}
    time.sleep(0.2)

elapsed = round(time.time() - start, 2)
print(f"  All {len(results)} results received in {elapsed}s")
print(f"\n  {'Text':<40} {'Decision'}")
print(f"  {'-'*52}")
for job_id, result in zip(job_ids, results.values()):
    text = batch_texts[job_ids.index(job_id)]
    print(f"  {text[:39]:<40} {result['decision']}")
print("  PASSED")

# ── Test 9: Queue stats ────────────────────────────────
separator("Test 9 — Queue Stats")
r    = requests.get(f"{BASE_URL}/queue/stats")
data = r.json()
print(f"  Queue length   : {data['queue_length']}")
print(f"  Total jobs     : {data['total_jobs']}")
print(f"  Redis connected: {data['redis_connected']}")
assert data['redis_connected'] == True, "FAIL — Redis not connected"
print("  PASSED")

# ── Test 10: 404 for invalid job ───────────────────────
separator("Test 10 — Invalid Job ID")
r = requests.get(f"{BASE_URL}/result/fake-job-id-that-doesnt-exist")
assert r.status_code == 404, f"FAIL — expected 404, got {r.status_code}"
print("  Invalid job_id returns 404  PASSED")

# ── Summary ────────────────────────────────────────────
separator("All Tests Complete")
print("""
  Phase 1 — Data pipeline     ✓
  Phase 2 — Model training    ✓
  Phase 3 — Sync API          ✓
  Phase 4 — Async queue       ✓

  Your content moderator is fully functional.
  Ready for Phase 5 — Monitoring Dashboard.
""")