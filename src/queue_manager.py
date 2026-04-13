# src/queue_manager.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import uuid
import redis
from datetime import datetime
from src.config import LABELS

# ── Redis connection ───────────────────────────────────
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB   = 0

# Job status constants
STATUS_PENDING    = "pending"
STATUS_PROCESSING = "processing"
STATUS_DONE       = "done"
STATUS_FAILED     = "failed"

# How long to keep results in Redis (seconds)
RESULT_TTL = 60 * 60  # 1 hour


def get_redis():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True
    )


# ── Job management ─────────────────────────────────────

def enqueue_job(text: str) -> str:
    """Push a moderation job onto the queue. Returns job_id."""
    r      = get_redis()
    job_id = str(uuid.uuid4())

    job = {
        "job_id":     job_id,
        "text":       text,
        "status":     STATUS_PENDING,
        "submitted_at": datetime.utcnow().isoformat(),
        "result":     None,
        "error":      None
    }

    # Store job metadata
    r.setex(f"job:{job_id}", RESULT_TTL, json.dumps(job))

    # Push job_id onto the queue list
    r.lpush("moderation_queue", job_id)

    return job_id


def get_job(job_id: str) -> dict | None:
    """Fetch a job by its ID."""
    r    = get_redis()
    data = r.get(f"job:{job_id}")
    if data is None:
        return None
    return json.loads(data)


def update_job(job_id: str, status: str, result=None, error=None):
    """Update job status and optionally store result."""
    r    = get_redis()
    data = r.get(f"job:{job_id}")
    if data is None:
        return

    job = json.loads(data)
    job["status"]       = status
    job["result"]       = result
    job["error"]        = error
    job["updated_at"]   = datetime.utcnow().isoformat()

    r.setex(f"job:{job_id}", RESULT_TTL, json.dumps(job))


def get_queue_length() -> int:
    """How many jobs are waiting in the queue."""
    r = get_redis()
    return r.llen("moderation_queue")


def get_queue_stats() -> dict:
    """Overview of queue health."""
    r = get_redis()
    return {
        "queue_length":   r.llen("moderation_queue"),
        "total_jobs":     len(r.keys("job:*")),
        "redis_connected": r.ping()
    }