# src/worker.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import logging
import redis
from src.database import log_result    # ← add this import

from src.queue_manager import (
    get_redis, update_job,
    STATUS_PROCESSING, STATUS_DONE, STATUS_FAILED
)
from src.predict import ContentModerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── How many jobs to batch together ───────────────────
BATCH_SIZE    = 8
POLL_INTERVAL = 0.1   # seconds between queue polls


def run_worker():
    logger.info("Worker starting — loading model...")
    moderator = ContentModerator()
    logger.info(f"Model loaded on {moderator.device}")

    r = get_redis()
    logger.info("Worker ready — polling queue...")

    while True:
        try:
            # Pull up to BATCH_SIZE job IDs from queue
            job_ids = []
            for _ in range(BATCH_SIZE):
                item = r.rpop("moderation_queue")
                if item is None:
                    break
                job_ids.append(item)

            if not job_ids:
                time.sleep(POLL_INTERVAL)
                continue

            logger.info(f"Processing batch of {len(job_ids)} jobs")

            # Fetch job data
            jobs  = []
            texts = []
            for job_id in job_ids:
                data = r.get(f"job:{job_id}")
                if data is None:
                    continue
                job = json.loads(data)
                jobs.append(job)
                texts.append(job["text"])
                update_job(job_id, STATUS_PROCESSING)

            if not texts:
                continue

            # Run batch inference
            try:
                results = moderator.predict_batch(texts)

                for job, result in zip(jobs, results):
                    update_job(
                        job["job_id"],
                        STATUS_DONE,
                        result=result
                    )
                    log_result(result, mode="async")   # ← add this line
                    logger.info(
                        f"Job {job['job_id'][:8]}... → {result['decision']}"
                    )

            except Exception as e:
                logger.error(f"Batch inference failed: {e}")
                for job in jobs:
                    update_job(job["job_id"], STATUS_FAILED, error=str(e))

        except redis.RedisError as e:
            logger.error(f"Redis error: {e}. Retrying in 5s...")
            time.sleep(5)

        except KeyboardInterrupt:
            logger.info("Worker stopped.")
            break


if __name__ == "__main__":
    run_worker()