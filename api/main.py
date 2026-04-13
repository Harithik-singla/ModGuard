# api/main.py  — full updated version
import sys, os
from src.database import log_result
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Protect multiprocessing on Windows
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

from src.predict import ContentModerator
from src.queue_manager import (
    enqueue_job, get_job, get_queue_stats,
    STATUS_DONE, STATUS_FAILED
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

moderator: Optional[ContentModerator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global moderator
    logger.info("Loading model...")
    moderator = ContentModerator()
    logger.info("Model ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Content Moderator API",
    description="Real-time text moderation using DistilBERT",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────
class ModerateRequest(BaseModel):
    text: str

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        if len(v) > 10000:
            raise ValueError("Text too long — max 10,000 characters")
        return v


class BatchModerateRequest(BaseModel):
    texts: list[str]

    @validator('texts')
    def validate_batch(cls, v):
        if not v:
            raise ValueError("texts list cannot be empty")
        if len(v) > 32:
            raise ValueError("Max 32 texts per batch")
        return v


# ── Logging middleware ─────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    elapsed  = round((time.time() - start) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} "
                f"→ {response.status_code} ({elapsed}ms)")
    return response


# ── Routes ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Content Moderator API",
        "version": "2.0.0",
        "endpoints": {
            "sync  POST": "/moderate",
            "sync  POST": "/moderate/batch",
            "async POST": "/moderate/async",
            "async GET":  "/result/{job_id}",
            "stats GET":  "/queue/stats",
            "health GET": "/health",
        }
    }


@app.get("/health")
def health():
    return {
        "status":       "healthy",
        "model_loaded": moderator is not None,
        "device":       str(moderator.device) if moderator else None
    }


# ── Sync endpoints (Phase 3) ───────────────────────────

@app.post("/moderate")
def moderate(request: ModerateRequest):
    if moderator is None:
        raise HTTPException(503, "Model not loaded")
    try:
        result = moderator.predict_one(request.text)
        log_result(result, mode="sync")
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/moderate/batch")
def moderate_batch(request: BatchModerateRequest):
    if moderator is None:
        raise HTTPException(503, "Model not loaded")
    try:
        results = moderator.predict_batch(request.texts)
        for r in results:                       # ← add this block
            log_result(r, mode="sync")
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Async endpoints (Phase 4) ──────────────────────────

@app.post("/moderate/async")
def moderate_async(request: ModerateRequest):
    """
    Submit a job to the queue.
    Returns immediately with a job_id.
    Poll GET /result/{job_id} to fetch the result.
    """
    try:
        job_id = enqueue_job(request.text)
        return {
            "job_id":  job_id,
            "status":  "pending",
            "poll_url": f"/result/{job_id}"
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/result/{job_id}")
def get_result(job_id: str):
    """Poll this endpoint to check job status and fetch result."""
    job = get_job(job_id)

    if job is None:
        raise HTTPException(404, f"Job {job_id} not found or expired")

    response = {
        "job_id":       job_id,
        "status":       job["status"],
        "submitted_at": job.get("submitted_at"),
        "updated_at":   job.get("updated_at"),
    }

    if job["status"] == STATUS_DONE:
        response["result"] = job["result"]

    elif job["status"] == STATUS_FAILED:
        response["error"] = job.get("error")

    return response


@app.get("/queue/stats")
def queue_stats():
    """Monitor queue health."""
    try:
        return get_queue_stats()
    except Exception as e:
        raise HTTPException(500, f"Redis error: {e}")