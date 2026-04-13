# src/database.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, Boolean, DateTime, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from pathlib import Path
from src.config import BASE_DIR

# ── Database setup ─────────────────────────────────────
DB_PATH = BASE_DIR / "logs" / "moderation.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine       = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)
Base         = declarative_base()


# ── Table schema ───────────────────────────────────────
class ModerationLog(Base):
    __tablename__ = "moderation_logs"

    id             = Column(Integer, primary_key=True, index=True)
    timestamp      = Column(DateTime, default=datetime.utcnow, index=True)
    text           = Column(Text)
    clean_text     = Column(Text)
    decision       = Column(String(20), index=True)   # APPROVED/FLAGGED/REMOVED
    processing_ms  = Column(Float)
    mode           = Column(String(10), default="sync") # sync/async

    # Per-label scores
    toxic_score          = Column(Float)
    severe_toxic_score   = Column(Float)
    obscene_score        = Column(Float)
    threat_score         = Column(Float)
    insult_score         = Column(Float)
    identity_hate_score  = Column(Float)

    # Per-label flags
    toxic_flagged          = Column(Boolean)
    severe_toxic_flagged   = Column(Boolean)
    obscene_flagged        = Column(Boolean)
    threat_flagged         = Column(Boolean)
    insult_flagged         = Column(Boolean)
    identity_hate_flagged  = Column(Boolean)


# Create tables
Base.metadata.create_all(engine)


# ── Log a single result ────────────────────────────────
def log_result(result: dict, mode: str = "sync"):
    session = SessionLocal()
    try:
        labels = result.get("labels", {})
        entry  = ModerationLog(
            timestamp     = datetime.utcnow(),
            text          = result.get("text", "")[:500],
            clean_text    = result.get("clean_text", "")[:500],
            decision      = result.get("decision", "UNKNOWN"),
            processing_ms = result.get("processing_time_ms", 0),
            mode          = mode,

            toxic_score         = labels.get("toxic",         {}).get("score", 0),
            severe_toxic_score  = labels.get("severe_toxic",  {}).get("score", 0),
            obscene_score       = labels.get("obscene",       {}).get("score", 0),
            threat_score        = labels.get("threat",        {}).get("score", 0),
            insult_score        = labels.get("insult",        {}).get("score", 0),
            identity_hate_score = labels.get("identity_hate", {}).get("score", 0),

            toxic_flagged         = labels.get("toxic",         {}).get("flagged", False),
            severe_toxic_flagged  = labels.get("severe_toxic",  {}).get("flagged", False),
            obscene_flagged       = labels.get("obscene",       {}).get("flagged", False),
            threat_flagged        = labels.get("threat",        {}).get("flagged", False),
            insult_flagged        = labels.get("insult",        {}).get("flagged", False),
            identity_hate_flagged = labels.get("identity_hate", {}).get("flagged", False),
        )
        session.add(entry)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"DB log error: {e}")
    finally:
        session.close()


# ── Query helpers for dashboard ────────────────────────
def get_summary_stats():
    session = SessionLocal()
    try:
        from sqlalchemy import func
        total    = session.query(func.count(ModerationLog.id)).scalar()
        flagged  = session.query(func.count(ModerationLog.id))\
                          .filter(ModerationLog.decision == "FLAGGED").scalar()
        removed  = session.query(func.count(ModerationLog.id))\
                          .filter(ModerationLog.decision == "REMOVED").scalar()
        approved = session.query(func.count(ModerationLog.id))\
                          .filter(ModerationLog.decision == "APPROVED").scalar()
        avg_ms   = session.query(func.avg(ModerationLog.processing_ms)).scalar()
        return {
            "total":    total    or 0,
            "approved": approved or 0,
            "flagged":  flagged  or 0,
            "removed":  removed  or 0,
            "avg_ms":   round(avg_ms or 0, 1)
        }
    finally:
        session.close()


def get_recent_logs(limit=20):
    session = SessionLocal()
    try:
        rows = session.query(ModerationLog)\
                      .order_by(ModerationLog.timestamp.desc())\
                      .limit(limit).all()
        return [{
            "time":     row.timestamp.strftime("%H:%M:%S"),
            "text":     row.text[:60] + "..." if len(row.text) > 60 else row.text,
            "decision": row.decision,
            "latency":  f"{row.processing_ms:.0f}ms",
            "toxic":    f"{row.toxic_score:.2f}",
            "insult":   f"{row.insult_score:.2f}",
            "threat":   f"{row.threat_score:.2f}",
        } for row in rows]
    finally:
        session.close()


def get_timeline_data(minutes=60):
    """Requests per minute for the last N minutes."""
    session = SessionLocal()
    try:
        from sqlalchemy import func, text
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        rows   = session.query(
            func.strftime("%H:%M", ModerationLog.timestamp).label("minute"),
            func.count(ModerationLog.id).label("count"),
            ModerationLog.decision
        ).filter(ModerationLog.timestamp >= cutoff)\
         .group_by("minute", ModerationLog.decision)\
         .order_by("minute").all()
        return [{"minute": r.minute, "count": r.count,
                 "decision": r.decision} for r in rows]
    finally:
        session.close()


def get_label_distribution():
    """Average score per label across all logs."""
    session = SessionLocal()
    try:
        from sqlalchemy import func
        row = session.query(
            func.avg(ModerationLog.toxic_score).label("toxic"),
            func.avg(ModerationLog.severe_toxic_score).label("severe_toxic"),
            func.avg(ModerationLog.obscene_score).label("obscene"),
            func.avg(ModerationLog.threat_score).label("threat"),
            func.avg(ModerationLog.insult_score).label("insult"),
            func.avg(ModerationLog.identity_hate_score).label("identity_hate"),
        ).first()
        if row is None:
            return {}
        return {
            "toxic":         round(row.toxic         or 0, 4),
            "severe_toxic":  round(row.severe_toxic  or 0, 4),
            "obscene":       round(row.obscene        or 0, 4),
            "threat":        round(row.threat         or 0, 4),
            "insult":        round(row.insult         or 0, 4),
            "identity_hate": round(row.identity_hate  or 0, 4),
        }
    finally:
        session.close()