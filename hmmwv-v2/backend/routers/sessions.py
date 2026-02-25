"""
Sessions router — CRUD for chat sessions stored as JSON files.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from config import SESSIONS_DIR
from models import (
    Session, SessionSummary, SessionMessage,
    CreateSessionRequest, UpdateSessionRequest,
)

router = APIRouter()


# ── helpers ───────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def _load_session(session_id: str) -> dict:
    p = _session_path(session_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return json.loads(p.read_text(encoding="utf-8"))


def _save_session(data: dict):
    p = _session_path(data["id"])
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _to_summary(data: dict) -> SessionSummary:
    return SessionSummary(
        id=data["id"],
        title=data["title"],
        vehicle_variant=data.get("vehicle_variant", ""),
        maintenance_category=data.get("maintenance_category", ""),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        message_count=len(data.get("messages", [])),
    )


# ── auto-title helper ─────────────────────────────────────────────────────

def _auto_title(query: str, max_len: int = 60) -> str:
    """Truncate query to max_len at a word boundary and append ellipsis if needed."""
    q = query.strip()
    if len(q) <= max_len:
        return q
    truncated = q[:max_len].rsplit(" ", 1)[0]
    return truncated + "…"


# ── CRUD endpoints ────────────────────────────────────────────────────────

@router.get("/sessions", response_model=List[SessionSummary])
def list_sessions() -> List[SessionSummary]:
    """Return all sessions sorted by updated_at descending (newest first)."""
    summaries = []
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            summaries.append(_to_summary(data))
        except Exception:
            continue
    summaries.sort(key=lambda s: s.updated_at, reverse=True)
    return summaries


@router.post("/sessions", response_model=Session)
def create_session(body: CreateSessionRequest) -> Session:
    now = _now_iso()
    sid = str(uuid.uuid4())
    data = {
        "id":                   sid,
        "title":                body.title or "New Chat",
        "vehicle_variant":      body.vehicle_variant,
        "maintenance_category": body.maintenance_category,
        "created_at":           now,
        "updated_at":           now,
        "messages":             [],
    }
    _save_session(data)
    return Session(**data)


@router.get("/sessions/{session_id}", response_model=Session)
def get_session(session_id: str) -> Session:
    data = _load_session(session_id)
    return Session(**data)


@router.put("/sessions/{session_id}", response_model=Session)
def update_session(session_id: str, body: UpdateSessionRequest) -> Session:
    data = _load_session(session_id)
    if body.title is not None:
        data["title"] = body.title
    if body.vehicle_variant is not None:
        data["vehicle_variant"] = body.vehicle_variant
    if body.maintenance_category is not None:
        data["maintenance_category"] = body.maintenance_category
    data["updated_at"] = _now_iso()
    _save_session(data)
    return Session(**data)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    p = _session_path(session_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    p.unlink()
    return {"ok": True}


@router.delete("/sessions")
def clear_all_sessions() -> dict:
    """Delete all session files."""
    count = 0
    for f in SESSIONS_DIR.glob("*.json"):
        f.unlink()
        count += 1
    return {"deleted": count}


# ── Internal helper (used by chat router) ─────────────────────────────────

def append_messages_to_session(
    session_id: str,
    user_query: str,
    assistant_content: str,
    sources: list,
    images: list,
) -> None:
    """
    Append a user message and assistant response to the session JSON file.
    Also auto-sets the session title from the first user message.
    """
    p = _session_path(session_id)
    if not p.exists():
        return  # session may have been deleted during streaming

    data = json.loads(p.read_text(encoding="utf-8"))
    now  = _now_iso()

    # Auto-title from first user message
    if not data.get("messages") and data.get("title") in ("New Chat", ""):
        data["title"] = _auto_title(user_query)

    data["messages"].append({
        "role":      "user",
        "content":   user_query,
        "timestamp": now,
        "sources":   [],
        "images":    [],
    })
    data["messages"].append({
        "role":      "assistant",
        "content":   assistant_content,
        "timestamp": now,
        "sources":   sources,
        "images":    images,
    })
    data["updated_at"] = now
    _save_session(data)


def get_conversation_history(session_id: str, max_messages: int = 12) -> list:
    """
    Return the last N messages as {role, content} dicts for the AI engine.
    Strips sources, images, and timestamp — only role+content are passed to the LLM.
    """
    p = _session_path(session_id)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    messages = data.get("messages", [])
    # Take last max_messages, return only role+content
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    return [{"role": m["role"], "content": m["content"]} for m in recent]
