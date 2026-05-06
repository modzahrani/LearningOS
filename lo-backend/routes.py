from datetime import datetime, timezone
from pathlib import Path
import os
import re
from typing import Any, List, Optional
from uuid import NAMESPACE_DNS, uuid5
import requests

import asyncpg
from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Query, Response
from supabase import create_client

import json

from agents.Lesson_agent import generate_lesson_storyboard, recommend_lessons
from agents.RAG_agent import answer_question
from auth import get_user_from_supabase, verify_token
from db import get_db
from models.models import (
    ChatMessageRequest,
    ChatMessageResponse,
    DashboardResponse,
    ForgotPasswordRequest,
    LessonDetailResponse,
    LessonStateRequest,
    LessonStateResponse,
    OAuthSessionRequest,
    OAuthStartRequest,
    QuizAnswerRequest,
    QuizAnswerResponse,
    QuizAnswerDevRequest,
    QuizStartDevRequest,
    QuizStartResponse,
    ResetPasswordRequest,
    LessonAssignmentResponse,
    LessonCompletionRequest,
    PathSelect,
    ProfileResponse,
    UserCreate,
    UserLogin,
)
from quiz_engine import (
    QUIZ_TOTAL_QUESTIONS,
    clear_active_session,
    create_session,
    evaluate_answer,
    get_active_session_id,
    get_session,
    next_question,
    save_session,
)
from redis_client import redis_client

# -----------------------------------------------------------------------------
# Router setup and environment configuration
# -----------------------------------------------------------------------------

router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
DOCS_FOLDER = Path(__file__).resolve().parent / "docs"

def parse_bool(value: str | None, default: bool = False) -> bool:
    """Parse common truthy string values from environment variables."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# Allow bypassing Supabase auth for testing or alternative auth setups.
BYPASS_SUPABASE_AUTH = parse_bool(os.getenv("BYPASS_SUPABASE_AUTH"), False)

if not BYPASS_SUPABASE_AUTH and (not SUPABASE_URL or not SUPABASE_KEY):
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY) must be set")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if not BYPASS_SUPABASE_AUTH else None

# Cookie and app behavior configuration.
ACCESS_TOKEN_COOKIE = os.getenv("ACCESS_TOKEN_COOKIE", "access_token")
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "true" if APP_ENV == "production" else "false").lower() in {"1", "true", "yes", "on"}
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "none" if APP_ENV == "production" else "lax")
COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN")
FRONTEND_URL = os.getenv("FRONTEND_URL", "").strip()
EMAIL_CONFIRM_REDIRECT_URL = os.getenv("EMAIL_CONFIRM_REDIRECT_URL", "").strip()
LESSON_STORYBOARD_CACHE_TTL_SECONDS = int(
    os.getenv("LESSON_STORYBOARD_CACHE_TTL_SECONDS", "3600")
)
LESSON_STATE_CACHE_TTL_SECONDS = int(
    os.getenv("LESSON_STATE_CACHE_TTL_SECONDS", "86400")
)

# Shared validation patterns used by auth-facing endpoints.
PASSWORD_PATTERN = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z\d]).{8,}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ALLOWED_AUTH_REDIRECT_PATHS = {"/auth/callback", "/reset-password"}
LESSON_PASSING_PERCENT = 70


# -----------------------------------------------------------------------------
# Small general-purpose helpers
# -----------------------------------------------------------------------------

def _progress_percent(question_number: int, total_questions: int) -> int:
    """Convert quiz progress into a 0-100 percentage."""
    if total_questions <= 0:
        return 0
    return int(round((question_number / total_questions) * 100))


def _serialize_auth_user(auth_user: Any, db_user: Any = None) -> dict[str, Any]:
    """Normalize auth identities into the frontend's expected user payload."""
    if db_user:
        return {
            "id": str(db_user["id"]),
            "email": db_user["email"],
            "first_name": db_user["first_name"],
            "last_name": db_user["last_name"],
        }

    if isinstance(auth_user, dict):
        metadata = auth_user.get("user_metadata") or auth_user.get("app_metadata") or {}
        return {
            "id": str(auth_user.get("id") or ""),
            "email": auth_user.get("email"),
            "first_name": metadata.get("first_name"),
            "last_name": metadata.get("last_name"),
        }

    metadata = getattr(auth_user, "user_metadata", None) or {}
    return {
        "id": str(getattr(auth_user, "id", "")),
        "email": getattr(auth_user, "email", None),
        "first_name": metadata.get("first_name"),
        "last_name": metadata.get("last_name"),
    }


def _set_access_token_cookie(response: Response, access_token: str) -> None:
    """Persist the Supabase access token in the same cookie used by password auth."""
    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE,
        value=access_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        domain=COOKIE_DOMAIN,
        max_age=60 * 60 * 24 * 7,
        path="/",
    )


def _clear_access_token_cookie(response: Response) -> None:
    """Remove the auth cookie on logout and session expiry flows."""
    response.delete_cookie(
        key=ACCESS_TOKEN_COOKIE,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        domain=COOKIE_DOMAIN,
        path="/",
    )


def _allowed_redirect_origins() -> set[str]:
    """Build the explicit allowlist for auth callback destinations."""
    origins: set[str] = set()

    if FRONTEND_URL:
        origins.add(FRONTEND_URL.rstrip("/"))

    if APP_ENV != "production":
        origins.update(
            {
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            }
        )

    return origins


def _build_first_party_redirect_url(path: str) -> str:
    """Build a trusted frontend redirect URL for auth email flows."""
    preferred = EMAIL_CONFIRM_REDIRECT_URL.rstrip("/")
    if preferred:
        return preferred

    allowed_origins = _allowed_redirect_origins()
    if FRONTEND_URL:
        return f"{FRONTEND_URL.rstrip('/')}{path}"
    if allowed_origins:
        origin = sorted(allowed_origins)[0].rstrip("/")
        return f"{origin}{path}"

    raise HTTPException(status_code=500, detail="Frontend redirect origin is not configured")


def _validate_auth_redirect_url(raw_url: str, expected_path: str) -> str:
    """Accept only known frontend origins and first-party auth callback paths."""
    candidate = str(raw_url or "").strip()
    if not candidate:
        raise HTTPException(status_code=400, detail="Redirect URL is required")

    allowed_origins = _allowed_redirect_origins()
    if not allowed_origins:
        raise HTTPException(status_code=500, detail="Frontend redirect origin is not configured")

    match = re.match(r"^(https?://[^/?#]+)(/[^?#]*)?(?:\?[^#]*)?(?:#.*)?$", candidate)
    if not match:
        raise HTTPException(status_code=400, detail="Redirect URL is invalid")

    origin = match.group(1).rstrip("/")
    path = match.group(2) or "/"

    if origin not in allowed_origins:
        raise HTTPException(status_code=400, detail="Redirect URL origin is not allowed")
    if path not in ALLOWED_AUTH_REDIRECT_PATHS or path != expected_path:
        raise HTTPException(status_code=400, detail="Redirect URL path is not allowed")

    return f"{origin}{path}"


def _normalize_level(level: int) -> int:
    """Clamp levels to the currently supported 1-3 range."""
    return max(1, min(3, int(level)))


def _score_to_level(score: int, total_questions: int) -> int:
    """Map the final quiz score to a coarse learner level."""
    if total_questions <= 0:
        return 1

    ratio = score / total_questions
    if ratio < 0.4:
        return 1
    if ratio < 0.8:
        return 2
    return 3


def _lesson_passing_score(total_questions: int) -> int:
    """Return the minimum number of correct answers needed to pass a lesson quiz."""
    if total_questions <= 0:
        return 0
    return (total_questions * LESSON_PASSING_PERCENT + 99) // 100


def _level_to_difficulty(level: int) -> str:
    """Translate a numeric level to the persisted difficulty label."""
    return {
        1: "beginner",
        2: "intermediate",
        3: "advanced",
    }.get(_normalize_level(level), "beginner")


def _level_from_recorded_difficulty(value: Any) -> int:
    """Legacy helper for difficulty-to-level conversion."""
    if value is None:
        return 1
    if isinstance(value, int):
        return _normalize_level(value)

    normalized = str(value).strip().lower()
    if normalized in {"1", "beginner", "easy"}:
        return 1
    if normalized in {"2", "intermediate", "medium"}:
        return 2
    if normalized in {"3", "advanced", "hard"}:
        return 3
    return 1


def _topic_from_source(source: str) -> str:
    """Build a human-readable lesson title from a file path."""
    filename = os.path.basename(source)
    stem, _ = os.path.splitext(filename)
    return stem.replace("_", " ").strip()


def _clean_generated_text(value: Any) -> str:
    """Remove lightweight markdown artifacts from generated lesson text."""
    text = str(value or "").strip()
    text = text.replace("***", "").replace("**", "").replace("*", "")
    return re.sub(r"\s+", " ", text).strip()


def _polish_lesson_topic(raw_topic: str, role: str, source: str) -> str:
    """Turn file-style lesson identifiers into cleaner UI-ready titles."""
    topic = (raw_topic or "").strip()
    if not topic:
        topic = _topic_from_source(source)

    topic = topic.replace("_", " ")
    topic = re.sub(rf"^{re.escape(role)}[\s_-]+", "", topic, flags=re.IGNORECASE)
    topic = re.sub(r"[\s_-]*level\s*\d+\b", "", topic, flags=re.IGNORECASE)
    topic = re.sub(r"[\s_-]+", " ", topic).strip()

    if not topic:
        topic = "Untitled Lesson"

    words = []
    acronyms = {"ai": "AI", "ml": "ML", "llm": "LLM", "nlp": "NLP", "dl": "DL", "nn": "NN"}
    for word in topic.split():
        lowered = word.lower()
        words.append(acronyms.get(lowered, word.capitalize()))

    polished = " ".join(words).strip()
    return polished or "Untitled Lesson"


# -----------------------------------------------------------------------------
# Lesson catalog and assignment helpers
# -----------------------------------------------------------------------------

def _docs_fallback_lessons(role: str, level: int) -> List[dict]:
    """Fallback lesson list derived directly from the docs folder."""
    role_dir = DOCS_FOLDER / role
    if not role_dir.exists():
        return []

    lessons: List[dict] = []
    level_marker = f"level{level}"

    for path in sorted(role_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".pdf"}:
            continue
        if level_marker not in path.stem.lower():
            continue

        lessons.append(
            {
                "source": str(path),
                "topic": _topic_from_source(path.name),
                "format": path.suffix.lstrip("."),
                "level": level,
                "role": role,
                "chunk_count": 0,
                "completed_chunks": 0,
                "status": "assigned",
            }
        )

    return lessons


async def _list_lesson_candidates(
    db: asyncpg.Connection, role: str
) -> List[dict]:
    """Group chunk rows into lesson candidates for the lesson recommender."""
    # The recommender chooses whole lessons, so raw chunk rows are grouped by
    # lesson source while keeping the chunk IDs attached for later persistence.
    rows = await db.fetch(
        """
        SELECT
            lc.metadata->>'source' AS source,
            COALESCE(lc.metadata->>'topic', lc.chunk_id, 'Untitled lesson') AS topic,
            COALESCE(lc.metadata->>'format', 'txt') AS format,
            COALESCE(lc.metadata->>'role', 'student') AS role,
            CASE
                WHEN COALESCE(lc.metadata->>'level', '') ~ '^[0-9]+$'
                    THEN (lc.metadata->>'level')::int
                ELSE 1
            END AS level,
            COUNT(*)::int AS chunk_count,
            array_agg(lc.id) AS chunk_ids,
            LEFT(MIN(lc.content), 500) AS sample_content
        FROM learning_chunks lc
        WHERE COALESCE(lc.metadata->>'role', 'student') = $1
        GROUP BY
            lc.metadata->>'source',
            COALESCE(lc.metadata->>'topic', lc.chunk_id, 'Untitled lesson'),
            COALESCE(lc.metadata->>'format', 'txt'),
            COALESCE(lc.metadata->>'role', 'student'),
            CASE
                WHEN COALESCE(lc.metadata->>'level', '') ~ '^[0-9]+$'
                    THEN (lc.metadata->>'level')::int
                ELSE 1
            END
        ORDER BY
            CASE
                WHEN COALESCE(lc.metadata->>'level', '') ~ '^[0-9]+$'
                    THEN (lc.metadata->>'level')::int
                ELSE 1
            END,
            COALESCE(lc.metadata->>'topic', lc.chunk_id, 'Untitled lesson')
        """,
        role,
    )

    candidates: List[dict] = []
    for row in rows:
        polished_topic = _polish_lesson_topic(
            str(row["topic"] or ""),
            str(row["role"] or role),
            str(row["source"] or ""),
        )
        candidates.append(
            {
                "source": row["source"],
                "topic": polished_topic,
                "format": row["format"],
                "role": row["role"],
                "level": _normalize_level(row["level"]),
                "chunk_count": row["chunk_count"],
                "chunk_ids": [str(chunk_id) for chunk_id in (row["chunk_ids"] or [])],
                "sample_content": row["sample_content"] or "",
            }
        )
    return candidates


async def _fetch_assigned_lessons(
    db: asyncpg.Connection, user_id: str, role: str
) -> List[dict]:
    """Return the lessons currently assigned to a user for the selected role."""
    # Lesson cards in the UI are reconstructed from per-chunk assignment rows.
    rows = await db.fetch(
        """
        SELECT
            lc.metadata->>'source' AS source,
            COALESCE(lc.metadata->>'topic', lc.chunk_id, 'Untitled lesson') AS topic,
            COALESCE(lc.metadata->>'format', 'txt') AS format,
            COALESCE(lc.metadata->>'role', 'student') AS role,
            CASE
                WHEN COALESCE(lc.metadata->>'level', '') ~ '^[0-9]+$'
                    THEN (lc.metadata->>'level')::int
                ELSE 1
            END AS level,
            COUNT(*)::int AS chunk_count,
            COUNT(CASE WHEN upl.status = 'completed' THEN 1 END)::int AS completed_chunks,
            BOOL_OR(upl.status = 'in_progress') AS has_in_progress,
            BOOL_OR(upl.status = 'assigned') AS has_assigned
        FROM user_progress_lessons upl
        JOIN learning_chunks lc
            ON lc.id = upl.chunk_id
        WHERE upl.user_id = $1
          AND COALESCE(lc.metadata->>'role', 'student') = $2
        GROUP BY
            lc.metadata->>'source',
            COALESCE(lc.metadata->>'topic', lc.chunk_id, 'Untitled lesson'),
            COALESCE(lc.metadata->>'format', 'txt'),
            COALESCE(lc.metadata->>'role', 'student'),
            CASE
                WHEN COALESCE(lc.metadata->>'level', '') ~ '^[0-9]+$'
                    THEN (lc.metadata->>'level')::int
                ELSE 1
            END
        ORDER BY
            CASE
                WHEN COALESCE(lc.metadata->>'level', '') ~ '^[0-9]+$'
                    THEN (lc.metadata->>'level')::int
                ELSE 1
            END,
            COALESCE(lc.metadata->>'topic', lc.chunk_id, 'Untitled lesson')
        """,
        user_id,
        role,
    )

    lessons: List[dict] = []
    for row in rows:
        chunk_count = row["chunk_count"] or 0
        completed_chunks = row["completed_chunks"] or 0
        polished_topic = _polish_lesson_topic(
            str(row["topic"] or ""),
            str(row["role"] or "student"),
            str(row["source"] or ""),
        )

        status = "assigned"
        if chunk_count > 0 and completed_chunks >= chunk_count:
            status = "completed"
        elif row["has_in_progress"] or completed_chunks > 0:
            status = "in_progress"

        lessons.append(
            {
                "source": row["source"],
                "topic": polished_topic,
                "format": row["format"],
                "level": _normalize_level(row["level"]),
                "role": row["role"],
                "chunk_count": chunk_count,
                "completed_chunks": completed_chunks,
                "status": status,
            }
        )

        lessons[-1]["progress"] = (
            _lesson_progress_from_cached_state(user_id, lessons[-1])
            if _lesson_progress_from_cached_state(user_id, lessons[-1]) is not None
            else _lesson_progress_value(lessons[-1])
        )

    return lessons


def _select_candidate_pool_for_level(
    candidates: List[dict[str, Any]], target_level: int
) -> List[dict[str, Any]]:
    """Prefer lessons at the learner's exact normalized level before widening."""
    normalized_target = _normalize_level(target_level)
    exact_matches = [
        candidate
        for candidate in candidates
        if _normalize_level(candidate.get("level", 1)) == normalized_target
    ]
    if exact_matches:
        return exact_matches

    if not candidates:
        return []

    closest_distance = min(
        abs(_normalize_level(candidate.get("level", 1)) - normalized_target)
        for candidate in candidates
    )
    return [
        candidate
        for candidate in candidates
        if abs(_normalize_level(candidate.get("level", 1)) - normalized_target)
        == closest_distance
    ]


def _lesson_storyboard_cache_key(user_id: str, source: str) -> str:
    """Cache generated lesson slide-books per user and lesson source."""
    source_part = uuid5(NAMESPACE_DNS, source).hex
    return f"lesson:storyboard:{user_id}:{source_part}"


def _lesson_state_cache_key(user_id: str, source: str) -> str:
    """Cache the learner's in-progress lesson page state."""
    source_part = uuid5(NAMESPACE_DNS, source).hex
    return f"lesson:state:{user_id}:{source_part}"


def _clear_user_cached_learning_state(user_id: str) -> None:
    """Remove cached lesson state and any active questionnaire session for a user."""
    active_quiz_id = get_active_session_id(user_id)
    if active_quiz_id:
        redis_client.delete(active_quiz_id)
    clear_active_session(user_id)

    for pattern in (
        f"lesson:storyboard:{user_id}:*",
        f"lesson:state:{user_id}:*",
    ):
        for key in redis_client.scan_iter(match=pattern):
            redis_client.delete(key)


async def _fetch_lesson_detail_context(
    db: asyncpg.Connection, user_id: str, source: str
) -> dict[str, Any]:
    """Collect the chunk context and assignment state for one lesson."""
    profile_row = await db.fetchrow(
        """
        SELECT
            up.id,
            COALESCE(up.role, 'student') AS role,
            COALESCE(up.difficulty, 'beginner') AS difficulty,
            COALESCE(u.email, '') AS email,
            COALESCE(u.first_name, '') AS first_name,
            COALESCE(u.last_name, '') AS last_name
        FROM user_profiles up
        LEFT JOIN users u
            ON u.id = up.id
        WHERE up.id = $1
        """,
        user_id,
    )
    if not profile_row:
        raise HTTPException(status_code=404, detail="User profile not found")

    lesson_row = await db.fetchrow(
        """
        SELECT
            COALESCE(lc.metadata->>'source', '') AS source,
            COALESCE(lc.metadata->>'topic', lc.chunk_id, 'Untitled lesson') AS topic,
            COALESCE(lc.metadata->>'role', 'student') AS role,
            CASE
                WHEN COALESCE(lc.metadata->>'level', '') ~ '^[0-9]+$'
                    THEN (lc.metadata->>'level')::int
                ELSE 1
            END AS level,
            COUNT(*)::int AS chunk_count,
            COUNT(CASE WHEN upl.status = 'completed' THEN 1 END)::int AS completed_chunks,
            BOOL_OR(upl.status = 'in_progress') AS has_in_progress,
            STRING_AGG(lc.content, E'\n\n---\n\n' ORDER BY lc.chunk_id) AS lesson_text
        FROM learning_chunks lc
        JOIN user_progress_lessons upl
            ON upl.chunk_id = lc.id
        WHERE upl.user_id = $1
          AND COALESCE(lc.metadata->>'source', '') = $2
        GROUP BY
            COALESCE(lc.metadata->>'source', ''),
            COALESCE(lc.metadata->>'topic', lc.chunk_id, 'Untitled lesson'),
            COALESCE(lc.metadata->>'role', 'student'),
            CASE
                WHEN COALESCE(lc.metadata->>'level', '') ~ '^[0-9]+$'
                    THEN (lc.metadata->>'level')::int
                ELSE 1
            END
        """,
        user_id,
        source,
    )
    if not lesson_row:
        raise HTTPException(status_code=404, detail="Lesson not found for this user")

    # lesson_text is the stitched source material for the selected lesson,
    # assembled from the user's assigned chunks in chunk_id order.

    progress_row = await db.fetchrow(
        """
        SELECT current_level
        FROM user_progress
        WHERE user_id = $1
        ORDER BY last_accessed DESC
        LIMIT 1
        """,
        user_id,
    )
    level = _normalize_level(
        progress_row["current_level"] if progress_row else lesson_row["level"]
    )

    answer_rows = await _fetch_completed_lessons(db, user_id)
    lesson = {
        "source": lesson_row["source"],
        "topic": _polish_lesson_topic(
            str(lesson_row["topic"] or ""),
            str(lesson_row["role"] or "student"),
            str(lesson_row["source"] or ""),
        ),
        "role": lesson_row["role"],
        "level": _normalize_level(lesson_row["level"]),
        "chunk_count": lesson_row["chunk_count"] or 0,
        "completed_chunks": lesson_row["completed_chunks"] or 0,
    }
    lesson["status"] = "completed" if lesson["chunk_count"] > 0 and lesson["completed_chunks"] >= lesson["chunk_count"] else "in_progress" if lesson_row["has_in_progress"] or lesson["completed_chunks"] > 0 else "assigned"

    assigned_lessons = await _fetch_assigned_lessons(db, user_id, profile_row["role"])
    next_lesson = next(
        (
            item
            for item in assigned_lessons
            if item.get("source") != source and item.get("status") != "completed"
        ),
        None,
    )

    profile = {
        "id": str(profile_row["id"]),
        "role": profile_row["role"],
        "difficulty": profile_row["difficulty"],
        "email": profile_row["email"],
        "first_name": profile_row["first_name"],
        "last_name": profile_row["last_name"],
        "target_level": level,
        "previously_completed_lessons": answer_rows,
    }

    return {
        "profile": profile,
        "lesson": lesson,
        "next_lesson": next_lesson,
        "lesson_text": str(lesson_row["lesson_text"] or ""),
        "estimated_minutes": max(10, int((lesson["chunk_count"] or 1) * 5)),
        "progress": _lesson_progress_value(lesson),
    }


async def _assign_lessons_to_user(
    db: asyncpg.Connection, user_id: str, profile: dict[str, Any], level: int
) -> List[dict]:
    """Ask the LLM for lesson recommendations and persist the selected chunks."""
    role = str(profile.get("role") or "student")
    level = _normalize_level(level)
    candidates = await _list_lesson_candidates(db, role)
    candidate_pool = _select_candidate_pool_for_level(candidates, level)

    if not candidates:
        return _docs_fallback_lessons(role, level)
    if not candidate_pool:
        return _docs_fallback_lessons(role, level)

    recommendation = await recommend_lessons(profile, candidate_pool, level)
    selected_sources = recommendation.get("selected_sources") or []

    if not selected_sources:
        # Deterministic fallback: keep assignment moving if the LLM cannot
        # return usable lesson sources.
        selected_sources = [
            candidate["source"] for candidate in candidate_pool
        ][:4]

    source_to_candidate = {
        candidate["source"]: candidate
        for candidate in candidate_pool
        if candidate.get("source")
    }
    selected_chunk_ids: List[str] = []
    for source in selected_sources:
        candidate = source_to_candidate.get(source)
        if candidate:
            # The recommender outputs lesson sources, and those sources are
            # expanded back into chunk IDs here for database assignment.
            selected_chunk_ids.extend(candidate.get("chunk_ids", []))

    if not selected_chunk_ids:
        return _docs_fallback_lessons(role, level)

    await db.execute(
        """
        DELETE FROM user_progress_lessons upl
        USING learning_chunks lc
        WHERE upl.user_id = $1
          AND upl.chunk_id = lc.id
          AND upl.status = 'assigned'
          AND COALESCE(lc.metadata->>'role', 'student') = $2
        """,
        user_id,
        role,
    )

    await db.execute(
        """
        INSERT INTO user_progress_lessons (user_id, chunk_id, status)
        SELECT $1, lc.id, 'assigned'
        FROM learning_chunks lc
        WHERE lc.id = ANY($2::uuid[])
          AND NOT EXISTS (
                SELECT 1
                FROM user_progress_lessons upl
                WHERE upl.user_id = $1
                  AND upl.chunk_id = lc.id
          )
        """,
        user_id,
        selected_chunk_ids,
    )
    lessons = await _fetch_assigned_lessons(db, user_id, role)
    return lessons if lessons else _docs_fallback_lessons(role, level)


async def _save_user_progress(
    db: asyncpg.Connection, user_id: str, level: int, status: str = "in_progress"
) -> None:
    """Create or update the user's overall progress row."""
    level = _normalize_level(level)
    updated = await db.fetchval(
        """
        UPDATE user_progress
        SET current_level = $2,
            status = $3,
            last_accessed = now()
        WHERE user_id = $1
        RETURNING id
        """,
        user_id,
        level,
        status,
    )
    if updated:
        return

    await db.execute(
        """
        INSERT INTO user_progress (user_id, current_level, status)
        VALUES ($1, $2, $3)
        """,
        user_id,
        level,
        status,
    )


async def _save_user_profile_difficulty(
    db: asyncpg.Connection, user_id: str, level: int
) -> None:
    """Keep the user profile difficulty label aligned with the assigned level."""
    await db.execute(
        """
        UPDATE user_profiles
        SET difficulty = $2
        WHERE id = $1
        """,
        user_id,
        _level_to_difficulty(level),
    )


async def _save_user_score(
    db: asyncpg.Connection,
    user_id: str,
    score: int,
    chunk_id: Optional[str] = None,
) -> None:
    """Persist a quiz score entry."""
    await db.execute(
        """
        INSERT INTO user_scores (user_id, chunk_id, score)
        VALUES ($1, $2, $3)
        """,
        user_id,
        chunk_id,
        score,
    )


async def _save_lesson_quiz_score(
    db: asyncpg.Connection,
    user_id: str,
    source: str,
    quiz_score: int | None,
    total_questions: int | None,
) -> None:
    """Persist a lesson-end quiz score as a percentage tied to the lesson."""
    if quiz_score is None or total_questions is None or total_questions <= 0:
        return

    normalized_score = max(0, min(100, int(round((quiz_score / total_questions) * 100))))
    chunk_id = await db.fetchval(
        """
        SELECT lc.id
        FROM learning_chunks lc
        WHERE COALESCE(lc.metadata->>'source', '') = $1
        ORDER BY lc.id
        LIMIT 1
        """,
        source,
    )
    await _save_user_score(db, user_id, normalized_score, str(chunk_id) if chunk_id else None)


async def _count_completed_lessons_for_role(
    db: asyncpg.Connection, user_id: str, role: str
) -> int:
    """Count distinct completed lessons for the current learning path."""
    return int(
        await db.fetchval(
            """
            SELECT COUNT(DISTINCT COALESCE(lc.metadata->>'source', ''))
            FROM user_progress_lessons upl
            JOIN learning_chunks lc
                ON lc.id = upl.chunk_id
            WHERE upl.user_id = $1
              AND upl.status = 'completed'
              AND COALESCE(lc.metadata->>'role', 'student') = $2
            """,
            user_id,
            role,
        )
        or 0
    )


def _level_from_completed_lessons(current_level: int, completed_lessons: int) -> int:
    """Advance learners one level for each 10 completed lessons, without downgrades."""
    earned_level = min(3, 1 + (completed_lessons // 10))
    return max(_normalize_level(current_level), earned_level)


async def _refresh_lessons_after_completion(
    db: asyncpg.Connection,
    user_id: str,
    role: str,
    current_level: int,
) -> tuple[int, List[dict]]:
    """Persist post-lesson progression and ensure a next lesson is available."""
    completed_lessons = await _count_completed_lessons_for_role(db, user_id, role)
    updated_level = _level_from_completed_lessons(current_level, completed_lessons)

    if updated_level != _normalize_level(current_level):
        # On level-up, clear still-unstarted assigned lessons for this role so
        # the next recommendation pass can refill them at the new level.
        await _save_user_progress(db, user_id, updated_level, status="in_progress")
        await _save_user_profile_difficulty(db, user_id, updated_level)

        await db.execute(
            """
            DELETE FROM user_progress_lessons upl
            USING learning_chunks lc
            WHERE upl.user_id = $1
              AND upl.chunk_id = lc.id
              AND upl.status = 'assigned'
              AND COALESCE(lc.metadata->>'role', 'student') = $2
            """,
            user_id,
            role,
        )

    lessons = await _fetch_assigned_lessons(db, user_id, role)
    open_lessons = [lesson for lesson in lessons if lesson.get("status") != "completed"]

    if len(open_lessons) < 1:
        profile_row = await db.fetchrow(
            """
            SELECT
                up.id,
                COALESCE(up.role, 'student') AS role,
                COALESCE(up.difficulty, 'beginner') AS difficulty,
                COALESCE(u.email, '') AS email,
                COALESCE(u.first_name, '') AS first_name,
                COALESCE(u.last_name, '') AS last_name
            FROM user_profiles up
            LEFT JOIN users u
                ON u.id = up.id
            WHERE up.id = $1
            """,
            user_id,
        )
        if profile_row:
            recommendation_profile = await _build_lesson_recommendation_profile(
                db,
                {
                    "id": str(profile_row["id"]),
                    "role": profile_row["role"],
                    "difficulty": profile_row["difficulty"],
                    "email": profile_row["email"],
                    "first_name": profile_row["first_name"],
                    "last_name": profile_row["last_name"],
                },
                updated_level,
            )
            lessons = await _assign_lessons_to_user(
                db, user_id, recommendation_profile, updated_level
            )

    return updated_level, lessons


def _extract_weak_areas(answer_history: List[dict[str, Any]]) -> List[str]:
    """Derive a short list of weak areas from missed quiz questions."""
    weak_areas: List[str] = []
    for answer in answer_history:
        if answer.get("correct"):
            continue

        question = str(answer.get("question") or "").strip()
        explanation = str(answer.get("explanation") or "").strip()
        summary = question if question else explanation
        if summary:
            weak_areas.append(summary[:160])

    return weak_areas[:5]


async def _fetch_completed_lessons(
    db: asyncpg.Connection, user_id: str
) -> List[dict[str, Any]]:
    """Return previously completed lessons for the learner profile."""
    rows = await db.fetch(
        """
        SELECT
            lc.metadata->>'source' AS source,
            COALESCE(lc.metadata->>'topic', lc.chunk_id, 'Untitled lesson') AS topic,
            COALESCE(lc.metadata->>'role', 'student') AS role,
            CASE
                WHEN COALESCE(lc.metadata->>'level', '') ~ '^[0-9]+$'
                    THEN (lc.metadata->>'level')::int
                ELSE 1
            END AS level
        FROM user_progress_lessons upl
        JOIN learning_chunks lc
            ON lc.id = upl.chunk_id
        WHERE upl.user_id = $1
          AND upl.status = 'completed'
        GROUP BY
            lc.metadata->>'source',
            COALESCE(lc.metadata->>'topic', lc.chunk_id, 'Untitled lesson'),
            COALESCE(lc.metadata->>'role', 'student'),
            CASE
                WHEN COALESCE(lc.metadata->>'level', '') ~ '^[0-9]+$'
                    THEN (lc.metadata->>'level')::int
                ELSE 1
            END
        ORDER BY
            CASE
                WHEN COALESCE(lc.metadata->>'level', '') ~ '^[0-9]+$'
                    THEN (lc.metadata->>'level')::int
                ELSE 1
            END,
            COALESCE(lc.metadata->>'topic', lc.chunk_id, 'Untitled lesson')
        """,
        user_id,
    )

    return [
        {
            "source": row["source"],
            "topic": _polish_lesson_topic(
                str(row["topic"] or ""),
                str(row["role"] or "student"),
                str(row["source"] or ""),
            ),
            "role": row["role"],
            "level": _normalize_level(row["level"]),
        }
        for row in rows
    ]


async def _build_lesson_recommendation_profile(
    db: asyncpg.Connection,
    base_profile: dict[str, Any],
    level: int,
    score: Optional[int] = None,
    total_questions: Optional[int] = None,
    answer_history: Optional[List[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Enrich the learner profile with quiz and completion context for the LLM."""
    completed_lessons = await _fetch_completed_lessons(
        db, str(base_profile.get("id") or "")
    )
    answers = answer_history or []

    enriched_profile = dict(base_profile)
    enriched_profile.update(
        {
            "target_level": _normalize_level(level),
            "final_questionnaire_score": score,
            "total_questions": total_questions,
            "answer_history": answers,
            "weak_areas": _extract_weak_areas(answers),
            "previously_completed_lessons": completed_lessons,
        }
    )
    return enriched_profile


def _level_label(level: int) -> str:
    """Format numeric levels with the same labels used in the UI."""
    return {
        1: "Beginner",
        2: "Intermediate",
        3: "Advanced",
    }.get(_normalize_level(level), "Beginner")


def _lesson_duration_label(chunk_count: int) -> str:
    """Estimate lesson duration from the stored chunk count."""
    if chunk_count <= 0:
        return "Self-paced"
    return f"{chunk_count * 5} min"


def _lesson_progress_value(lesson: dict[str, Any]) -> int:
    """Convert lesson chunk completion into a 0-100 progress value."""
    chunk_count = int(lesson.get("chunk_count") or 0)
    completed_chunks = int(lesson.get("completed_chunks") or 0)

    if chunk_count <= 0:
        status = str(lesson.get("status") or "assigned")
        if status == "completed":
            return 100
        if status == "in_progress":
            return 50
        return 0

    return max(0, min(100, int(round((completed_chunks / chunk_count) * 100))))


def _lesson_progress_from_cached_state(user_id: str, lesson: dict[str, Any]) -> int | None:
    """Prefer saved slide-book progress when lesson state is available."""
    source = str(lesson.get("source") or "").strip()
    if not source:
        return None

    cached_state = redis_client.get(_lesson_state_cache_key(user_id, source))
    if not cached_state:
        return None

    try:
        state_payload = json.loads(cached_state)
    except Exception:
        return None

    phase = str(state_payload.get("phase") or "story")
    active_slide = max(0, int(state_payload.get("active_slide", 0)))
    quiz_submitted = bool(state_payload.get("quiz_submitted", False))

    if phase == "next" or str(lesson.get("status") or "") == "completed":
        return 100
    if phase == "quiz":
        return 100 if quiz_submitted else 90

    cached_storyboard = redis_client.get(_lesson_storyboard_cache_key(user_id, source))
    if cached_storyboard:
        try:
            storyboard_payload = json.loads(cached_storyboard)
            slides = storyboard_payload.get("slides") or []
            slide_count = len(slides) if isinstance(slides, list) else 0
            if slide_count > 0:
                return max(0, min(100, int(round(((active_slide + 1) / slide_count) * 100))))
        except Exception:
            pass

    if str(lesson.get("status") or "") == "in_progress":
        return 50

    return None


def _daily_goal_message(progress: int) -> str:
    """Keep the dashboard goal copy friendly and status-aware."""
    if progress >= 100:
        return "Daily goal complete. Nice work."
    if progress >= 50:
        return "Great work! Keep going."
    if progress > 0:
        return "Good start. Keep the momentum going."
    return "Start one lesson today to build momentum."


def _coerce_score_percent(score: int | float | None) -> int:
    """Normalize stored score values into a UI-friendly percentage."""
    if score is None:
        return 0

    numeric_score = float(score)
    if numeric_score <= QUIZ_TOTAL_QUESTIONS:
        return int(round((numeric_score / QUIZ_TOTAL_QUESTIONS) * 100))

    return max(0, min(100, int(round(numeric_score))))


def _display_name_from_identity(
    first_name: Any, last_name: Any, email: Any
) -> str:
    """Build a user-facing name without leaking awkward legacy placeholders."""
    raw_first = str(first_name or "").strip()
    raw_last = str(last_name or "").strip()

    blocked = {"learner", "user", "learner user", "-", "member"}
    normalized_first = "" if raw_first.lower() in blocked else raw_first
    normalized_last = "" if raw_last.lower() in blocked else raw_last

    full_name = " ".join(
        part for part in [normalized_first, normalized_last] if part
    ).strip()
    if full_name:
        return full_name

    email_value = str(email or "").strip()
    if email_value and "@" in email_value:
        email_prefix = email_value.split("@", 1)[0].strip("._- ")
        if email_prefix:
            return email_prefix

    return "LearningOS Member"


def _extract_access_token(
    authorization: str | None, access_token: str | None
) -> str | None:
    """Extract a bearer token from either the auth header or cookie."""
    if authorization:
        return authorization.replace("Bearer ", "").strip()
    if access_token:
        return access_token.strip()
    return None


async def _upsert_local_user_from_auth(
    db: asyncpg.Connection, auth_user: Any
) -> Optional[dict[str, Any]]:
    """Ensure the local users table has a row for the authenticated user."""
    if auth_user is None:
        return None

    if isinstance(auth_user, dict):
        user_id = auth_user.get("id")
        email = auth_user.get("email")
        metadata = auth_user.get("user_metadata") or auth_user.get("app_metadata") or {}
    else:
        user_id = getattr(auth_user, "id", None)
        email = getattr(auth_user, "email", None)
        metadata = getattr(auth_user, "user_metadata", None) or {}

    if not user_id or not email:
        return None

    email_name = str(email).split("@", 1)[0].strip() or "Member"
    first_name_value = str(metadata.get("first_name") or "").strip() or email_name
    last_name_value = str(metadata.get("last_name") or "").strip() or ""
    first_name = normalize_name(first_name_value, "First name")
    last_name = normalize_name(last_name_value or "-", "Last name")
    agree_terms = bool(metadata.get("agree_terms", False))

    row = await db.fetchrow(
        """
        INSERT INTO users (id, email, first_name, last_name, agree_terms)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (id)
        DO UPDATE SET
            email = EXCLUDED.email,
            first_name = COALESCE(NULLIF(users.first_name, ''), EXCLUDED.first_name),
            last_name = CASE
                WHEN NULLIF(users.last_name, '') IS NOT NULL THEN users.last_name
                WHEN EXCLUDED.last_name = '-' THEN ''
                ELSE EXCLUDED.last_name
            END
        RETURNING id, email, first_name, last_name, created_at
        """,
        user_id,
        email,
        first_name,
        last_name,
        agree_terms,
    )
    return dict(row) if row else None


async def _build_dashboard_payload(
    db: asyncpg.Connection, user_id: str
) -> dict[str, Any]:
    """Assemble the live dashboard payload from the learner's current state."""
    profile_row = await db.fetchrow(
        """
        SELECT
            up.role,
            up.difficulty,
            COALESCE(u.email, '') AS email,
            COALESCE(u.first_name, '') AS first_name,
            COALESCE(u.last_name, '') AS last_name
        FROM user_profiles up
        LEFT JOIN users u
            ON u.id = up.id
        WHERE up.id = $1
        """,
        user_id,
    )
    if not profile_row:
        raise HTTPException(status_code=404, detail="User profile not found")

    progress_row = await db.fetchrow(
        """
        SELECT current_level
        FROM user_progress
        WHERE user_id = $1
        ORDER BY last_accessed DESC
        LIMIT 1
        """,
        user_id,
    )
    if not progress_row:
        raise HTTPException(status_code=404, detail="Questionnaire not completed yet")

    role = str(profile_row["role"] or "student")
    level = _normalize_level(progress_row["current_level"])
    lessons = await _fetch_assigned_lessons(db, user_id, role)

    if not lessons:
        recommendation_profile = await _build_lesson_recommendation_profile(
            db,
            {
                "id": user_id,
                "role": role,
                "difficulty": profile_row["difficulty"],
                "first_name": profile_row["first_name"],
                "last_name": profile_row["last_name"],
            },
            level,
        )
        lessons = await _assign_lessons_to_user(
            db, user_id, recommendation_profile, level
        )

    total_lessons = len(lessons)
    completed_lessons = sum(1 for lesson in lessons if lesson["status"] == "completed")
    in_progress_lessons = sum(
        1 for lesson in lessons if lesson["status"] == "in_progress"
    )

    current_lesson = next(
        (lesson for lesson in lessons if lesson["status"] == "in_progress"),
        None,
    ) or next((lesson for lesson in lessons if lesson["status"] == "assigned"), None)
    if current_lesson is None and lessons:
        current_lesson = lessons[0]

    if current_lesson is None:
        current_lesson = {
            "topic": f"{role.title()} Learning Path",
            "role": role,
            "level": level,
            "chunk_count": 0,
            "completed_chunks": 0,
            "status": "assigned",
        }

    current_progress = int(
        current_lesson.get("progress")
        if current_lesson.get("progress") is not None
        else _lesson_progress_value(current_lesson)
    )
    current_chunk_count = int(current_lesson.get("chunk_count") or 0)
    current_completed_chunks = int(current_lesson.get("completed_chunks") or 0)

    score_rows = await db.fetch(
        """
        SELECT score, completed_at
        FROM user_scores
        WHERE user_id = $1
        ORDER BY completed_at DESC
        """,
        user_id,
    )

    average_score_percent = 0
    last_quiz_label = "No quizzes yet"
    if score_rows:
        average_score_percent = int(
            round(
                sum(_coerce_score_percent(row["score"]) for row in score_rows)
                / len(score_rows)
            )
        )
        last_score = score_rows[0]["score"]
        if last_score is None:
            last_quiz_label = "No quizzes yet"
        elif float(last_score) <= QUIZ_TOTAL_QUESTIONS:
            last_quiz_label = f"{int(last_score)}/{QUIZ_TOTAL_QUESTIONS}"
        else:
            last_quiz_label = f"{int(round(float(last_score)))}%"

    today_activity_count = await db.fetchval(
        """
        SELECT COUNT(*)
        FROM user_progress_lessons
        WHERE user_id = $1
          AND DATE(last_accessed AT TIME ZONE 'UTC') = CURRENT_DATE
        """,
        user_id,
    )
    learned_minutes_today = int(today_activity_count or 0) * 5

    daily_goal_progress = min(100, learned_minutes_today)

    started_rows = await db.fetch(
        """
        SELECT
            lc.metadata->>'source' AS source,
            COALESCE(lc.metadata->>'topic', lc.chunk_id, 'Untitled lesson') AS topic,
            COALESCE(lc.metadata->>'role', 'student') AS role,
            MAX(upl.last_accessed) AS activity_at
        FROM user_progress_lessons upl
        JOIN learning_chunks lc
            ON lc.id = upl.chunk_id
        WHERE upl.user_id = $1
          AND upl.status = 'in_progress'
        GROUP BY
            lc.metadata->>'source',
            COALESCE(lc.metadata->>'topic', lc.chunk_id, 'Untitled lesson'),
            COALESCE(lc.metadata->>'role', 'student')
        ORDER BY activity_at DESC
        LIMIT 3
        """,
        user_id,
    )
    completed_rows = await db.fetch(
        """
        SELECT
            lc.metadata->>'source' AS source,
            COALESCE(lc.metadata->>'topic', lc.chunk_id, 'Untitled lesson') AS topic,
            COALESCE(lc.metadata->>'role', 'student') AS role,
            MAX(upl.last_accessed) AS activity_at
        FROM user_progress_lessons upl
        JOIN learning_chunks lc
            ON lc.id = upl.chunk_id
        WHERE upl.user_id = $1
          AND upl.status = 'completed'
        GROUP BY
            lc.metadata->>'source',
            COALESCE(lc.metadata->>'topic', lc.chunk_id, 'Untitled lesson'),
            COALESCE(lc.metadata->>'role', 'student')
        ORDER BY activity_at DESC
        LIMIT 3
        """,
        user_id,
    )

    activity_items: List[dict[str, Any]] = []
    for row in score_rows[:3]:
        score_value = row["score"]
        if score_value is None:
            continue
        if float(score_value) <= QUIZ_TOTAL_QUESTIONS:
            subtitle = f"Score: {int(score_value)}/{QUIZ_TOTAL_QUESTIONS}"
        else:
            subtitle = f"Score: {int(round(float(score_value)))}%"
        activity_items.append(
            {
                "type": "quiz_completed",
                "title": "Completed Quiz",
                "subtitle": subtitle,
                "activity_at": row["completed_at"],
            }
        )

    for row in started_rows:
        activity_items.append(
            {
                "type": "module_started",
                "title": "Started Lesson",
                "subtitle": _polish_lesson_topic(
                    str(row["topic"] or ""),
                    str(row["role"] or "student"),
                    str(row["source"] or ""),
                ),
                "activity_at": row["activity_at"],
            }
        )

    for row in completed_rows:
        activity_items.append(
            {
                "type": "module_completed",
                "title": "Completed Lesson",
                "subtitle": _polish_lesson_topic(
                    str(row["topic"] or ""),
                    str(row["role"] or "student"),
                    str(row["source"] or ""),
                ),
                "activity_at": row["activity_at"],
            }
        )

    activity_items.sort(
        key=lambda item: item.get("activity_at")
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    recent_activity = [
        {
            "type": item["type"],
            "title": item["title"],
            "subtitle": item["subtitle"],
        }
        for item in activity_items[:3]
    ]

    first_name = str(profile_row["first_name"] or "").strip()
    last_name = str(profile_row["last_name"] or "").strip()
    display_name = _display_name_from_identity(
        first_name,
        last_name,
        profile_row["email"],
    )

    recommendations = [
        {
            "title": lesson["topic"],
            "subtitle": f"Recommended for your {role} path",
            "duration": _lesson_duration_label(int(lesson.get("chunk_count") or 0)),
            "level": _level_label(int(lesson.get("level") or level)),
        }
        for lesson in lessons
        if lesson.get("topic") != current_lesson.get("topic")
        and lesson.get("status") != "completed"
    ][:2]

    return {
        "user": {
            "name": display_name,
            "learned_minutes_today": learned_minutes_today,
        },
        "stats": [
            {
                "title": "Lessons complete",
                "value": f"{completed_lessons}/{total_lessons}",
            },
            {
                "title": "Average quiz score",
                "value": f"{average_score_percent}%",
            },
            {
                "title": "Last quiz result",
                "value": last_quiz_label,
            },
            {
                "title": "Current level",
                "value": _level_label(level),
            },
        ],
        "current_lesson": {
            "source": str(current_lesson.get("source") or ""),
            "title": str(current_lesson.get("topic") or "Current lesson"),
            "description": (
                f"Continue your {role} path with this "
                f"{_level_label(int(current_lesson.get('level') or level)).lower()} lesson."
            ),
            "progress": current_progress,
            "completed_label": f"{current_progress}% Completed",
            "modules_label": (
                f"{current_completed_chunks}/{current_chunk_count} Modules"
                if current_chunk_count > 0
                else "Self-paced"
            ),
            "status": str(current_lesson.get("status") or "assigned").replace("_", " ").upper(),
        },
        "daily_goal": {
            "progress": daily_goal_progress,
            "message": _daily_goal_message(daily_goal_progress),
        },
        "recommendations": recommendations,
        "recent_activity": recent_activity,
    }


async def _build_profile_payload(
    db: asyncpg.Connection, user_id: str
) -> dict[str, Any]:
    """Build the authenticated user's profile page payload."""
    row = await db.fetchrow(
        """
        SELECT
            u.id,
            u.email,
            COALESCE(u.first_name, '') AS first_name,
            COALESCE(u.last_name, '') AS last_name,
            u.created_at,
            COALESCE(up.role, 'student') AS role,
            COALESCE(up.difficulty, 'beginner') AS difficulty,
            COALESCE((
                SELECT current_level
                FROM user_progress
                WHERE user_id = $1
                ORDER BY last_accessed DESC
                LIMIT 1
            ), 1) AS current_level
        FROM users u
        LEFT JOIN user_profiles up
            ON up.id = u.id
        WHERE u.id = $1
        """,
        user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="User profile not found")

    role = str(row["role"] or "student")
    lessons: List[dict[str, Any]] = []
    has_profile_row = await db.fetchval(
        """
        SELECT EXISTS (
            SELECT 1
            FROM user_profiles
            WHERE id = $1
        )
        """,
        user_id,
    )
    if has_profile_row:
        lessons = await _fetch_assigned_lessons(db, user_id, role)

    lessons_completed = sum(1 for lesson in lessons if lesson["status"] == "completed")
    lessons_in_progress = sum(
        1 for lesson in lessons if lesson["status"] == "in_progress"
    )
    lessons_assigned = sum(1 for lesson in lessons if lesson["status"] == "assigned")

    current_lesson = next(
        (lesson for lesson in lessons if lesson["status"] == "in_progress"),
        None,
    ) or next((lesson for lesson in lessons if lesson["status"] == "assigned"), None)

    score_rows = await db.fetch(
        """
        SELECT score
        FROM user_scores
        WHERE user_id = $1
        ORDER BY completed_at DESC
        """,
        user_id,
    )

    last_quiz_score = 0
    average_quiz_score = 0
    if score_rows:
        normalized_scores = [
            _coerce_score_percent(score_row["score"]) for score_row in score_rows
        ]
        average_quiz_score = int(round(sum(normalized_scores) / len(normalized_scores)))
        last_quiz_score = normalized_scores[0]

    first_name = str(row["first_name"] or "").strip()
    last_name = str(row["last_name"] or "").strip()
    full_name = _display_name_from_identity(
        first_name,
        last_name,
        row["email"],
    )

    payload = {
        "id": str(row["id"]),
        "email": str(row["email"] or ""),
        "first_name": first_name,
        "last_name": last_name,
        "full_name": full_name,
        "role": role,
        "difficulty": str(row["difficulty"] or "beginner"),
        "current_level": _normalize_level(row["current_level"]),
        "joined_at": row["created_at"],
        "stats": {
            "lessons_completed": lessons_completed,
            "lessons_in_progress": lessons_in_progress,
            "lessons_assigned": lessons_assigned,
            "average_quiz_score": average_quiz_score,
            "last_quiz_score": last_quiz_score,
        },
        "current_lesson": None,
    }

    if current_lesson:
        payload["current_lesson"] = {
            "source": str(current_lesson.get("source") or ""),
            "title": str(current_lesson.get("topic") or "Current lesson"),
            "status": str(current_lesson.get("status") or "assigned"),
            "progress": int(
                current_lesson.get("progress")
                if current_lesson.get("progress") is not None
                else _lesson_progress_value(current_lesson)
            ),
        }

    return payload

# -----------------------------------------------------------------------------
# Input validation helpers
# -----------------------------------------------------------------------------

def normalize_email(value: str) -> str:
    """Normalize and validate a user email address."""
    email = value.strip().strip('"\'').lower()
    email = re.sub(r"[\u200B-\u200D\uFEFF]", "", email)
    if not email or not EMAIL_PATTERN.match(email):
        raise HTTPException(status_code=400, detail="Invalid email address")
    return email


def normalize_name(value: str, field_name: str = "Name") -> str:
    """Normalize a free-text name field and enforce simple limits."""
    normalized = re.sub(r"\s+", " ", value.strip())
    if not normalized:
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty")
    if len(normalized) > 120:
        raise HTTPException(status_code=400, detail=f"{field_name} is too long")
    return normalized

def validate_password_strength(password: str) -> None:
    """Enforce the current password strength policy."""
    if not PASSWORD_PATTERN.match(password):
        raise HTTPException(
            status_code=400,
            detail=(
                "Password must be at least 8 characters and include uppercase, "
                "lowercase, number, and special character."
            ),
        )
        
# -----------------------------------------------------------------------------
# Authentication and onboarding routes
# -----------------------------------------------------------------------------

@router.post("/register")
async def create_user(user: UserCreate):
    email = normalize_email(user.email)
    first_name = normalize_name(user.first_name, "First name")
    last_name = normalize_name(user.last_name, "Last name")
    validate_password_strength(user.password)
    email_confirm_redirect = _build_first_party_redirect_url("/login")

    if not user.agree_terms:
        raise HTTPException(status_code=400, detail="You must agree to the terms and conditions")

    db = await get_db()
    try:
        if BYPASS_SUPABASE_AUTH:
            user_id = str(uuid5(NAMESPACE_DNS, email))
        else:
            auth_user = supabase.auth.sign_up({
                "email": email,
                "password": user.password,
                "options": {
                    "email_redirect_to": email_confirm_redirect,
                    "data": {
                        "first_name": first_name,
                        "last_name": last_name,
                        "agree_terms": user.agree_terms,
                    }
                }
            })

            if auth_user.user is None:
                raise HTTPException(status_code=400, detail="Registration failed")

            user_id = auth_user.user.id

        try:
            row = await db.fetchrow("""
                INSERT INTO users (id, email, first_name, last_name, agree_terms)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
            """, user_id, email, first_name, last_name, user.agree_terms)
        except asyncpg.UniqueViolationError:
            raise HTTPException(status_code=409, detail="Email already exists")
        finally:
            await db.close()

        return {
            "id": row["id"],
            "email": row["email"],
            "first_name": row["first_name"],
            "last_name": row["last_name"],
            "agree_terms": row["agree_terms"],
            "detail": "Registration successful. Check your email to confirm your account."
        }

    except HTTPException:
        raise
    except Exception as e:
        error_message = str(e).lower()
        if "already registered" in error_message or "user already registered" in error_message:
            raise HTTPException(status_code=409, detail="Email already exists")
        if "rate limit" in error_message:
            raise HTTPException(
                status_code=429,
                detail="Signup is temporarily rate-limited by Supabase. Please wait and try again.",
            )
        if "for security purposes" in error_message or "you can only request this after" in error_message:
            raise HTTPException(status_code=429, detail="Signup is rate-limited by Supabase. Please wait and retry.")
        if "invalid email" in error_message or ("email" in error_message and "is invalid" in error_message):
            raise HTTPException(status_code=400, detail="Invalid email address")
        if "invalid api key" in error_message or "apikey" in error_message:
            raise HTTPException(status_code=500, detail="Supabase API key is misconfigured on the server")
        if "column" in error_message and "does not exist" in error_message:
            raise HTTPException(status_code=500, detail="Database schema mismatch for users table")

        print(f"Register error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")
@router.get("/check-email")
async def check_email_exists(email: str = Query(..., min_length=3)):
    normalized_email = normalize_email(email)

    db = await get_db()
    exists = await db.fetchval("""
        SELECT EXISTS (
            SELECT 1
            FROM users
            WHERE LOWER(email) = $1
        )
    """, normalized_email)
    await db.close()

    return {"available": not exists}


@router.post("/auth/oauth/start")
async def start_oauth_login(payload: OAuthStartRequest):
    if BYPASS_SUPABASE_AUTH:
        raise HTTPException(
            status_code=501,
            detail="OAuth login is not available when BYPASS_SUPABASE_AUTH is enabled",
        )

    redirect_to = _validate_auth_redirect_url(payload.redirect_to, "/auth/callback")

    provider_map = {
        "google": "google",
    }
    provider = provider_map[payload.provider]
    query_params = (
        {
            "access_type": "offline",
            "prompt": "consent",
        }
        if payload.provider == "google"
        else {}
    )

    try:
        oauth_response = supabase.auth.sign_in_with_oauth(
            {
                "provider": provider,
                "options": {
                    "redirect_to": redirect_to,
                    "query_params": query_params,
                },
            }
        )
    except Exception as e:
        print(f"OAuth start error: {e}")
        raise HTTPException(status_code=500, detail="Failed to start OAuth login")

    return {"url": oauth_response.url}


@router.post("/auth/oauth/session")
async def create_oauth_session(payload: OAuthSessionRequest, response: Response):
    if BYPASS_SUPABASE_AUTH:
        raise HTTPException(
            status_code=501,
            detail="OAuth login is not available when BYPASS_SUPABASE_AUTH is enabled",
        )

    access_token = payload.access_token.strip()
    if not access_token:
        raise HTTPException(status_code=400, detail="Access token is required")

    auth_user = get_user_from_supabase(access_token)
    _set_access_token_cookie(response, access_token)

    db_user = None
    db = await get_db()
    try:
        db_user = await _upsert_local_user_from_auth(db, auth_user)
        db_user = await db.fetchrow(
            """
            SELECT id, email, first_name, last_name
            FROM users
            WHERE id = $1
            """,
            str(auth_user.get("id")),
        )
    except Exception as db_error:
        print(f"OAuth session profile lookup warning: {db_error}")
    finally:
        await db.close()

    return {
        "detail": "Login successful",
        "user": _serialize_auth_user(auth_user, db_user),
    }

@router.post("/login")
async def login_user(user: UserLogin, response: Response):
    email = normalize_email(user.email)

    try:
        if BYPASS_SUPABASE_AUTH:
            raise HTTPException(
                status_code=501,
                detail="Login is not available when BYPASS_SUPABASE_AUTH is enabled"
            )

        # Ask Supabase Auth to sign in
        auth_response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": user.password
        })

        # If no user/session comes back, credentials are wrong
        if not auth_response.user or not auth_response.session:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        access_token = auth_response.session.access_token

        # Store token in an HTTP cookie
        _set_access_token_cookie(response, access_token)

        
        db_user = None
        db = await get_db()
        try:
            db_user = await _upsert_local_user_from_auth(db, auth_response.user)
            db_user = await db.fetchrow("""
                SELECT id, email, first_name, last_name
                FROM users
                WHERE id = $1
            """, str(auth_response.user.id))
        except Exception as db_error:
            # Supabase auth succeeded; treat local profile lookup as best-effort.
            print(f"Login profile lookup warning: {db_error}")
        finally:
            await db.close()

        # Return success message
        return {
            "detail": "Login successful",
            "user": _serialize_auth_user(auth_response.user, db_user),
        }

    except HTTPException:
        raise

    except Exception as e:
        error_message = str(e).lower()

        if "invalid login credentials" in error_message:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        if "email not confirmed" in error_message:
            raise HTTPException(
                status_code=401,
                detail="Please verify your email before logging in"
            )

        if any(
            fragment in error_message
            for fragment in {
                "nodename nor servname provided",
                "name or service not known",
                "temporary failure in name resolution",
                "failed to establish a new connection",
                "connection refused",
                "timed out",
                "timeout",
                "network is unreachable",
            }
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Unable to reach Supabase right now. "
                    "Check your network connection and SUPABASE_URL, then try again."
                ),
            )

        if "invalid api key" in error_message or "apikey" in error_message:
            raise HTTPException(
                status_code=500,
                detail="Supabase API key is misconfigured on the server"
            )

        #  unexpected server error
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@router.post("/auth/logout")
async def logout_user(response: Response):
    _clear_access_token_cookie(response)
    return {"detail": "Logout successful"}


@router.post("/auth/forgot-password")
async def forgot_password(payload: ForgotPasswordRequest):
    if BYPASS_SUPABASE_AUTH:
        raise HTTPException(
            status_code=501,
            detail="Password recovery is not available when BYPASS_SUPABASE_AUTH is enabled",
        )

    email = normalize_email(payload.email)
    redirect_to = _validate_auth_redirect_url(payload.redirect_to, "/reset-password")

    try:
        supabase.auth.reset_password_for_email(
            email,
            {
                "redirect_to": redirect_to,
            },
        )
    except Exception as e:
        error_message = str(e).lower()

        if "invalid email" in error_message or ("email" in error_message and "is invalid" in error_message):
            raise HTTPException(status_code=400, detail="Invalid email address")
        if "rate limit" in error_message or "for security purposes" in error_message:
            raise HTTPException(
                status_code=429,
                detail="Too many reset requests. Please wait a bit and try again.",
            )

        print(f"Forgot password error: {e}")
        raise HTTPException(status_code=500, detail="Failed to send password reset email")

    return {
        "detail": "If an account exists for that email, a password reset link has been sent."
    }


@router.post("/auth/reset-password")
async def reset_password(payload: ResetPasswordRequest):
    if BYPASS_SUPABASE_AUTH:
        raise HTTPException(
            status_code=501,
            detail="Password reset is not available when BYPASS_SUPABASE_AUTH is enabled",
        )

    access_token = payload.access_token.strip()
    if not access_token:
        raise HTTPException(status_code=400, detail="Access token is required")

    validate_password_strength(payload.new_password)

    try:
        reset_response = requests.put(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {access_token}",
                "apikey": SUPABASE_KEY,
                "Content-Type": "application/json",
            },
            json={"password": payload.new_password},
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"Reset password network error: {e}")
        raise HTTPException(status_code=503, detail="Unable to reach Supabase right now")

    if reset_response.status_code == 401:
        raise HTTPException(
            status_code=401,
            detail="This password reset link is invalid or has expired.",
        )

    if reset_response.status_code == 422:
        detail = reset_response.json().get("msg") or "Password reset request is invalid."
        raise HTTPException(status_code=400, detail=detail)

    if reset_response.status_code >= 400:
        try:
            error_payload = reset_response.json()
        except ValueError:
            error_payload = {}
        detail = (
            error_payload.get("msg")
            or error_payload.get("error_description")
            or error_payload.get("error")
            or "Failed to reset password"
        )
        raise HTTPException(status_code=400, detail=detail)

    return {"detail": "Password updated successfully"}
@router.post("/select-path")
async def select_path(
    payload: PathSelect,
    user_id: str = Depends(verify_token)
):
    db = await get_db()
    try:
        existing_profile = await db.fetchrow(
            """
            SELECT id, role, difficulty
            FROM user_profiles
            WHERE id = $1
            """,
            user_id,
        )
        previous_role = str(existing_profile["role"]) if existing_profile and existing_profile["role"] else None
        role_changed = previous_role is not None and previous_role != payload.role

        row = await db.fetchrow("""
            INSERT INTO user_profiles (id, role, difficulty)
            VALUES ($1, $2, 'beginner')
            ON CONFLICT (id)
            DO UPDATE SET
                role = EXCLUDED.role,
                difficulty = CASE
                    WHEN user_profiles.role IS DISTINCT FROM EXCLUDED.role THEN 'beginner'
                    ELSE COALESCE(user_profiles.difficulty, 'beginner')
                END
            RETURNING id, role, difficulty
        """, user_id, payload.role)

        if role_changed:
            # A true path switch should start the learner fresh for that role,
            # instead of carrying forward the previous role's level, scores,
            # cached lessons, or active questionnaire session.
            await db.execute(
                """
                DELETE FROM user_progress_lessons
                WHERE user_id = $1
                """,
                user_id,
            )
            await db.execute(
                """
                DELETE FROM user_progress
                WHERE user_id = $1
                """,
                user_id,
            )
            await db.execute(
                """
                DELETE FROM user_scores
                WHERE user_id = $1
                """,
                user_id,
            )
            _clear_user_cached_learning_state(user_id)
        else:
            # Keep only lesson assignments that belong to the currently selected path.
            await db.execute(
                """
                DELETE FROM user_progress_lessons upl
                USING learning_chunks lc
                WHERE upl.user_id = $1
                  AND upl.chunk_id = lc.id
                  AND COALESCE(lc.metadata->>'role', 'student') <> $2
                """,
                user_id,
                payload.role,
            )

        return {
            "detail": "Path saved successfully",
            "profile": {
                "id": str(row["id"]),
                "role": row["role"],
                "difficulty": row["difficulty"],
            },
            "questionnaire_reset": role_changed,
        }

    except Exception as e:
        print(f"Select path error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save selected path to database")
    finally:
        await db.close()

@router.get("/selected-path")
async def get_selected_path(user_id: str = Depends(verify_token)):
    db = await get_db()
    try:
        row = await db.fetchrow("""
            SELECT id, role, difficulty
            FROM user_profiles
            WHERE id = $1
        """, user_id)

        if not row:
            raise HTTPException(status_code=404, detail="User profile not found")

        return {
            "id": str(row["id"]),
            "role": row["role"],
            "difficulty": row["difficulty"],
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Get selected path error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch selected path")
    finally:
        await db.close()


# -----------------------------------------------------------------------------
# Dashboard route
# -----------------------------------------------------------------------------

@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(user_id: str = Depends(verify_token)):
    db = await get_db()
    try:
        payload = await _build_dashboard_payload(db, user_id)
    finally:
        await db.close()

    return payload


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(
    user_id: str = Depends(verify_token),
    authorization: Optional[str] = Header(None),
    access_token: Optional[str] = Cookie(None, alias=ACCESS_TOKEN_COOKIE),
):
    db = await get_db()
    try:
        try:
            payload = await _build_profile_payload(db, user_id)
        except HTTPException as exc:
            if exc.status_code != 404 or exc.detail != "User profile not found":
                raise

            token = _extract_access_token(authorization, access_token)
            if not token:
                raise

            auth_user = get_user_from_supabase(token)
            await _upsert_local_user_from_auth(db, auth_user)
            payload = await _build_profile_payload(db, user_id)
    finally:
        await db.close()

    return payload


@router.post("/chatbot/message", response_model=ChatMessageResponse)
async def chatbot_message(
    payload: ChatMessageRequest,
    user_id: str = Depends(verify_token),
):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    db = await get_db()
    try:
        profile_row = await db.fetchrow(
            """
            SELECT
                COALESCE(up.role, 'student') AS role,
                COALESCE((
                    SELECT current_level
                    FROM user_progress
                    WHERE user_id = $1
                    ORDER BY last_accessed DESC
                    LIMIT 1
                ), 1) AS current_level
            FROM users u
            LEFT JOIN user_profiles up
                ON up.id = u.id
            WHERE u.id = $1
            """,
            user_id,
        )
        lesson_context = None
        requested_source = (payload.source or "").strip()
        if requested_source:
            lesson_context = await _fetch_lesson_detail_context(db, user_id, requested_source)
        elif profile_row:
            assigned_lessons = await _fetch_assigned_lessons(
                db,
                user_id,
                str(profile_row["role"] or "student"),
            )
            current_lesson = next(
                (lesson for lesson in assigned_lessons if lesson["status"] == "in_progress"),
                None,
            ) or next(
                (lesson for lesson in assigned_lessons if lesson["status"] == "assigned"),
                None,
            )
            if current_lesson is None and assigned_lessons:
                current_lesson = assigned_lessons[0]

            if current_lesson and current_lesson.get("source"):
                lesson_context = await _fetch_lesson_detail_context(
                    db,
                    user_id,
                    str(current_lesson["source"]),
                )
    finally:
        await db.close()

    if not profile_row:
        raise HTTPException(status_code=404, detail="User profile not found")

    try:
        result = answer_question(
            message,
            role=str(profile_row["role"] or "student"),
            level=int(profile_row["current_level"] or 1),
            current_lesson_context=(
                str(lesson_context["lesson_text"] or "") if lesson_context else None
            ),
            current_lesson_title=(
                _clean_generated_text(str(lesson_context["lesson"]["topic"] or ""))
                if lesson_context
                else None
            ),
            current_lesson_source=(
                str(lesson_context["lesson"]["source"] or "") if lesson_context else None
            ),
            strict_lesson_focus=bool(requested_source),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print(f"Chatbot error: {exc}")
        raise HTTPException(status_code=500, detail="Chatbot is unavailable right now")

    return result


# -----------------------------------------------------------------------------
# Quiz routes
# -----------------------------------------------------------------------------

@router.get("/quiz/resume", response_model=QuizStartResponse)
async def quiz_resume(user_id: str = Depends(verify_token)):
    active_quiz_id = get_active_session_id(user_id)
    if not active_quiz_id:
        raise HTTPException(status_code=404, detail="No active quiz session")

    session = get_session(active_quiz_id)
    if not session:
        clear_active_session(user_id)
        raise HTTPException(status_code=404, detail="No active quiz session")

    if session.get("profile", {}).get("id") != user_id:
        clear_active_session(user_id)
        raise HTTPException(status_code=403, detail="Quiz session does not belong to user")

    current = session.get("current_question")
    if not current:
        raise HTTPException(status_code=404, detail="No active quiz question")

    return {
        "quiz_id": active_quiz_id,
        "question": {
            "text": current["question"],
            "options": current["options"],
        },
        "question_number": session["step"],
        "total_questions": QUIZ_TOTAL_QUESTIONS,
        "progress_percent": _progress_percent(session["step"], QUIZ_TOTAL_QUESTIONS),
    }


@router.post("/quiz/start", response_model=QuizStartResponse)
async def quiz_start(user_id: str = Depends(verify_token)):
    active_quiz_id = get_active_session_id(user_id)
    if active_quiz_id:
        active_session = get_session(active_quiz_id)
        if (
            active_session
            and active_session.get("profile", {}).get("id") == user_id
            and active_session.get("current_question")
            and int(active_session.get("step", 0)) <= QUIZ_TOTAL_QUESTIONS
        ):
            current = active_session["current_question"]
            return {
                "quiz_id": active_quiz_id,
                "question": {
                    "text": current["question"],
                    "options": current["options"],
                },
                "question_number": active_session["step"],
                "total_questions": QUIZ_TOTAL_QUESTIONS,
                "progress_percent": _progress_percent(
                    active_session["step"], QUIZ_TOTAL_QUESTIONS
                ),
            }

    db = await get_db()
    try:
        profile_row = await db.fetchrow(
            """
            SELECT id, role, difficulty
            FROM user_profiles
            WHERE id = $1
            """,
            user_id,
        )
        if not profile_row:
            raise HTTPException(status_code=404, detail="User profile not found")

        user_row = await db.fetchrow(
            """
            SELECT id, email, first_name, last_name
            FROM users
            WHERE id = $1
            """,
            user_id,
        )
    finally:
        await db.close()

    profile = {
        "id": str(profile_row["id"]),
        "role": profile_row["role"],
        "difficulty": profile_row["difficulty"],
    }

    if user_row:
        profile.update(
            {
                "email": user_row["email"],
                "first_name": user_row["first_name"],
                "last_name": user_row["last_name"],
            }
        )

    quiz_id = await create_session(profile)
    session = get_session(quiz_id)
    if session is None:
        raise HTTPException(status_code=500, detail="Failed to initialize quiz session")

    question = await next_question(session)
    save_session(quiz_id, session)

    return {
        "quiz_id": quiz_id,
        "question": {
            "text": question["question"],
            "options": question["options"],
        },
        "question_number": session["step"],
        "total_questions": QUIZ_TOTAL_QUESTIONS,
        "progress_percent": _progress_percent(session["step"], QUIZ_TOTAL_QUESTIONS),
    }


@router.post("/quiz/answer", response_model=QuizAnswerResponse)
async def quiz_answer(payload: QuizAnswerRequest, user_id: str = Depends(verify_token)):
    session = get_session(payload.quiz_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Quiz session not found")

    if session.get("profile", {}).get("id") != user_id:
        raise HTTPException(status_code=403, detail="Quiz session does not belong to user")

    try:
        correct, explanation = evaluate_answer(session, payload.answer_index)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    finished = session["step"] >= QUIZ_TOTAL_QUESTIONS
    if finished:
        save_session(payload.quiz_id, session)
        clear_active_session(user_id)

        profile = session.get("profile", {}) or {}
        role = profile.get("role")
        if not role:
            raise HTTPException(status_code=400, detail="Quiz session is missing the user role")

        assigned_level = _score_to_level(session["score"], QUIZ_TOTAL_QUESTIONS)

        db = await get_db()
        try:
            # Finishing the questionnaire sets the learner's starting platform
            # level and immediately seeds their first lesson assignments.
            await _save_user_progress(db, user_id, assigned_level, status="assigned")
            await _save_user_profile_difficulty(db, user_id, assigned_level)
            await _save_user_score(db, user_id, session["score"])
            recommendation_profile = await _build_lesson_recommendation_profile(
                db,
                profile,
                assigned_level,
                score=session["score"],
                total_questions=QUIZ_TOTAL_QUESTIONS,
                answer_history=session.get("answers", []),
            )
            assigned_lessons = await _assign_lessons_to_user(
                db, user_id, recommendation_profile, assigned_level
            )
        finally:
            await db.close()

        return {
            "quiz_id": payload.quiz_id,
            "correct": correct,
            "finished": True,
            "score": session["score"],
            "question_number": session["step"],
            "total_questions": QUIZ_TOTAL_QUESTIONS,
            "progress_percent": 100,
            "explanation": explanation,
            "question": None,
            "assigned_level": assigned_level,
            "assigned_lessons": assigned_lessons,
        }

    question = await next_question(session)
    save_session(payload.quiz_id, session)

    return {
        "quiz_id": payload.quiz_id,
        "correct": correct,
        "finished": False,
        "score": session["score"],
        "question_number": session["step"],
        "total_questions": QUIZ_TOTAL_QUESTIONS,
        "progress_percent": _progress_percent(session["step"], QUIZ_TOTAL_QUESTIONS),
        "explanation": explanation,
        "question": {
            "text": question["question"],
            "options": question["options"],
        },
    }


# -----------------------------------------------------------------------------
# Lesson routes
# -----------------------------------------------------------------------------

@router.get("/lessons/assigned", response_model=LessonAssignmentResponse)
async def get_assigned_lessons(user_id: str = Depends(verify_token)):
    db = await get_db()
    try:
        profile_row = await db.fetchrow(
            """
            SELECT id, role, difficulty
            FROM user_profiles
            WHERE id = $1
            """,
            user_id,
        )
        if not profile_row:
            raise HTTPException(status_code=404, detail="User profile not found")

        progress_row = await db.fetchrow(
            """
            SELECT current_level
            FROM user_progress
            WHERE user_id = $1
            ORDER BY last_accessed DESC
            LIMIT 1
            """,
            user_id,
        )

        if not progress_row:
            raise HTTPException(
                status_code=404,
                detail="Questionnaire not completed yet",
            )

        level = _normalize_level(progress_row["current_level"])
        lessons = await _fetch_assigned_lessons(db, user_id, profile_row["role"])
        if not lessons:
            # Lazily create recommendations the first time this learner opens
            # lessons after receiving a valid role and level.
            assignment_profile = {
                "id": str(profile_row["id"]),
                "role": profile_row["role"],
                "difficulty": profile_row["difficulty"],
            }
            recommendation_profile = await _build_lesson_recommendation_profile(
                db,
                assignment_profile,
                level,
            )
            lessons = await _assign_lessons_to_user(
                db, user_id, recommendation_profile, level
            )
    finally:
        await db.close()

    return {
        "role": profile_row["role"],
        "level": level,
        "lessons": lessons,
    }


@router.get("/lessons/detail", response_model=LessonDetailResponse)
async def get_lesson_detail(
    source: str = Query(..., min_length=1),
    user_id: str = Depends(verify_token),
):
    db = await get_db()
    try:
        context = await _fetch_lesson_detail_context(db, user_id, source)
        await db.execute(
            """
            UPDATE user_progress_lessons upl
            SET status = CASE
                    WHEN upl.status = 'assigned' THEN 'in_progress'
                    ELSE upl.status
                END,
                last_accessed = now()
            FROM learning_chunks lc
            WHERE upl.user_id = $1
              AND upl.chunk_id = lc.id
              AND COALESCE(lc.metadata->>'source', '') = $2
            """,
            user_id,
            source,
        )

        cache_key = _lesson_storyboard_cache_key(user_id, source)
        cached_storyboard = redis_client.get(cache_key)
        if cached_storyboard:
            storyboard = json.loads(cached_storyboard)
        else:
            # Storyboards are generated once per user/source and then cached, so
            # reopening a lesson does not re-run generation every time.
            storyboard = await generate_lesson_storyboard(
                context["profile"],
                context["lesson"],
                context["lesson_text"],
                context.get("next_lesson"),
            )
            redis_client.set(
                cache_key,
                json.dumps(storyboard),
                ex=LESSON_STORYBOARD_CACHE_TTL_SECONDS,
            )
    finally:
        await db.close()

    lesson = context["lesson"]
    return {
        "source": lesson["source"],
        "topic": _clean_generated_text(lesson["topic"]),
        "role": lesson["role"],
        "level": lesson["level"],
        "status": "completed" if lesson["status"] == "completed" else "in_progress",
        "estimated_minutes": context["estimated_minutes"],
        "progress": context["progress"],
        "overview": _clean_generated_text(storyboard["overview"]),
        "learning_objectives": [
            _clean_generated_text(item) for item in (storyboard["learning_objectives"] or [])
        ],
        "slides": [
            {
                **slide,
                "title": _clean_generated_text(slide.get("title")),
                "narrative": _clean_generated_text(slide.get("narrative")),
                "bullets": [_clean_generated_text(item) for item in (slide.get("bullets") or [])],
                "scene_caption": _clean_generated_text(slide.get("scene_caption")) or None,
                "dialogue_line": _clean_generated_text(slide.get("dialogue_line")) or None,
                "illustration_prompt": _clean_generated_text(slide.get("illustration_prompt")),
                "speaker_note": _clean_generated_text(slide.get("speaker_note")) or None,
                "checkpoint_question": _clean_generated_text(slide.get("checkpoint_question")) or None,
            }
            for slide in (storyboard["slides"] or [])
        ],
        "end_quiz": [
            {
                **item,
                "question": _clean_generated_text(item.get("question")),
                "options": [_clean_generated_text(option) for option in (item.get("options") or [])],
                "explanation": _clean_generated_text(item.get("explanation")),
            }
            for item in (storyboard["end_quiz"] or [])
        ],
        "next_lesson": (
            {
                **storyboard["next_lesson"],
                "title": _clean_generated_text(storyboard["next_lesson"].get("title")),
                "reason": _clean_generated_text(storyboard["next_lesson"].get("reason")),
            }
            if lesson["status"] == "completed" and storyboard.get("next_lesson")
            else None
        ),
    }


@router.get("/lessons/state", response_model=LessonStateResponse)
async def get_lesson_state(
    source: str = Query(..., min_length=1),
    user_id: str = Depends(verify_token),
):
    cached_state = redis_client.get(_lesson_state_cache_key(user_id, source))
    if not cached_state:
        return {
            "source": source,
            "active_slide": 0,
            "phase": "story",
            "quiz_answers": {},
            "quiz_submitted": False,
        }

    payload = json.loads(cached_state)
    return {
        "source": source,
        "active_slide": max(0, int(payload.get("active_slide", 0))),
        "phase": payload.get("phase", "story"),
        "quiz_answers": {
            str(key): int(value)
            for key, value in dict(payload.get("quiz_answers") or {}).items()
        },
        "quiz_submitted": bool(payload.get("quiz_submitted", False)),
    }


@router.post("/lessons/state", response_model=LessonStateResponse)
async def save_lesson_state(
    payload: LessonStateRequest,
    user_id: str = Depends(verify_token),
):
    normalized_payload = {
        "source": payload.source,
        "active_slide": max(0, int(payload.active_slide)),
        "phase": payload.phase,
        "quiz_answers": {str(key): int(value) for key, value in payload.quiz_answers.items()},
        "quiz_submitted": bool(payload.quiz_submitted),
    }
    redis_client.set(
        _lesson_state_cache_key(user_id, payload.source),
        json.dumps(normalized_payload),
        ex=LESSON_STATE_CACHE_TTL_SECONDS,
    )
    return normalized_payload


@router.post("/lessons/complete", response_model=LessonAssignmentResponse)
async def complete_lesson(
    payload: LessonCompletionRequest, user_id: str = Depends(verify_token)
):
    if payload.status == "completed":
        total_questions = int(payload.total_questions or 0)
        quiz_score = int(payload.quiz_score or 0)
        required_score = _lesson_passing_score(total_questions)
        if total_questions <= 0:
            raise HTTPException(
                status_code=400,
                detail="A lesson quiz is required before completing this lesson.",
            )
        if quiz_score < required_score:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"You need at least {LESSON_PASSING_PERCENT}% to pass this lesson quiz "
                    f"({required_score}/{total_questions})."
                ),
            )

    db = await get_db()
    try:
        profile_row = await db.fetchrow(
            """
            SELECT id, role, difficulty
            FROM user_profiles
            WHERE id = $1
            """,
            user_id,
        )
        if not profile_row:
            raise HTTPException(status_code=404, detail="User profile not found")

        progress_row = await db.fetchrow(
            """
            SELECT current_level
            FROM user_progress
            WHERE user_id = $1
            ORDER BY last_accessed DESC
            LIMIT 1
            """,
            user_id,
        )

        if not progress_row:
            raise HTTPException(
                status_code=404,
                detail="Questionnaire not completed yet",
            )

        level = _normalize_level(progress_row["current_level"])

        await db.execute(
            """
            UPDATE user_progress_lessons upl
            SET status = $3,
                last_accessed = now()
            FROM learning_chunks lc
            WHERE upl.user_id = $1
              AND upl.chunk_id = lc.id
              AND lc.metadata->>'source' = $2
              AND COALESCE(lc.metadata->>'role', 'student') = $4
            """,
            user_id,
            payload.source,
            payload.status,
            profile_row["role"],
        )

        await db.execute(
            """
            INSERT INTO user_progress_lessons (user_id, chunk_id, status)
            SELECT $1, lc.id, $3
            FROM learning_chunks lc
            WHERE lc.metadata->>'source' = $2
              AND COALESCE(lc.metadata->>'role', 'student') = $4
              AND NOT EXISTS (
                    SELECT 1
                    FROM user_progress_lessons upl
                    WHERE upl.user_id = $1
                      AND upl.chunk_id = lc.id
              )
            """,
            user_id,
            payload.source,
            payload.status,
            profile_row["role"],
        )

        if payload.status == "completed":
            # Completing a lesson persists its quiz result and then runs the
            # progression logic that may level the learner up.
            await _save_lesson_quiz_score(
                db,
                user_id,
                payload.source,
                payload.quiz_score,
                payload.total_questions,
            )
            redis_client.delete(_lesson_storyboard_cache_key(user_id, payload.source))
            redis_client.delete(_lesson_state_cache_key(user_id, payload.source))
            level, lessons = await _refresh_lessons_after_completion(
                db,
                user_id,
                profile_row["role"],
                level,
            )
        else:
            await _save_user_progress(db, user_id, level, status="in_progress")
            lessons = await _fetch_assigned_lessons(db, user_id, profile_row["role"])
    finally:
        await db.close()

    return {
        "role": profile_row["role"],
        "level": level,
        "lessons": lessons,
    }


# -----------------------------------------------------------------------------
# Development-only quiz routes
# -----------------------------------------------------------------------------

@router.post("/quiz/start/dev", response_model=QuizStartResponse)
async def quiz_start_dev(payload: QuizStartDevRequest):
    if APP_ENV == "production":
        raise HTTPException(status_code=404, detail="Not found")

    if not isinstance(payload.profile, dict):
        raise HTTPException(status_code=400, detail="profile must be an object")

    quiz_id = await create_session(payload.profile)
    session = get_session(quiz_id)
    if session is None:
        raise HTTPException(status_code=500, detail="Failed to initialize quiz session")

    question = await next_question(session)
    save_session(quiz_id, session)

    return {
        "quiz_id": quiz_id,
        "question": {
            "text": question["question"],
            "options": question["options"],
        },
        "question_number": session["step"],
        "total_questions": QUIZ_TOTAL_QUESTIONS,
        "progress_percent": _progress_percent(session["step"], QUIZ_TOTAL_QUESTIONS),
    }


@router.post("/quiz/answer/dev", response_model=QuizAnswerResponse)
async def quiz_answer_dev(payload: QuizAnswerDevRequest):
    if APP_ENV == "production":
        raise HTTPException(status_code=404, detail="Not found")

    session = get_session(payload.quiz_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Quiz session not found")

    try:
        correct, explanation = evaluate_answer(session, payload.answer_index)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    finished = session["step"] >= QUIZ_TOTAL_QUESTIONS
    if finished:
        save_session(payload.quiz_id, session)
        return {
            "quiz_id": payload.quiz_id,
            "correct": correct,
            "finished": True,
            "score": session["score"],
            "question_number": session["step"],
            "total_questions": QUIZ_TOTAL_QUESTIONS,
            "progress_percent": 100,
            "explanation": explanation,
            "question": None,
        }

    question = await next_question(session)
    save_session(payload.quiz_id, session)

    return {
        "quiz_id": payload.quiz_id,
        "correct": correct,
        "finished": False,
        "score": session["score"],
        "question_number": session["step"],
        "total_questions": QUIZ_TOTAL_QUESTIONS,
        "progress_percent": _progress_percent(session["step"], QUIZ_TOTAL_QUESTIONS),
        "explanation": explanation,
        "question": {
            "text": question["question"],
            "options": question["options"],
        },
    }
