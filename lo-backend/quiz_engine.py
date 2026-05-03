import json
import uuid
import hashlib
from typing import Any, Dict, List

from redis_client import redis_client
from agents.Quiz_agent import generate_question

SESSION_TTL_SECONDS = 1800
QUIZ_TOTAL_QUESTIONS = 10
QUESTION_CACHE_TTL_SECONDS = 86400


def _active_session_key(user_id: str) -> str:
    return f"quiz:active:{user_id}"


def _validate_question(q: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(q, dict):
        raise ValueError("Question payload must be a dict")
    if "question" not in q or not isinstance(q["question"], str):
        raise ValueError("Question missing 'question' string")
    if "options" not in q or not isinstance(q["options"], list) or len(q["options"]) != 4:
        raise ValueError("Question missing 4 options")
    if "correct_index" not in q or not isinstance(q["correct_index"], int):
        raise ValueError("Question missing correct_index")
    return q


async def create_session(profile: Dict[str, Any]) -> str:
    session_id = str(uuid.uuid4())
    starting_difficulty = profile.get("difficulty") or 1
    try:
        starting_difficulty = int(starting_difficulty)
    except Exception:
        starting_difficulty = 1
    starting_difficulty = max(1, min(10, starting_difficulty))
    
    session = {
        "profile": profile,
        "difficulty": int(starting_difficulty),
        "score": 0,
        "step": 0,
        "history": [],
        "current_question": None,
    }
    
    redis_client.set(session_id, json.dumps(session), ex=SESSION_TTL_SECONDS)
    profile_id = profile.get("id")
    if profile_id:
        redis_client.set(
            _active_session_key(str(profile_id)),
            session_id,
            ex=SESSION_TTL_SECONDS,
        )
    return session_id


def get_session(session_id: str) -> Dict[str, Any] | None:
    data = redis_client.get(session_id)
    return json.loads(data) if data else None


def save_session(session_id: str, session: Dict[str, Any]) -> None:
    redis_client.set(session_id, json.dumps(session), ex=SESSION_TTL_SECONDS)
    profile_id = session.get("profile", {}).get("id")
    if profile_id:
        redis_client.set(
            _active_session_key(str(profile_id)),
            session_id,
            ex=SESSION_TTL_SECONDS,
        )


def get_active_session_id(user_id: str) -> str | None:
    return redis_client.get(_active_session_key(user_id))


def clear_active_session(user_id: str) -> None:
    redis_client.delete(_active_session_key(user_id))


def _question_cache_key(session: Dict[str, Any], next_step: int) -> str:
    profile = session.get("profile", {}) or {}
    user_id = profile.get("id")
    if user_id:
        user_part = str(user_id)
    else:
        user_part = hashlib.sha1(
            json.dumps(profile, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
    role = str(profile.get("role", "unknown"))
    return f"quiz:question:{user_part}:{role}:step:{next_step}"


async def next_question(session: Dict[str, Any]) -> Dict[str, Any]:
    next_step = int(session.get("step", 0)) + 1
    cache_key = _question_cache_key(session, next_step)
    cached = redis_client.get(cache_key)

    if cached:
        q = _validate_question(json.loads(cached))
    else:
        q = await generate_question(
            session["profile"], session["difficulty"], session["history"]
        )
        q = _validate_question(q)
        redis_client.set(cache_key, json.dumps(q), ex=QUESTION_CACHE_TTL_SECONDS)

    session["current_question"] = q
    session["history"].append(q["question"])
    session["step"] += 1
    return q


def evaluate_answer(session: Dict[str, Any], selected_index: int) -> tuple[bool, str]:
    q = session.get("current_question")
    if not q:
        raise ValueError("No current question in session")
    correct = selected_index == q["correct_index"]
    if correct:
        session["score"] += 1
        session["difficulty"] = min(session["difficulty"] + 1, 10)
    else:
        session["difficulty"] = max(session["difficulty"] - 1, 1)
    return correct, q["explanation"]
    
