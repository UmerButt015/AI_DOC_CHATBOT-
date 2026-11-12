from __future__ import annotations

import json
from typing import Optional

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import trim_messages

try:
    from langchain_community.chat_message_histories import RedisChatMessageHistory
except ImportError:  # pragma: no cover
    RedisChatMessageHistory = None

from app.config import settings

_session_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str, redis_url: Optional[str] = None) -> BaseChatMessageHistory:
    """Return a session history store. Prefer Redis when configured, otherwise fallback to memory."""
    if redis_url and RedisChatMessageHistory is not None:
        return RedisChatMessageHistory(session_id=session_id, url=redis_url)

    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def trim_session_history(history: BaseChatMessageHistory, token_counter=None, max_tokens: int = 2000):
    """Keep the conversation bounded while preserving the most recent messages."""
    if not hasattr(history, "messages"):
        return

    messages = list(history.messages)
    if not messages:
        return

    trimmed = trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",
        token_counter=token_counter,
    )

    history.clear()
    for message in trimmed:
        history.add_message(message)


def clear_session(session_id: str):
    """Clear a single session history object."""
    _session_store.pop(session_id, None)


def set_redis_session_metadata(session_id: str, key: str, value: object, ttl: Optional[int] = None) -> bool:
    """Store lightweight session metadata in Redis for fast session state access."""
    if not settings.REDIS_URL:
        return False

    try:
        import redis
    except ImportError:
        return False

    client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        client.set(f"session:{session_id}:{key}", json.dumps(value), ex=ttl or settings.TEMP_STATE_TTL)
        return True
    except redis.exceptions.RedisError:
        return False


def get_redis_session_metadata(session_id: str, key: str) -> object:
    """Read lightweight session metadata from Redis."""
    if not settings.REDIS_URL:
        return None

    try:
        import redis
    except ImportError:
        return None

    client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        raw = client.get(f"session:{session_id}:{key}")
        if raw is None:
            return None
        return json.loads(raw)
    except (TypeError, ValueError, redis.exceptions.RedisError):
        return None