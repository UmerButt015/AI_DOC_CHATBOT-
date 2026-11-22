import json
import time
from typing import Any, Dict, Optional

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None

try:
    from upstash_redis import Redis as UpstashRedis
except ImportError:  # pragma: no cover
    UpstashRedis = None

from app.config import settings


_upstash_client = None


def get_redis_client():
    """Return an Upstash REST client or a standard Redis client when reachable."""
    global _upstash_client

    if settings.UPSTASH_REDIS_REST_URL and settings.UPSTASH_REDIS_REST_TOKEN:
        if UpstashRedis is not None:
            try:
                if _upstash_client is None:
                    _upstash_client = UpstashRedis(
                        url=settings.UPSTASH_REDIS_REST_URL,
                        token=settings.UPSTASH_REDIS_REST_TOKEN,
                    )
                _upstash_client.ping()
                return _upstash_client
            except Exception:
                return None

    if not settings.REDIS_URL or redis is None:
        return None

    try:
        client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except (redis.exceptions.RedisError, ValueError, TypeError):
        return None


def cache_get(key: str) -> Optional[Any]:
    """Get a JSON value from Redis cache."""
    client = get_redis_client()
    if client is None:
        return None

    raw = client.get(key)
    if raw is None:
        return None

    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Set a JSON value into Redis cache with optional TTL."""
    client = get_redis_client()
    if client is None:
        return False

    payload = json.dumps(value)
    expires_in = ttl if ttl is not None else settings.CACHE_TTL

    client.set(key, payload, ex=expires_in)
    return True


def cache_delete(key: str) -> bool:
    """Delete a cached value."""
    client = get_redis_client()
    if client is None:
        return False

    return bool(client.delete(key))


def set_temp_state(namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Persist short-lived application state in Redis."""
    client = get_redis_client()
    if client is None:
        return False

    expires_in = ttl if ttl is not None else settings.TEMP_STATE_TTL
    redis_key = f"{settings.TEMP_STATE_PREFIX}{namespace}:{key}"
    client.set(redis_key, json.dumps(value), ex=expires_in)
    return True


def get_temp_state(namespace: str, key: str) -> Optional[Any]:
    """Read short-lived application state from Redis."""
    client = get_redis_client()
    if client is None:
        return None

    redis_key = f"{settings.TEMP_STATE_PREFIX}{namespace}:{key}"
    raw = client.get(redis_key)
    if raw is None:
        return None

    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def rate_limit_check(identifier: str, limit: Optional[int] = None, window_seconds: Optional[int] = None) -> bool:
    """Allow at most N requests per window using Redis as the source of truth."""
    client = get_redis_client()
    if client is None:
        return True

    max_requests = limit if limit is not None else settings.RATE_LIMIT_PER_MINUTE
    window = window_seconds if window_seconds is not None else settings.RATE_LIMIT_WINDOW_SECONDS

    bucket = int(time.time() // window)
    key = f"rate_limit:{identifier}:{bucket}"

    current = client.incr(key)
    if current == 1:
        client.expire(key, window + 1)

    return current <= max_requests


def enqueue_job(job_name: str, payload: Dict[str, Any]) -> bool:
    """Push a background job onto a Redis queue."""
    client = get_redis_client()
    if client is None:
        return False

    item = {
        "name": job_name,
        "payload": payload,
        "created_at": int(time.time()),
    }
    client.rpush(settings.QUEUE_NAME, json.dumps(item))
    return True


def get_queue_size() -> int:
    """Return how many jobs are currently waiting in the Redis queue."""
    client = get_redis_client()
    if client is None:
        return 0

    return int(client.llen(settings.QUEUE_NAME))
