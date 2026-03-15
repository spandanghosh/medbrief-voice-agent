"""
Redis conversation history manager.

Schema (from SKILL.md):
  Key:   session:{session_id}:history
  Type:  Redis List — newest turn at index 0 (LPUSH)
  Value: JSON {"role": "user"|"assistant", "content": "..."}
  TTL:   SESSION_TTL seconds (default 1800 = 30 min)
  Max:   MAX_HISTORY_TURNS entries (LTRIM after push)

get_history() reverses the list so the LLM receives turns oldest-first.
"""
import json
from typing import Optional

import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger()


class ConversationMemory:
    def __init__(self, settings):
        self._settings = settings
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        self._redis = await aioredis.from_url(
            self._settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        await self._redis.ping()
        logger.info("redis_connected", url=self._settings.redis_url)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
        logger.info("redis_closed")

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}:history"

    async def get_history(self, session_id: str) -> list[dict]:
        """
        Return conversation turns oldest-first for the LLM messages array.
        Returns empty list on Redis error (graceful degradation).
        """
        key = self._key(session_id)
        try:
            # lrange returns newest-first; reverse for LLM chronological order
            raw = await self._redis.lrange(key, 0, -1)
            history = [json.loads(item) for item in reversed(raw)]
            logger.debug("redis_history_loaded", session_id=session_id, turns=len(history))
            return history
        except Exception as exc:
            logger.error(
                "redis_get_history_failed",
                session_id=session_id,
                error=str(exc),
            )
            return []

    async def append_turn(self, session_id: str, role: str, content: str) -> None:
        """
        Prepend a turn to the list (newest at index 0), trim to max turns,
        and refresh the TTL — all atomically via a pipeline.
        """
        key = self._key(session_id)
        item = json.dumps({"role": role, "content": content})
        try:
            pipe = self._redis.pipeline()
            pipe.lpush(key, item)
            pipe.ltrim(key, 0, self._settings.max_history_turns - 1)
            pipe.expire(key, self._settings.session_ttl)
            await pipe.execute()
            logger.debug(
                "redis_turn_appended",
                session_id=session_id,
                role=role,
                chars=len(content),
            )
        except Exception as exc:
            logger.error(
                "redis_append_failed",
                session_id=session_id,
                role=role,
                error=str(exc),
            )
            raise
