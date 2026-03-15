"""
MongoDB session persistence using motor (async driver).

Collection: sessions
Schema (from SKILL.md):
  session_id:  String (UUID, unique index)
  started_at:  ISODate
  ended_at:    ISODate | None
  turns:       Array of {role, content, timestamp, tool_calls}
  metadata:    {domain: "medbrief", user_agent: String}

ensure_session uses $setOnInsert so concurrent first-turn calls are safe.
append_turn pushes both user and assistant turns atomically and updates ended_at.
"""
from datetime import datetime
from typing import Optional

import structlog
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = structlog.get_logger()


class SessionDB:
    def __init__(self, settings):
        self._settings = settings
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None

    async def connect(self) -> None:
        self._client = AsyncIOMotorClient(self._settings.mongodb_url)
        self._db = self._client[self._settings.mongodb_db]
        # Verify connectivity
        await self._client.admin.command("ping")
        # Unique index on session_id for fast lookups and safe upserts
        await self._db.sessions.create_index(
            "session_id", unique=True, background=True
        )
        logger.info("mongodb_connected", db=self._settings.mongodb_db)

    async def close(self) -> None:
        if self._client:
            self._client.close()
        logger.info("mongodb_closed")

    async def ensure_session(
        self, session_id: str, user_agent: str = ""
    ) -> None:
        """
        Create a session document the first time this session_id is seen.
        $setOnInsert is a no-op if the document already exists, making this
        safe to call on every message without overwriting existing turns.
        """
        doc = {
            "session_id": session_id,
            "started_at": datetime.utcnow(),
            "ended_at": None,
            "turns": [],
            "metadata": {
                "domain": "medbrief",
                "user_agent": user_agent,
            },
        }
        try:
            await self._db.sessions.update_one(
                {"session_id": session_id},
                {"$setOnInsert": doc},
                upsert=True,
            )
            logger.debug("mongodb_session_ensured", session_id=session_id)
        except Exception as exc:
            logger.error(
                "mongodb_ensure_session_failed",
                session_id=session_id,
                error=str(exc),
            )
            raise

    async def append_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        tool_calls: Optional[list[dict]] = None,
    ) -> None:
        """
        Atomically append user + assistant turns and update ended_at.
        tool_calls is logged on the assistant turn for auditability.
        """
        if tool_calls is None:
            tool_calls = []

        now = datetime.utcnow()
        user_turn = {
            "role": "user",
            "content": user_text,
            "timestamp": now,
            "tool_calls": [],
        }
        assistant_turn = {
            "role": "assistant",
            "content": assistant_text,
            "timestamp": now,
            "tool_calls": tool_calls,
        }
        try:
            await self._db.sessions.update_one(
                {"session_id": session_id},
                {
                    "$push": {"turns": {"$each": [user_turn, assistant_turn]}},
                    "$set": {"ended_at": now},
                },
            )
            logger.debug(
                "mongodb_turns_appended",
                session_id=session_id,
                tool_calls_count=len(tool_calls),
            )
        except Exception as exc:
            logger.error(
                "mongodb_append_turn_failed",
                session_id=session_id,
                error=str(exc),
            )
            raise
