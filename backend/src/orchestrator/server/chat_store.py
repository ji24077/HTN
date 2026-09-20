"""Durable chat checkpoints; no database transaction spans a model call."""

from contextlib import asynccontextmanager

from ..agent import Conversation
from .db.store import Conflict, NotFound


class ChatStore:
    def __init__(self, pool):
        self.pool = pool

    async def read(self, owner, conversation_id):
        data = await self.pool.fetchval(
            "SELECT data FROM chat_conversations WHERE id=$1 AND owner=$2", conversation_id, owner
        )
        if data is None:
            raise NotFound("Conversation not found")
        return Conversation.model_validate(data)

    @asynccontextmanager
    async def edit(self, owner, conversation_id):
        async with self.pool.acquire(timeout=2) as conn:
            # One active turn per principal across processes. A session lock is
            # released on disconnect; it does not block scheduler transactions.
            locked = await conn.fetchval(
                "SELECT pg_try_advisory_lock(hashtextextended($1, 17))", owner
            )
            if not locked:
                raise Conflict("A chat message is still running. Check its response shortly.")
            try:
                row = await conn.fetchrow(
                    "SELECT owner,data FROM chat_conversations WHERE id=$1", conversation_id
                )
                if row and row["owner"] != owner:
                    raise NotFound("Conversation not found")
                if row:
                    state = Conversation.model_validate(row["data"])
                else:
                    count = await conn.fetchval(
                        "SELECT count(*) FROM chat_conversations WHERE owner=$1", owner
                    )
                    if count >= 100:
                        raise Conflict(
                            "Conversation limit reached. Ask an operator to clear old chats."
                        )
                    state = Conversation()
                    inserted = await conn.fetchval(
                        "INSERT INTO chat_conversations(id,owner,data) VALUES($1,$2,$3) ON CONFLICT DO NOTHING RETURNING id",
                        conversation_id,
                        owner,
                        state.model_dump(mode="json"),
                    )
                    if inserted is None:
                        raise NotFound("Conversation not found")

                async def save(value):
                    await conn.execute(
                        "UPDATE chat_conversations SET data=$3,updated_at=clock_timestamp() WHERE id=$1 AND owner=$2",
                        conversation_id,
                        owner,
                        value.model_dump(mode="json"),
                    )

                yield state, save
            finally:
                await conn.execute("SELECT pg_advisory_unlock(hashtextextended($1, 17))", owner)
