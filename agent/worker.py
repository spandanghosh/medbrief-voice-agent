"""
Agent Worker — RabbitMQ consumer and conversation orchestrator.

Orchestration flow for each inbound message:
  1. Decode JSON payload {session_id, text}
  2. Ensure MongoDB session document exists
  3. Load conversation history from Redis
  4. Call Groq LLM (with optional PubMed tool use)
  5. Persist turns to Redis (hot) and MongoDB (cold)
  6. Publish LLM response back to the reply_to queue

Dead-letter behaviour: if process_message raises, aio-pika's
message.process(requeue=False) nacks the message → routed to DLQ.
"""
import asyncio
import json
import logging

import aio_pika
import aio_pika.abc
import structlog
from openai import RateLimitError
from pydantic_settings import BaseSettings

from db import SessionDB
from llm import LLMClient
from memory import ConversationMemory


# ── Settings ──────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    groq_api_key: str
    llm_model: str = "llama-3.3-70b-versatile"

    rabbitmq_url: str = "amqp://guest:guest@rabbitmq:5672/"
    rabbitmq_exchange: str = "agent.direct"
    rabbitmq_request_queue: str = "agent.requests"
    rabbitmq_dlx: str = "agent.dlx"
    rabbitmq_dlq: str = "agent.dead_letters"

    redis_url: str = "redis://redis:6379/0"
    session_ttl: int = 1800
    max_history_turns: int = 20

    mongodb_url: str = "mongodb://mongodb:27017"
    mongodb_db: str = "medbrief"

    log_level: str = "INFO"

    pubmed_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    pubmed_api_key: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# ── Logging ───────────────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.getLevelName(settings.log_level)
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()


# ── Message processor ─────────────────────────────────────────────────────────

async def process_message(
    message: aio_pika.IncomingMessage,
    memory: ConversationMemory,
    llm_client: LLMClient,
    db: SessionDB,
    channel: aio_pika.abc.AbstractChannel,
) -> None:
    """
    Process one inbound AMQP message.
    Context manager ensures ack on success, nack (→ DLQ) on exception.
    """
    async with message.process(requeue=False):
        payload = json.loads(message.body)
        session_id = payload["session_id"]
        user_text = payload["text"]

        log = logger.bind(
            session_id=session_id,
            correlation_id=message.correlation_id,
        )
        log.info("agent_message_received", preview=user_text[:80])

        response_text = ""
        tool_calls: list[dict] = []

        try:
            # ── 1. Ensure MongoDB session document exists ────────────────────
            await db.ensure_session(session_id=session_id)

            # ── 2. Load conversation history from Redis ──────────────────────
            history = await memory.get_history(session_id=session_id)
            log.info("history_loaded", turns=len(history))

            # ── 3. Call LLM (may invoke tool calls internally) ───────────────
            response_text, tool_calls = await llm_client.chat(
                history=history,
                user_text=user_text,
            )
            log.info(
                "llm_replied",
                preview=response_text[:80],
                tool_calls=len(tool_calls),
            )

            # ── 4. Persist to Redis (hot memory for next turn) ───────────────
            await memory.append_turn(session_id, "user", user_text)
            await memory.append_turn(session_id, "assistant", response_text)

            # ── 5. Persist to MongoDB (cold storage / audit trail) ───────────
            await db.append_turn(
                session_id=session_id,
                user_text=user_text,
                assistant_text=response_text,
                tool_calls=tool_calls,
            )

        except RateLimitError:
            log.error("agent_rate_limited")
            response_text = (
                "I am currently rate limited by the AI service. "
                "Please wait a moment and try again."
            )
        except Exception as exc:
            log.error("agent_processing_failed", error=str(exc))
            response_text = (
                "I encountered an error processing your request. Please try again."
            )

        # ── 6. Always publish reply — even on errors, so the gateway never
        #       times out waiting for a response that will never arrive. ───────
        if message.reply_to and message.correlation_id:
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=response_text.encode(),
                    correlation_id=message.correlation_id,
                    content_type="text/plain",
                ),
                routing_key=message.reply_to,
            )
            log.info("agent_reply_published", reply_to=message.reply_to)
        else:
            log.warning("agent_message_missing_reply_to")


# ── Main loop ─────────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("agent_worker_starting")

    memory = ConversationMemory(settings)
    await memory.connect()

    db = SessionDB(settings)
    await db.connect()

    llm_client = LLMClient(settings)

    connection = await aio_pika.connect_robust(settings.rabbitmq_url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=5)

    # Declare topology (idempotent — mirrors gateway declarations)
    dlx = await channel.declare_exchange(
        settings.rabbitmq_dlx,
        aio_pika.ExchangeType.DIRECT,
        durable=True,
    )
    dlq = await channel.declare_queue(settings.rabbitmq_dlq, durable=True)
    await dlq.bind(dlx, routing_key=settings.rabbitmq_dlq)

    request_queue = await channel.declare_queue(
        settings.rabbitmq_request_queue,
        durable=True,
        arguments={
            "x-dead-letter-exchange": settings.rabbitmq_dlx,
            "x-dead-letter-routing-key": settings.rabbitmq_dlq,
        },
    )
    exchange = await channel.declare_exchange(
        settings.rabbitmq_exchange,
        aio_pika.ExchangeType.DIRECT,
        durable=True,
    )
    await request_queue.bind(exchange, routing_key=settings.rabbitmq_request_queue)

    logger.info(
        "agent_worker_ready",
        queue=settings.rabbitmq_request_queue,
        exchange=settings.rabbitmq_exchange,
    )

    async def on_message(msg: aio_pika.IncomingMessage) -> None:
        try:
            await process_message(msg, memory, llm_client, db, channel)
        except Exception as exc:
            logger.error("agent_message_failed", error=str(exc))

    await request_queue.consume(on_message)

    try:
        await asyncio.Future()  # run forever until cancelled
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("agent_worker_stopping")
    finally:
        await memory.close()
        await db.close()
        await connection.close()
        logger.info("agent_worker_stopped")


if __name__ == "__main__":
    asyncio.run(main())
