"""
AMQP RPC Publisher — implements the request/reply pattern over RabbitMQ.

Pattern (mirrors Haptik's async bot pipeline):
  1. Gateway publishes message to agent.direct exchange → agent.requests queue
  2. Message carries correlation_id + reply_to = exclusive anonymous queue name
  3. Agent worker processes message, publishes reply back to reply_to queue
  4. Gateway awaits the reply via asyncio.Future keyed on correlation_id
  5. asyncio.wait_for enforces the RPC_TIMEOUT_SECONDS deadline
  6. Failed messages that exhaust delivery are routed to agent.dlq (dead-letter)
"""
import asyncio
import json
import uuid
from typing import Optional

import aio_pika
import aio_pika.abc
import structlog
from fastapi import HTTPException

logger = structlog.get_logger()


class AMQPPublisher:
    def __init__(self, settings):
        self._settings = settings
        self._connection: Optional[aio_pika.abc.AbstractRobustConnection] = None
        self._channel: Optional[aio_pika.abc.AbstractChannel] = None
        self._exchange: Optional[aio_pika.abc.AbstractExchange] = None
        self._reply_queue: Optional[aio_pika.abc.AbstractQueue] = None
        # Maps correlation_id → asyncio.Future awaiting the agent reply
        self._futures: dict[str, asyncio.Future] = {}

    async def connect(self):
        self._connection = await aio_pika.connect_robust(self._settings.rabbitmq_url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=10)

        # ── Dead-letter exchange + queue ──────────────────────────────────────
        dlx = await self._channel.declare_exchange(
            self._settings.rabbitmq_dlx,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        dlq = await self._channel.declare_queue(
            self._settings.rabbitmq_dlq,
            durable=True,
        )
        await dlq.bind(dlx, routing_key=self._settings.rabbitmq_dlq)

        # ── Main request queue (with DLX routing on rejection/expiry) ─────────
        request_queue = await self._channel.declare_queue(
            self._settings.rabbitmq_request_queue,
            durable=True,
            arguments={
                "x-dead-letter-exchange": self._settings.rabbitmq_dlx,
                "x-dead-letter-routing-key": self._settings.rabbitmq_dlq,
            },
        )

        # ── Direct exchange ───────────────────────────────────────────────────
        self._exchange = await self._channel.declare_exchange(
            self._settings.rabbitmq_exchange,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        await request_queue.bind(
            self._exchange,
            routing_key=self._settings.rabbitmq_request_queue,
        )

        # ── Exclusive reply queue (RPC callback queue) ────────────────────────
        self._reply_queue = await self._channel.declare_queue("", exclusive=True)
        await self._reply_queue.consume(self._on_response, no_ack=True)

        logger.info(
            "amqp_publisher_ready",
            exchange=self._settings.rabbitmq_exchange,
            request_queue=self._settings.rabbitmq_request_queue,
            reply_queue=self._reply_queue.name,
        )

    async def _on_response(self, message: aio_pika.IncomingMessage) -> None:
        """Called by aio-pika when a reply arrives on the exclusive reply queue."""
        cid = message.correlation_id
        if not cid:
            logger.warning("amqp_reply_missing_correlation_id")
            return

        future = self._futures.pop(cid, None)
        if future is None:
            logger.warning("amqp_reply_unknown_correlation", correlation_id=cid)
            return
        if future.done():
            logger.warning("amqp_reply_future_already_done", correlation_id=cid)
            return

        future.set_result(message.body.decode())

    async def rpc_call(self, session_id: str, text: str) -> str:
        """
        Publish a user message to the agent and await the LLM reply.
        Raises HTTPException on publish failure or timeout.
        """
        if self._exchange is None or self._reply_queue is None:
            raise HTTPException(
                status_code=503,
                detail={"error": "AMQP publisher not connected", "service": "publisher"},
            )

        cid = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._futures[cid] = future

        payload = json.dumps({"session_id": session_id, "text": text})

        try:
            await self._exchange.publish(
                aio_pika.Message(
                    body=payload.encode(),
                    correlation_id=cid,
                    reply_to=self._reply_queue.name,
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    content_type="application/json",
                ),
                routing_key=self._settings.rabbitmq_request_queue,
            )
            logger.info(
                "amqp_rpc_published",
                session_id=session_id,
                correlation_id=cid,
            )
        except Exception as exc:
            self._futures.pop(cid, None)
            logger.error("amqp_publish_failed", error=str(exc))
            raise HTTPException(
                status_code=502,
                detail={"error": f"Failed to enqueue message: {exc}", "service": "publisher"},
            )

        try:
            result = await asyncio.wait_for(
                future,
                timeout=float(self._settings.rpc_timeout_seconds),
            )
            logger.info(
                "amqp_rpc_reply_received",
                session_id=session_id,
                correlation_id=cid,
            )
            return result
        except asyncio.TimeoutError:
            self._futures.pop(cid, None)
            logger.error(
                "amqp_rpc_timeout",
                session_id=session_id,
                timeout_s=self._settings.rpc_timeout_seconds,
            )
            raise HTTPException(
                status_code=504,
                detail={
                    "error": (
                        f"Agent did not respond within "
                        f"{self._settings.rpc_timeout_seconds}s"
                    ),
                    "service": "publisher",
                },
            )

    async def close(self) -> None:
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
        logger.info("amqp_publisher_closed")
