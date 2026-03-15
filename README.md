# MedBrief — Voice Clinical Protocol Assistant

MedBrief is a voice-driven, multi-turn AI assistant that lets healthcare workers query
clinical protocol summaries hands-free. Speak a question into the browser, and MedBrief
transcribes it with OpenAI Whisper, routes it through a RabbitMQ message queue to a
GPT-4o-powered agent worker, searches PubMed for relevant evidence when needed, and
streams the spoken response back within seconds — all while remembering the context of
the conversation so follow-up questions work naturally. The architecture mirrors
Haptik's Contakt LLM platform: FastAPI gateway, async message queuing, Redis hot-memory,
and MongoDB cold persistence, Dockerised and runnable from a single command.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser / Client                         │
│   Mic Input → JS MediaRecorder → POST /voice → Speaker Output  │
└────────────────────┬───────────────────────────────────────────┘
                     │ audio blob (webm/wav)
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Gateway  :8000                      │
│   /voice  → STT (Whisper API)  → text                          │
│   /chat   → publish to RabbitMQ exchange: agent.requests       │
│   /tts    → TTS (OpenAI TTS API) → audio stream back           │
└────────────────────┬───────────────────────────────────────────┘
                     │ AMQP message
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│               Agent Worker (Consumer)                           │
│   Consumes from: agent.requests                                 │
│   1. Load conversation history from Redis (session_id key)      │
│   2. Append new user turn                                       │
│   3. Call GPT-4o with system prompt + history                   │
│   4. If tool_call detected → call external API (PubMed)        │
│   5. Append assistant turn to Redis                             │
│   6. Publish response to agent.responses (reply_to routing)     │
└────────────────────┬───────────────────────────────────────────┘
                     │ stores turns
                     ▼
┌──────────────┐   ┌──────────────────────────────────────────────┐
│    Redis     │   │              MongoDB                         │
│  (hot conv   │   │  Persists completed sessions, analytics,     │
│   history)   │   │  tool call logs for post-analysis            │
└──────────────┘   └──────────────────────────────────────────────┘
```

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose v2)
- An [OpenAI API key](https://platform.openai.com/api-keys) with access to `gpt-4o`, `whisper-1`, and `tts-1`
- A modern browser (Chrome or Firefox recommended for MediaSource streaming)

---

## Quickstart

```bash
# 1. Copy the env template
cp .env.example .env

# 2. Open .env and set your OpenAI API key
#    OPENAI_API_KEY=sk-...

# 3. Build and start all services
make run

# 4. Watch services come up healthy
make ps

# 5. Open the UI
#    http://localhost
#
# 6. RabbitMQ management console (guest / guest)
#    http://localhost:15672
```

To stop:
```bash
make stop          # stop containers, keep data volumes
make clean         # stop + delete all volumes (fresh start)
```

---

## How It Works

### FastAPI Gateway (`gateway/`)

The gateway is the sole entry point for the browser. On `POST /voice` it reads the
uploaded audio blob, sends it to OpenAI Whisper for transcription, then publishes the
transcript to RabbitMQ using the AMQP RPC pattern: the message carries a `correlation_id`
and a `reply_to` exclusive queue name. The gateway suspends via `asyncio.wait_for` until
the agent posts the reply back, then feeds the response text into OpenAI TTS and streams
the MP3 back to the browser as a chunked `StreamingResponse`. The `lifespan` context
manager ensures the AMQP connection is created and torn down cleanly with the server.

### Agent Worker (`agent/`)

The worker consumes messages from `agent.requests`. For each message it loads the
session's conversation history from Redis, calls GPT-4o (with the full history as
context for multi-turn memory), handles any `tool_calls` by hitting the PubMed
E-utilities API, then appends the new turns to both Redis (hot memory for the next
turn) and MongoDB (cold audit trail). The reply is published to the gateway's temporary
reply queue identified by `reply_to`. Any unhandled exception causes the message to be
nack'd and routed to the dead-letter queue `agent.dead_letters` rather than silently
dropped.

### Redis Memory (`agent/memory.py`)

Conversation history lives under `session:{session_id}:history` as a Redis List.
Turns are pushed newest-first (`LPUSH`) so `LTRIM` can keep the newest 20 entries
efficiently. A pipeline executes `LPUSH + LTRIM + EXPIRE` atomically on every turn
append. The 30-minute TTL means an idle session automatically expires without manual
cleanup — matching Haptik's stateless session design.

### MongoDB Persistence (`agent/db.py`)

Completed turns are written to the `sessions` collection in the `medbrief` database.
Each session document holds `started_at`, `ended_at`, the full `turns` array (with
`tool_calls` on assistant turns for auditability), and metadata. `ensure_session` uses
`$setOnInsert` with `upsert=True` so the first message atomically creates the document
without overwriting it on subsequent turns.

---

## Design Decisions

### Why RabbitMQ instead of calling the LLM directly from the HTTP handler?

A synchronous LLM call blocks the gateway coroutine for 2–20 seconds and couples
availability of the HTTP tier to availability of the LLM. By publishing to RabbitMQ
first, the gateway immediately frees the event loop, the agent worker can be scaled
horizontally (multiple consumer replicas), failed messages land in a dead-letter queue
instead of timing out silently, and the broker provides backpressure when demand
exceeds worker capacity. This is precisely the async pipeline architecture Haptik uses
in its Contakt platform for enterprise-scale concurrent conversations.

### Why Redis for conversation memory instead of MongoDB?

Conversation state is hot (read on every turn), small (20 JSON objects per session),
and temporary (expires after 30 minutes of inactivity). Redis gives sub-millisecond
reads, atomic pipeline operations for LPUSH + LTRIM + EXPIRE, and automatic expiry
with no background job needed. MongoDB handles cold storage: it has richer query
support, durable replication, and is suited to the sparse, analytical access pattern
of session post-processing and tool-call auditing. This two-tier strategy mirrors
Haptik's production data architecture.

---

## Known Limitations and Stretch Goals

### Known limitations

- TTS playback falls back to blob-URL (collect-then-play) in Safari because Safari
  does not support `MediaSource` with `audio/mpeg`. A WebSocket-based streaming
  endpoint would resolve this cross-browser.
- The `/voice` endpoint chains STT → AMQP RPC → TTS sequentially; end-to-end latency
  is roughly 3–8 s depending on OpenAI API response times.
- No authentication layer — suitable for local / demo use only.
- PubMed E-utilities returns at most 3 abstracts; the free tier is rate-limited to
  3 requests/second without `PUBMED_API_KEY`.

### Stretch goals

- [ ] WebSocket endpoint replacing the polling fetch loop for true real-time
      streaming of both transcript and audio
- [ ] Language detection + multilingual TTS (Haptik supports 135 languages;
      OpenAI TTS supports multiple languages natively)
- [ ] Agent Co-Pilot mode: post a Slack webhook summary after each conversation ends
- [ ] Prometheus `/metrics` endpoint: request count, LLM latency histogram,
      RabbitMQ queue depth
- [ ] Redis token-bucket rate limiting per `session_id` to prevent abuse

---

## Interview Talking Points

### 1. Why async message queuing instead of a direct LLM call from the gateway?

Haptik processes millions of conversations. A synchronous LLM call blocks the gateway
thread and cannot scale. RabbitMQ decouples ingestion from processing, enables retries
on failure, and allows multiple worker instances — identical to how Haptik's bot
pipeline handles concurrent enterprise clients. In this project, `publisher.py` uses
the AMQP RPC pattern with `correlation_id` and `reply_to` to simulate synchronous
behaviour to the HTTP caller while keeping the actual LLM call fully asynchronous.

### 2. Why Redis for conversation memory instead of a database?

Conversation state is hot, small, and temporary. Redis gives sub-millisecond reads
with automatic TTL-based expiry. MongoDB is for cold, persistent analytics. This
mirrors Haptik's two-tier data strategy across their Contakt platform. Using
`LPUSH + LTRIM + EXPIRE` in a single pipeline is an atomic, O(1) operation — there
is no cheaper way to maintain a bounded, time-expiring rolling window of messages.

### 3. How would you scale this to Haptik's production load (10 B+ conversations)?

Horizontal agent workers consuming from the same `agent.requests` queue — RabbitMQ
distributes messages round-robin, so adding replicas linearly increases throughput.
Redis Cluster shards session keys by `session_id` hash slot. MongoDB shards the
`sessions` collection on `session_id`. RabbitMQ queue mirroring (or quorum queues)
removes the broker as a SPOF. Kubernetes HPA watches queue depth via a Prometheus
exporter and scales the agent `Deployment` automatically — exactly the stack Haptik
runs on Azure AKS.
