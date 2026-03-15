# MedBrief — Voice Clinical Protocol Assistant

MedBrief is a voice-driven, multi-turn AI assistant that lets healthcare workers query
clinical protocol summaries hands-free. Speak a question into the browser, and MedBrief
transcribes it with Groq Whisper, routes it through a RabbitMQ message queue to a
Llama 3.3 70B-powered agent worker, searches PubMed for relevant evidence when needed,
and streams the spoken response back within seconds — all while remembering the context
of the conversation so follow-up questions work naturally.

Fully Dockerised, runs from a single command, and uses the **Groq free tier** throughout
— no credit card required.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser / Client                         │
│   Mic Input → JS MediaRecorder → POST /voice → Speaker Output  │
└────────────────────┬───────────────────────────────────────────┘
                     │ audio blob (webm)
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Gateway  :8000                      │
│   /voice  → STT (Groq Whisper)  → text                         │
│   /chat   → publish to RabbitMQ exchange: agent.requests       │
│   /tts    → TTS (edge-tts, Microsoft neural) → audio stream    │
└────────────────────┬───────────────────────────────────────────┘
                     │ AMQP message
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│               Agent Worker (Consumer)                           │
│   Consumes from: agent.requests                                 │
│   1. Load conversation history from Redis (session_id key)      │
│   2. Append new user turn                                       │
│   3. Call Llama 3.3 70B (Groq) with system prompt + history    │
│   4. If tool_call detected → call external API (PubMed)        │
│   5. Append assistant turn to Redis                             │
│   6. Publish response to reply_to queue (RPC pattern)          │
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
- A [Groq API key](https://console.groq.com) — free tier, no credit card required
- Chrome or Firefox (recommended for MediaSource audio streaming)

---

## Quickstart

```bash
# 1. Copy the env template
cp .env.example .env       # Mac/Linux
# Windows PowerShell:
# Copy-Item .env.example .env

# 2. Open .env and set your Groq API key
#    GROQ_API_KEY=gsk_...

# 3. Build and start all services
docker compose up --build -d

# 4. Watch services come up healthy
docker compose ps

# 5. Open the UI
#    http://localhost
#
# 6. RabbitMQ management console (guest / guest)
#    http://localhost:15672
```

To stop:
```bash
docker compose down          # stop containers, keep data volumes
docker compose down -v       # stop + delete all volumes (fresh start)
```

---

## AI Stack

| Component | Service | Model | Cost |
|-----------|---------|-------|------|
| STT | Groq | `whisper-large-v3` | Free tier |
| LLM | Groq | `llama-3.3-70b-versatile` | Free tier |
| TTS | edge-tts (Microsoft neural) | `en-US-AriaNeural` | Free, no key |
| Tool | NCBI PubMed E-utilities | — | Free, no key |

---

## How It Works

### FastAPI Gateway (`gateway/`)

The gateway is the sole HTTP entry point. On `POST /voice` it reads the uploaded audio
blob, sends it to Groq Whisper for transcription, then publishes the transcript to
RabbitMQ using the AMQP RPC pattern: the message carries a `correlation_id` and a
`reply_to` exclusive queue name. The gateway suspends via `asyncio.wait_for` until the
agent posts the reply, then feeds the response text into edge-tts and streams the MP3
back as a chunked `StreamingResponse`. A `lifespan` context manager manages the AMQP
connection lifecycle cleanly.

### Agent Worker (`agent/`)

The worker consumes messages from `agent.requests`. For each message it loads the
session's conversation history from Redis, calls Llama 3.3 70B on Groq (with the full
history for multi-turn memory), handles any `tool_calls` by hitting the PubMed
E-utilities API, then appends turns to both Redis (hot memory) and MongoDB (cold audit
trail). The reply is published back to the gateway's temporary reply queue. Any
unhandled exception nacks the message → routed via dead-letter exchange `agent.dlx`
to the dead-letter queue `agent.dead_letters`. A regex filter strips function-call markup that Llama occasionally
leaks into response text before the reply is sent.

### Redis Memory (`agent/memory.py`)

Conversation history lives under `session:{session_id}:history` as a Redis List.
Turns are pushed newest-first (`LPUSH`), trimmed to 20 entries (`LTRIM`), and the TTL
is refreshed — all in a single atomic pipeline on every append. Sessions expire
automatically after 30 minutes of inactivity.

### MongoDB Persistence (`agent/db.py`)

Turns are written to the `sessions` collection in the `medbrief` database. Each session
document holds `started_at`, `ended_at`, the full `turns` array with `tool_calls` on
assistant turns for auditability, and metadata. `ensure_session` uses `$setOnInsert`
with `upsert=True` so concurrent first-turn calls are safe.

---

## Design Decisions

### Why RabbitMQ instead of calling the LLM directly from the HTTP handler?

A synchronous LLM call blocks the gateway coroutine for several seconds and couples
HTTP availability to LLM availability. Publishing to RabbitMQ first frees the event
loop immediately, allows the agent worker to scale horizontally, routes failed messages
to a dead-letter queue instead of dropping them silently, and gives the broker
natural backpressure when load is high.

### Why Redis for conversation memory instead of MongoDB?

Conversation state is hot (read on every turn), small (≤20 JSON objects), and
temporary (expires after 30 minutes). Redis gives sub-millisecond reads and atomic
`LPUSH + LTRIM + EXPIRE` in a single pipeline with no background cleanup needed.
MongoDB is used for cold storage where its richer query support and durable
replication are actually needed — session analytics and tool-call audit logs.

### Why Groq + edge-tts instead of a paid AI service?

Groq exposes an OpenAI-compatible API, so the swap required only changing `base_url`
in the existing `openai` SDK client — no rewrite. The free tier handles demo workloads
comfortably. edge-tts uses Microsoft's neural TTS with no key or billing required,
producing quality comparable to paid TTS APIs.

---

## Known Limitations and Stretch Goals

### Known limitations

- Safari does not support `MediaSource` with `audio/mpeg`; playback falls back to
  collect-then-play via a Blob URL.
- End-to-end latency is roughly 3–8 s (STT → AMQP RPC → LLM → TTS chained sequentially).
- No authentication layer — for local / demo use only.
- PubMed E-utilities returns at most 3 abstracts and is rate-limited to 3 req/s
  without a `PUBMED_API_KEY`.
- Llama 3.x occasionally leaks function-call markup into response text; a regex
  filter in `agent/llm.py` strips it before the reply is sent.

### Stretch goals

- [ ] WebSocket endpoint for true real-time streaming of transcript and audio
- [ ] Language detection + multilingual TTS (edge-tts supports 40+ locales)
- [ ] Session summary posted to a Slack webhook after each conversation ends
- [ ] Prometheus `/metrics` endpoint: request count, LLM latency, queue depth
- [ ] Redis token-bucket rate limiting per `session_id`

---

## Architecture Q&A

### Why async message queuing instead of a direct LLM call from the gateway?

Decoupling ingestion from processing means the HTTP tier stays responsive regardless
of LLM latency. Workers can be scaled horizontally — multiple consumers on the same
queue process requests in parallel. Failed messages land in a dead-letter queue rather
than vanishing. The broker absorbs traffic spikes naturally.

### Why Redis for conversation memory instead of a database?

Hot, small, temporary data is exactly what Redis is built for. Sub-millisecond reads,
atomic bounded-list operations (`LPUSH + LTRIM + EXPIRE`), and automatic TTL expiry
with zero background jobs. The database handles durable, queryable, cold storage — a
different access pattern entirely.

### How would you scale this for high concurrency?

Horizontal agent workers share the same `agent.requests` queue — RabbitMQ distributes
round-robin, so throughput scales linearly with replicas. Redis Cluster shards session
keys by hash slot. MongoDB shards the `sessions` collection on `session_id`. Quorum
queues remove RabbitMQ as a single point of failure. Kubernetes HPA on queue depth
metrics handles auto-scaling automatically.
