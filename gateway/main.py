"""
MedBrief FastAPI Gateway
Routes: GET /health  POST /voice  POST /chat  POST /tts
"""
import logging
import uuid
from contextlib import asynccontextmanager
from urllib.parse import quote

import structlog
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ── Settings ──────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    groq_api_key: str
    llm_model: str = "llama-3.3-70b-versatile"
    stt_model: str = "whisper-large-v3"
    tts_voice: str = "en-US-AriaNeural"

    rabbitmq_url: str = "amqp://guest:guest@rabbitmq:5672/"
    rabbitmq_exchange: str = "agent.direct"
    rabbitmq_request_queue: str = "agent.requests"
    rabbitmq_dlx: str = "agent.dlx"
    rabbitmq_dlq: str = "agent.dead_letters"
    rpc_timeout_seconds: int = 30

    redis_url: str = "redis://redis:6379/0"
    session_ttl: int = 1800
    max_history_turns: int = 20

    mongodb_url: str = "mongodb://mongodb:27017"
    mongodb_db: str = "medbrief"

    log_level: str = "INFO"
    gateway_port: int = 8000

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

# ── Lifespan: connect/disconnect RabbitMQ publisher ───────────────────────────

_publisher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _publisher
    from publisher import AMQPPublisher

    _publisher = AMQPPublisher(settings)
    await _publisher.connect()
    logger.info("gateway_started", rabbitmq_url=settings.rabbitmq_url)
    yield
    await _publisher.close()
    logger.info("gateway_stopped")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="MedBrief Gateway", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id", "X-Response-Text"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    session_id: str
    text: str


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)


class HealthResponse(BaseModel):
    status: str
    service: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", service="gateway")


@app.post("/voice")
async def voice_endpoint(request: Request, audio: UploadFile = File(...)):
    """
    Full voice pipeline:
    1. STT: audio bytes → transcript text (Whisper)
    2. AMQP RPC: publish to agent.requests, await reply
    3. TTS: response text → MP3 audio stream (OpenAI TTS)
    Returns a streaming audio/mpeg response.
    """
    req_id = str(uuid.uuid4())
    log = logger.bind(request_id=req_id, endpoint="/voice")

    session_id = request.headers.get("X-Session-Id") or str(uuid.uuid4())
    log.info("voice_request", session_id=session_id, filename=audio.filename)

    try:
        audio_bytes = await audio.read()
    except Exception as exc:
        log.error("audio_read_failed", error=str(exc))
        raise HTTPException(
            status_code=400,
            detail={"error": "Failed to read uploaded audio", "service": "gateway"},
        )

    from stt import transcribe
    from tts import synthesize

    user_text = await transcribe(
        audio_bytes=audio_bytes,
        filename=audio.filename or "audio.webm",
        settings=settings,
    )
    log.info("stt_done", preview=user_text[:80])

    if not user_text.strip():
        raise HTTPException(
            status_code=422,
            detail={"error": "No speech detected in audio", "service": "stt"},
        )

    response_text = await _publisher.rpc_call(
        session_id=session_id, text=user_text
    )
    log.info("agent_replied", preview=response_text[:80])

    encoded_text = quote(response_text, safe="")
    return StreamingResponse(
        synthesize(text=response_text, settings=settings),
        media_type="audio/mpeg",
        headers={
            "X-Session-Id": session_id,
            "X-Response-Text": encoded_text,
            "X-User-Text": quote(user_text, safe=""),
        },
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest, request: Request):
    """Text-only endpoint for testing without audio hardware."""
    req_id = str(uuid.uuid4())
    log = logger.bind(request_id=req_id, endpoint="/chat", session_id=body.session_id)
    log.info("chat_request", preview=body.text[:80])

    response_text = await _publisher.rpc_call(
        session_id=body.session_id, text=body.text
    )
    log.info("agent_replied", preview=response_text[:80])
    return ChatResponse(session_id=body.session_id, text=response_text)


@app.post("/tts")
async def tts_endpoint(body: TTSRequest):
    """Standalone TTS endpoint — converts text to streaming MP3."""
    log = logger.bind(endpoint="/tts")
    log.info("tts_request", chars=len(body.text))

    from tts import synthesize

    return StreamingResponse(
        synthesize(text=body.text, settings=settings),
        media_type="audio/mpeg",
    )


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "unhandled_exception",
        path=str(request.url.path),
        error=str(exc),
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": {"error": "Internal server error", "service": "gateway"}},
    )
