"""
TTS wrapper — converts text to streaming MP3 audio via edge-tts.
edge-tts uses Microsoft Edge's neural TTS service — completely free,
no API key required, high-quality voices.
Returns an async generator of bytes for FastAPI StreamingResponse.
"""
from typing import AsyncIterator

import edge_tts
import structlog
from fastapi import HTTPException

logger = structlog.get_logger()


async def synthesize(text: str, settings) -> AsyncIterator[bytes]:
    """
    Async generator that yields MP3 audio chunks from edge-tts.
    Caller should wrap in FastAPI StreamingResponse(media_type="audio/mpeg").
    """
    logger.info(
        "tts_request",
        chars=len(text),
        voice=settings.tts_voice,
        preview=text[:60],
    )

    try:
        communicate = edge_tts.Communicate(text=text, voice=settings.tts_voice)
        total_bytes = 0
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                data: bytes = chunk["data"]
                total_bytes += len(data)
                yield data
        logger.info("tts_success", total_bytes=total_bytes)

    except Exception as exc:
        logger.error("tts_failed", error=str(exc), error_type=type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail={"error": f"TTS failed: {exc}", "service": "tts"},
        )
