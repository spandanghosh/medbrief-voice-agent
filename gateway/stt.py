"""
STT wrapper — transcribes audio bytes to text via Groq Whisper API (free tier).
Groq exposes an OpenAI-compatible endpoint; we use the openai SDK with a
custom base_url pointing at api.groq.com.  No card required.
"""
import structlog
from fastapi import HTTPException
from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

logger = structlog.get_logger()

_CONTENT_TYPE_MAP = {
    ".webm": "audio/webm",
    ".wav": "audio/wav",
    ".mp4": "audio/mp4",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
}


def _content_type(filename: str) -> str:
    for ext, ct in _CONTENT_TYPE_MAP.items():
        if filename.lower().endswith(ext):
            return ct
    return "audio/webm"


async def transcribe(audio_bytes: bytes, filename: str, settings) -> str:
    """
    Send audio bytes to Whisper and return the transcript string.
    Raises HTTPException on API or connection failure.
    """
    client = AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    ct = _content_type(filename)

    logger.info(
        "stt_request",
        filename=filename,
        content_type=ct,
        bytes=len(audio_bytes),
        model=settings.stt_model,
    )

    try:
        response = await client.audio.transcriptions.create(
            model=settings.stt_model,
            file=(filename, audio_bytes, ct),
        )
        transcript = response.text.strip()
        logger.info("stt_success", chars=len(transcript), preview=transcript[:60])
        return transcript

    except RateLimitError as exc:
        logger.error("stt_rate_limit", error=str(exc))
        raise HTTPException(
            status_code=429,
            detail={"error": "Groq Whisper rate limit reached, retry shortly", "service": "stt"},
        )
    except APIConnectionError as exc:
        logger.error("stt_connection_error", error=str(exc))
        raise HTTPException(
            status_code=502,
            detail={"error": "Cannot reach Groq STT service", "service": "stt"},
        )
    except APIError as exc:
        logger.error("stt_api_error", status_code=exc.status_code, error=str(exc))
        raise HTTPException(
            status_code=502,
            detail={"error": f"Groq Whisper error: {exc.message}", "service": "stt"},
        )
    except Exception as exc:
        logger.error("stt_unexpected", error=str(exc), error_type=type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail={"error": f"STT failed: {exc}", "service": "stt"},
        )
