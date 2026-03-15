"""
Groq LLM client with tool calling (free tier).
Uses the OpenAI SDK pointed at api.groq.com (OpenAI-compatible endpoint).
Default model: llama-3.3-70b-versatile (free, supports tool calling).

System prompt encodes:
  - Domain scope: clinical protocols only
  - Safety guardrails: no dosage prescriptions, no hallucination, decline off-topic
  - Response format: voice-optimised (no markdown, ≤3 sentences, ≤300 tokens)

Tool calling loop:
  - First completion may return finish_reason="tool_calls"
  - Worker dispatches to tools.py, appends tool result messages
  - Re-calls the model to produce the final user-facing text
  - All tool invocations are logged to MongoDB for auditability
"""
import json
import re
from typing import Optional

import structlog
from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

from tools import search_protocol

logger = structlog.get_logger()

SYSTEM_PROMPT = """You are MedBrief, a clinical protocol assistant serving healthcare \
workers in hospital and clinical settings.

Domain scope: Evidence-based medical protocols, drug information, treatment guidelines, \
and clinical procedures only.

Safety rules (non-negotiable):
1. Never state specific dosages as definitive prescriptions — always say they must be \
verified against current formulary or prescriber guidelines.
2. If you are uncertain about a clinical fact, say so explicitly. Never fabricate \
medical information, drug names, or statistics.
3. Politely decline questions that fall outside clinical protocol scope \
(e.g. personal advice, diagnoses, billing questions).
4. Do not identify individual patients or make patient-specific recommendations.

Response format (critical for voice output):
- Maximum 3 short sentences.
- Plain English only — no markdown, no bullet points, no headers, no numbered lists.
- Suitable for text-to-speech: avoid symbols, abbreviations the listener might not know.
- If the search_protocol tool returns useful results, summarise them in plain language."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_protocol",
            "description": (
                "Search PubMed for evidence-based clinical protocol summaries, "
                "drug information, and treatment guidelines. Use this when asked "
                "about specific medications, dosing protocols, clinical procedures, "
                "or evidence-based guidelines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Clinical search query, e.g. "
                            "'sepsis management bundle protocol' or "
                            "'metformin dosing guidelines type 2 diabetes'"
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    }
]


# Llama 3.x on Groq sometimes leaks raw function-call markup into the
# content field alongside (or instead of) the proper tool_calls structure.
# Strip it so the voice response stays clean.
_TOOL_ARTIFACT_RE = re.compile(r"<function=\w+>.*?</function>", re.DOTALL)


def _strip_tool_artifacts(text: str) -> str:
    return _TOOL_ARTIFACT_RE.sub("", text).strip()


class LLMClient:
    def __init__(self, settings):
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    async def chat(
        self,
        history: list[dict],
        user_text: str,
    ) -> tuple[str, list[dict]]:
        """
        Send conversation history + new user message to GPT-4o.
        Handles the tool-call loop internally.
        Returns (response_text, tool_calls_log).
        tool_calls_log entries are stored in MongoDB for auditability.
        """
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + history
            + [{"role": "user", "content": user_text}]
        )
        tool_calls_log: list[dict] = []

        response = await self._call_llm(messages)
        logger.info(
            "llm_initial_response",
            finish_reason=response.choices[0].finish_reason,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )

        # ── Tool-call loop ────────────────────────────────────────────────────
        while response.choices[0].finish_reason == "tool_calls":
            assistant_msg = response.choices[0].message
            # Append the assistant's tool-call request to the message thread
            messages.append(assistant_msg.model_dump(exclude_none=True))

            tool_result_messages: list[dict] = []

            for tc in assistant_msg.tool_calls or []:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                logger.info("llm_tool_call", tool=fn_name, args=fn_args)

                if fn_name == "search_protocol":
                    try:
                        result = await search_protocol(
                            query=fn_args["query"],
                            settings=self._settings,
                        )
                    except Exception as exc:
                        result = f"Tool error: {exc}"
                        logger.error("tool_execution_failed", tool=fn_name, error=str(exc))
                else:
                    result = f"Unknown tool '{fn_name}' — cannot execute."
                    logger.warning("llm_unknown_tool", tool=fn_name)

                tool_calls_log.append(
                    {
                        "tool": fn_name,
                        "args": fn_args,
                        "result_preview": result[:200],
                        "tool_call_id": tc.id,
                    }
                )

                tool_result_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

            messages.extend(tool_result_messages)
            response = await self._call_llm(messages)
            logger.info(
                "llm_post_tool_response",
                finish_reason=response.choices[0].finish_reason,
            )

        final_text = _strip_tool_artifacts(response.choices[0].message.content or "")
        return final_text, tool_calls_log

    async def _call_llm(self, messages: list[dict]):
        """Single LLM API call with error handling."""
        try:
            return await self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=300,
                temperature=0.3,
            )
        except RateLimitError as exc:
            logger.error("llm_rate_limit", error=str(exc))
            raise
        except APIConnectionError as exc:
            logger.error("llm_connection_error", error=str(exc))
            raise
        except APIError as exc:
            logger.error("llm_api_error", status_code=exc.status_code, error=str(exc))
            raise
        except Exception as exc:
            logger.error("llm_unexpected", error=str(exc))
            raise
