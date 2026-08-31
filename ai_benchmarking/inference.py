import asyncio
import json
import os
import random
import time
from typing import Any, Iterable, cast

from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceToolParam,
    ToolParam,
    ToolUseBlock,
)
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# CLIENT FACTORY FUNCTIONS
# ---------------------------------------------------------------------------


def create_google_genai_client(api_key: str | None = None) -> genai.Client:
    """Create a google-genai Client using an API key or Application Default Credentials."""
    final_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if final_key:
        return genai.Client(api_key=final_key)

    kwargs: dict[str, Any] = {}
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    if project and location:
        kwargs["enterprise"] = True
        kwargs["project"] = project
        kwargs["location"] = location
    return genai.Client(**kwargs)


def create_openai_client(api_key: str | None = None) -> AsyncOpenAI:
    """Create an AsyncOpenAI Client using an explicit key or OPENAI_API_KEY env variable."""
    return AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))


def create_anthropic_client(api_key: str | None = None) -> AsyncAnthropic:
    """Create an AsyncAnthropic Client using an explicit key or ANTHROPIC_API_KEY env variable."""
    return AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# STRUCTURED RESPONSE SCHEMA
# ---------------------------------------------------------------------------


class RiskResponse(BaseModel):
    risk_level: int = Field(
        description="The predicted C-SSRS risk score as an integer from 0 to 8"
    )


# Precompile strict schema for OpenAI structured outputs
OPENAI_STRICT_SCHEMA = to_strict_json_schema(RiskResponse)


# ---------------------------------------------------------------------------
# CORE ASYNC MULTI-PROVIDER EVALUATION ENGINE
# ---------------------------------------------------------------------------


async def generate_ai_response_async(
    query: str,
    provider: str = "gemini",
    model: str = "gemini-3.6-flash",
    cache_name: str | None = None,
    fallback_prompt: str = "",
    client: Any | None = None,  # Shared persistent connection pool passed from eval.py
) -> dict:
    """Executes target string classification across isolated token-cached frameworks."""

    # Fallback storage variables
    raw_content = ""
    p_tokens = 0
    c_tokens = 0
    cached_tokens = 0
    api_latency = 0.0

    # 1. STRUCTURAL ISOLATION FENCE
    formatted_query = f"Classify this specific user target query string:\n<target_query>{query}</target_query>"

    # -----------------------------------------------------------------------
    # PROVIDER METRICS LAYER: OPENAI
    # -----------------------------------------------------------------------
    if provider == "openai":
        local_client = client if client else create_openai_client()

        start_time = time.time()
        response = await local_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": fallback_prompt},
                {"role": "user", "content": formatted_query},
            ],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "risk_response",
                    "schema": OPENAI_STRICT_SCHEMA,
                    "strict": True,
                },
            },
        )
        api_latency = time.time() - start_time

        raw_content = (
            str(response.choices[0].message.content)
            if response.choices[0].message.content
            else ""
        )
        if response.usage is not None:
            p_tokens = response.usage.prompt_tokens
            c_tokens = response.usage.completion_tokens

    # -----------------------------------------------------------------------
    # PROVIDER METRICS LAYER: ANTHROPIC (WITH PROMPT CACHING & TOOL USE)
    # -----------------------------------------------------------------------
    elif provider == "anthropic":
        local_client = client if client else create_anthropic_client()

        tool_definition = cast(
            ToolParam,
            {
                "name": "risk_response",
                "description": "Record the predicted C-SSRS risk classification score.",
                "input_schema": RiskResponse.model_json_schema(),
            },
        )

        system_input: str | list[TextBlockParam] = (
            [
                cast(
                    TextBlockParam,
                    {
                        "type": "text",
                        "text": fallback_prompt,
                        "cache_control": {"type": "ephemeral"},
                    },
                )
            ]
            if fallback_prompt
            else ""
        )

        tool_choice = cast(
            ToolChoiceToolParam, {"type": "tool", "name": "risk_response"}
        )
        messages = cast(
            Iterable[MessageParam], [{"role": "user", "content": formatted_query}]
        )

        start_time = time.time()
        response = await local_client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_input,
            tools=[tool_definition],
            tool_choice=tool_choice,
            messages=messages,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        api_latency = time.time() - start_time

        for content_block in response.content:
            if (
                isinstance(content_block, ToolUseBlock)
                and content_block.name == "risk_response"
            ):
                raw_content = json.dumps(content_block.input)
                break

        if response.usage is not None:
            p_tokens = response.usage.input_tokens or 0
            c_tokens = response.usage.output_tokens or 0
            cached_tokens = getattr(response.usage, "cache_read_input_tokens", 0) or 0

    # -----------------------------------------------------------------------
    # PROVIDER METRICS LAYER: GEMINI (EXPLICIT CONTEXT CACHING ACTIVE)
    # -----------------------------------------------------------------------
    elif provider == "gemini":
        local_client = client if client else create_google_genai_client()

        max_retries = 8
        initial_delay = 1.0

        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        config_payload = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=RiskResponse,
            safety_settings=safety_settings,
        )

        if cache_name:
            config_payload.cached_content = cache_name
        else:
            config_payload.system_instruction = fallback_prompt

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = await local_client.aio.models.generate_content(
                    model=model, contents=formatted_query, config=config_payload
                )
                api_latency = time.time() - start_time

                raw_content = str(response.text) if response.text else ""

                if response.usage_metadata:
                    total_prompt_sum = response.usage_metadata.prompt_token_count or 0
                    cached_tokens = (
                        getattr(
                            response.usage_metadata, "cached_content_token_count", 0
                        )
                        or 0
                    )
                    p_tokens = total_prompt_sum - cached_tokens
                    c_tokens = response.usage_metadata.candidates_token_count or 0

                break

            except Exception as api_err:
                if "Too many open files" in str(api_err):
                    print(
                        "!!! OS Socket Exhaustion encountered. Retrying execution context frame..."
                    )

                if attempt == max_retries - 1:
                    return {
                        "error": f"API connection failure after {max_retries} attempts: {str(api_err)}",
                        "cached_tokens": 0,
                    }

                sleep_duration = (initial_delay * (2**attempt)) + random.uniform(
                    0.1, 1.0
                )
                await asyncio.sleep(sleep_duration)

    # -----------------------------------------------------------------------
    # PRODUCTION COMPILATION & DATA SAFETY RAIL
    # -----------------------------------------------------------------------
    try:
        parsed_json = json.loads(raw_content) if raw_content else {}

        if isinstance(parsed_json, list):
            parsed_json = (
                parsed_json[0]
                if len(parsed_json) > 0 and isinstance(parsed_json[0], dict)
                else {}
            )

        if not isinstance(parsed_json, dict):
            parsed_json = {}

    except Exception:
        parsed_json = {}

    return {
        "reasoning": "Skipped for production optimization",
        "risk_level": int(parsed_json.get("risk_level", 0)),
        "latency": api_latency,
        "prompt_tokens": p_tokens,
        "completion_tokens": c_tokens,
        "cached_tokens": cached_tokens,
    }
