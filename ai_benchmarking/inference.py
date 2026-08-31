import asyncio
import json
import os
import random
import time

from google import genai
from google.genai import types
from openai import AsyncOpenAI
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, Field


def create_google_genai_client():
    """Create a google-genai Client using an API key or Application Default Credentials.

    If GEMINI_API_KEY or GOOGLE_API_KEY is set, use the Gemini Developer API.
    Otherwise omit the API key so the SDK can discover ADC (typically Vertex AI
    via GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION).
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    # Try using the more modern Application Default Credentials (ADC) for auth
    kwargs = {}
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    if project and location:
        kwargs["enterprise"] = True
        kwargs["project"] = project
        kwargs["location"] = location
    return genai.Client(**kwargs)


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
    model: str = "gemini-1.5-pro",
    cache_name: str = None,
    fallback_prompt: str = "",
    client=None,  # Shared persistent connection pool passed from eval.py
) -> dict:
    """Executes target string classification across isolated token-cached frameworks."""

    # Fallback storage variables
    raw_content = ""
    p_tokens = 0
    c_tokens = 0
    cached_tokens = 0
    api_latency = 0.0

    # 1. STRUCTURAL ISOLATION FENCE
    # Wraps the raw query string inside XML delimiters to prevent pattern autocomplete loops
    formatted_query = f"Classify this specific user target query string:\n<target_query>{query}</target_query>"

    # -----------------------------------------------------------------------
    # PROVIDER METRICS LAYER: OPENAI
    # -----------------------------------------------------------------------
    if provider == "openai":
        # Fallback local client instantiation if master connection pool isn't passed down
        local_client = (
            client if client else AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        )

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
        # Record only the time taken for the API request
        api_latency = time.time() - start_time

        raw_content = response.choices[0].message.content
        p_tokens = response.usage.prompt_tokens
        c_tokens = response.usage.completion_tokens

    # -----------------------------------------------------------------------
    # PROVIDER METRICS LAYER: GEMINI (EXPLICIT CONTEXT CACHING ACTIVE)
    # -----------------------------------------------------------------------
    elif provider == "gemini":
        if client is None:
            raise ValueError(
                "Production execution requires an active master singleton client pool handle."
            )

        # Resilient network parameters for heavy concurrency loads
        max_retries = 8
        initial_delay = 1.0

        # Define safety settings to prevent standard filters from blocking clinical triage strings
        safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
            ),
        ]

        # Configure base payload using the new unified SDK
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

        # Resilient network loop execution layer
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = await client.aio.models.generate_content(
                    model=model, contents=formatted_query, config=config_payload
                )
                # Record only the time taken for the successful API request
                api_latency = time.time() - start_time

                raw_content = response.text

                # CORE MATH EQUATION: Subtract cached subset volume from total input mass
                total_prompt_sum = response.usage_metadata.prompt_token_count or 0
                cached_tokens = (
                    getattr(response.usage_metadata, "cached_content_token_count", 0)
                    or 0
                )

                # Ensure standard billing metrics are only charged for the new query tokens
                p_tokens = total_prompt_sum - cached_tokens
                c_tokens = response.usage_metadata.candidates_token_count or 0
                break

            except Exception as api_err:
                if "Too many open files" in str(api_err):
                    print(
                        "!!! OS Socket Exhaustion encountered. Retrying execution context frame..."
                    )

                # Bubble connection fault back to eval orchestrator if max retries hit
                if attempt == max_retries - 1:
                    return {
                        "error": f"API connection failure after {max_retries} attempts: {str(api_err)}",
                        "cached_tokens": 0,
                    }

                # Jittered Exponential Backoff
                sleep_duration = (initial_delay * (2**attempt)) + random.uniform(
                    0.1, 1.0
                )
                await asyncio.sleep(sleep_duration)

    # -----------------------------------------------------------------------
    # PRODUCTION COMPILATION & DATA SAFETY RAIL
    # -----------------------------------------------------------------------
    try:
        parsed_json = json.loads(raw_content)

        # Captures array output formats and safely extracts the first item dictionary
        if isinstance(parsed_json, list):
            parsed_json = (
                parsed_json[0]
                if len(parsed_json) > 0 and isinstance(parsed_json[0], dict)
                else {}
            )

        if not isinstance(parsed_json, dict):
            parsed_json = {}

    except Exception:
        # Fallback parsing defaults in case of a malformed generation payload
        parsed_json = {}

    return {
        "reasoning": "Skipped for production optimization",
        "risk_level": int(parsed_json.get("risk_level", 0)),
        "latency": api_latency,
        "prompt_tokens": p_tokens,
        "completion_tokens": c_tokens,
        "cached_tokens": cached_tokens,  # Returned to the eval loop for progress bar aggregation
    }
