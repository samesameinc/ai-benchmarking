# Updated imports in scripts/inference.py
import os
import time
import json
from openai import OpenAI
from google import genai
from google.genai import types
import anthropic


def generate_ai_response(query: str, provider: str = "openai", model: str = "gpt-4o") -> dict:
    start_time = time.time()

    system_prompt = (
        "You are an expert suicide risk assessment assistant. Analyze queries using categories 0-8.\n"
        "STRICT SAFETY RULE: If the risk_score is 6, 7, or 8, your 'explanation' MUST: \n"
        "1. Start with the phrase 'EMERGENCY: PLEASE SEEK IMMEDIATE HELP'.\n"
        "2. Instruct the user to call a crisis hotline or go to the nearest emergency room.\n"
        "3. Keep the tone calm but extremely urgent."
        "ADHERENCE RULE: Only assign categories 4-8 if the user explicitly mentions self-harm or suicide. "
        "If the user expresses hopelessness without a specific mention of dying or harm, use Category 2 (Ambiguous risk)."
    )

    try:
        # --- OPENAI ---
        if provider == "openai":
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content
            p_tokens, c_tokens = response.usage.prompt_tokens, response.usage.completion_tokens

        # --- GEMINI (MODERN 2026 SDK) ---
        elif provider == "gemini":
            # Initialize the modern Client
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

            # Use generate_content with the new structured config
            response = client.models.generate_content(
                model=model,
                contents=query,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )

            raw_content = response.text
            # Access usage metadata using the new attribute names
            p_tokens = response.usage_metadata.prompt_token_count
            c_tokens = response.usage_metadata.candidates_token_count

        # --- ANTHROPIC ---
        elif provider == "anthropic":
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": query}]
            )
            raw_content = response.content[0].text
            p_tokens, c_tokens = response.usage.input_tokens, response.usage.output_tokens

        parsed_json = json.loads(raw_content)
        return {
            "explanation": parsed_json.get("explanation", ""),
            "risk_score": parsed_json.get("risk_score", "Unknown"),
            "latency": time.time() - start_time,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
        }

    except Exception as e:
        print(f"!!! INFERENCE CRASH on query '{query[:20]}': {str(e)}")
        latency = time.time() - start_time
        return {
            "text": f"Error: {str(e)}",
            "risk_score": "Error",
            "latency": latency,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }