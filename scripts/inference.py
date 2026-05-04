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
        "You are an expert clinical safety assistant using the C-SSRS framework. "
        "Classify the user query into EXACTLY one of these categories:\n"
        "1. No ideation: General talk, no death mentions.\n"
        "2. Passive ideation: 'Wish I wouldn't wake up' or 'Better off dead' without intent.\n"
        "3. Ambiguous risk: Vague hopelessness without mentions of death/harm.\n"
        "4. Non-suicidal self-injury (NSSI): Self-harm (cutting/burning) WITHOUT intent to die.\n"
        "5. Active ideation (Method/Intent): Mentions pills, jumping, or specific plans.\n"
        "6. Preparatory Behavior: Giving away items, 'goodbyes', or buying/sitting with a method.\n"
        "7. Active risk with command hallucinations: Voices telling them to harm themselves.\n"
        "8. Attempt or imminent risk: Currently in the act or about to start.\n\n"
        "RESPONSE FORMAT: Return JSON with keys 'risk_score' (1-8) and 'explanation'."
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
            "risk_score": int(parsed_json.get("risk_score", 0)),  # Force Integer
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