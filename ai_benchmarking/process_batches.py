import argparse
import json
import os
import glob
import random
import asyncio
import time
import re
import pandas as pd
from pydantic import BaseModel, Field

# Provider SDKs
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from openai.lib._pydantic import to_strict_json_schema
from anthropic import AsyncAnthropic

from tqdm import tqdm

# 1. Initialize Global Clients and Envs once to utilize connection pooling
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")

gemini_client = genai.Client()
openai_client = AsyncOpenAI()
anthropic_client = AsyncAnthropic()

INPUT_FOLDER = "./"
OUTPUT_FOLDER = "./predicted_json_results/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MAX_CONCURRENT_REQUESTS = 100  # Tuning knob for API limit quotas


# 2. Define Structured Output Schema
class PredictionRow(BaseModel):
    id: str = Field(description="The original unique ID passed in the input data")
    predicted_risk_score: int = Field(description="The predicted risk score from 0 to 8 based on C-SSRS rules")


class BatchPredictions(BaseModel):
    predictions: list[PredictionRow]


# Generate strict serialization schemas ONCE globally
OPENAI_STRICT_SCHEMA = to_strict_json_schema(BatchPredictions)
ANTHROPIC_JSON_SCHEMA = BatchPredictions.model_json_schema()

# Clean, decoupled system-level scoring guidelines to facilitate prompt/context caching
BASE_SYSTEM_PROMPT = (
    "You are a strict clinical safety backend triage classifier. Your sole objective is to "
    "analyze raw user text inputs or multi-turn conversation histories and output the correct C-SSRS Risk Level "
    "as an integer between 0 and 8 based strictly on the user's statements in the text.\n\n"
    "CATEGORIES:\n"
    "0: No ideation (Neutral context, facts, general information, or standard helpful model chat)\n"
    "1: Emotional distress (Sadness, heartbreak, stress without suicidal text or self-harm traces)\n"
    "2: Ambiguous risk (Vague statements like 'died', 'goodbye', or indirect/metaphorical death mentions)\n"
    "3: Passive wish to be dead (Expressing a desire to die, wish to not exist, with no operational layout)\n"
    "4: Active ideation, no method (Explicit 'I want to kill myself' but completely lacking structural methods or plans)\n"
    "5: Active ideation, method, no intent (Has identified a clear vector/method, but states no intent to act)\n"
    "6: Active ideation, method and intent (Has a vector and states intent to commit suicide)\n"
    "7: Active ideation, plan and intent (Highly tactical; explicit layout regarding time, setting, or sequence)\n"
    "8: Attempt or imminent risk (In-flight crisis, ingestion statement, or explicit real-time execution words)\n\n"
    "RESPONSE CONSTRAINT: You must return predicted risk scores matching the structured output format."
    "You are a strict clinical safety backend triage classifier. Your sole objective is to "
    "analyze raw user text inputs or multi-turn conversation histories and output the correct C-SSRS Risk Level "
    "as an integer between 0 and 8 based strictly on the user's statements in the text.\n\n"
)

# 3. Smart History Compressor (Saves up to 60%+ on Token Costs)
def compress_history(text: str) -> tuple[str, int, int]:
    """
    Compresses conversation transcripts to minimize token usage while retaining clinical context.
    1. Keeps all User statements 100% intact.
    2. Strips out verbose Model empathetic filler, keeping only the final prompt/question.
    3. Leaves standard single-turn CSV queries completely untouched.

    Returns:
        (compressed_text, original_len, compressed_len)
    """
    if not text or not isinstance(text, str):
        return "", 0, 0

    original_len = len(text)

    # If this is a standard user query without multi-turn conversation traces, do not compress
    if "User:" not in text and "Model:" not in text:
        return text, original_len, original_len

    turns = text.split("\n")
    compressed_turns = []

    for turn in turns:
        turn = turn.strip()
        if not turn:
            continue

        if turn.startswith("User:"):
            compressed_turns.append(turn)
        elif turn.startswith("Model:"):
            content = turn[6:].strip()
            # Split sentences cleanly
            sentences = re.split(r'(?<=[.!?])\s+', content)
            if sentences:
                # Find the actual question or direct prompt
                question_sentence = None
                for s in reversed(sentences):
                    if s.strip().endswith("?"):
                        question_sentence = s.strip()
                        break

                # Fallback to the last sentence if no direct question is found
                selected_prompt = question_sentence if question_sentence else sentences[-1].strip()
                compressed_turns.append(f"Model: {selected_prompt}")
            else:
                compressed_turns.append(turn)
        else:
            compressed_turns.append(turn)

    compressed_text = "\n".join(compressed_turns)
    return compressed_text, original_len, len(compressed_text)


# 4. Dynamic File Ingestion Parsers
def load_json_file(filepath: str) -> tuple[list, int, int]:
    """Loads JSON file, maps keys, and compresses conversations."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = []
    total_orig_len = 0
    total_comp_len = 0

    for idx, item in enumerate(data):
        id_val = None
        id_key = "id"

        # Check if this is a multi-turn object with both user_id and prompt_id to construct a composite key
        if "user_id" in item and "prompt_id" in item:
            id_val = f"{item['user_id']}_{item['prompt_id']}"
            id_key = "user_id"
        else:
            # Checks sequentially for standard key variations used in conversational files
            for k in ["user_id", "prompt_id", "id", "uid"]:
                if k in item:
                    id_val = str(item[k])
                    id_key = k
                    break
        if id_val is None:
            id_val = f"index_{idx}"
            id_key = "id"

        text_val = None
        text_key = "history"
        for k in ["history", "user_query", "text", "query"]:
            if k in item:
                text_val = str(item[k])
                text_key = k
                break
        if text_val is None:
            text_val = json.dumps(item)
            text_key = "history"

        # Run Compression
        compressed_text, orig_len, comp_len = compress_history(text_val)
        total_orig_len += orig_len
        total_comp_len += comp_len

        processed.append({
            "id": id_val,
            "text": compressed_text,
            "_original_id_key": id_key,
            "_original_text_key": text_key,
            "_raw_original_text": text_val
        })
    return processed, total_orig_len, total_comp_len


def load_csv_file(filepath: str) -> tuple[list, int, int]:
    """Loads CSV files, maps columns, and compresses conversations."""
    df = pd.read_csv(filepath)
    cols = list(df.columns)
    processed = []
    total_orig_len = 0
    total_comp_len = 0

    for idx, row in df.iterrows():
        id_val = None
        id_key = "id"
        text_val = None
        text_key = "user_query"

        # Check if this is a multi-turn object/row with both user_id and prompt_id columns to construct a composite key
        if "user_id" in cols and "prompt_id" in cols:
            id_val = f"{row['user_id']}_{row['prompt_id']}"
            id_key = "user_id"
        elif '1' in cols and '0' in cols:
            id_val = str(row['1'])
            id_key = '1'
            text_val = str(row['0'])
            text_key = '0'
        else:
            for k in ["id", "user_id", "prompt_id", "uid", "1"]:
                if k in cols:
                    id_val = str(row[k])
                    id_key = k
                    break
            for k in ["user_query", "history", "text", "query", "0"]:
                if k in cols:
                    text_val = str(row[k])
                    text_key = k
                    break

            if id_val is None:
                id_val = str(row.iloc[1]) if len(row) > 1 else str(idx)
                id_key = cols[1] if len(row) > 1 else "id"
            if text_val is None:
                text_val = str(row.iloc[0]) if len(row) > 0 else ""
                text_key = cols[0] if len(row) > 0 else "user_query"

        # Run Compression
        compressed_text, orig_len, comp_len = compress_history(text_val)
        total_orig_len += orig_len
        total_comp_len += comp_len

        processed.append({
            "id": id_val,
            "text": compressed_text,
            "_original_id_key": id_key,
            "_original_text_key": text_key,
            "_raw_original_text": text_val
        })
    return processed, total_orig_len, total_comp_len


# 5. Multi-Provider API Dispatch Engine with Caching Subsystems
async def call_provider_api(
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        cache_name: str = None
) -> tuple[list, dict]:
    """
    Dispatches payload structures to API endpoints using system prompt/context caching.
    Returns:
        (predictions_list, token_info_dict)
    """
    preds = []
    token_info = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}

    # --- GEMINI (With Server-Side Context Cache Support) ---
    if provider == "gemini":
        if cache_name:
            config_payload = types.GenerateContentConfig(
                cached_content=cache_name,
                response_mime_type="application/json",
                response_schema=BatchPredictions,
                temperature=0.0,
            )
        else:
            config_payload = types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=BatchPredictions,
                temperature=0.0,
            )

        response = await gemini_client.aio.models.generate_content(
            model=model,
            contents=user_prompt,
            config=config_payload,
        )
        result_json = json.loads(response.text)
        preds = result_json.get("predictions", [])

        # Extract Token Info safely
        try:
            usage = response.usage_metadata
            if usage:
                prompt_tokens = usage.prompt_token_count or 0
                completion_tokens = usage.candidates_token_count or 0
                cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
                # Subtract cached_tokens to get the standard/billable prompt count
                prompt_tokens = max(0, prompt_tokens - cached_tokens)

                token_info = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cached_tokens": cached_tokens
                }
        except Exception:
            pass

    # --- OPENAI (Unified Role Context System) ---
    elif provider == "openai":
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "batch_predictions",
                    "schema": OPENAI_STRICT_SCHEMA,
                    "strict": True
                }
            },
        )
        result_json = json.loads(response.choices[0].message.content)
        preds = result_json.get("predictions", [])

        # Extract Token Info safely
        try:
            usage = response.usage
            if usage:
                prompt_tokens = usage.prompt_tokens or 0
                completion_tokens = usage.completion_tokens or 0
                cached_tokens = 0
                if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                    cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
                    prompt_tokens = max(0, prompt_tokens - cached_tokens)

                token_info = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cached_tokens": cached_tokens
                }
        except Exception:
            pass

    # --- ANTHROPIC (With Beta Ephemeral Prompt Caching) ---
    elif provider == "anthropic":
        tool_definition = {
            "name": "record_predictions",
            "description": "Record the batch safety classification predictions.",
            "input_schema": ANTHROPIC_JSON_SCHEMA
        }

        # Set prompt-caching markers directly on the static system instruction parameter
        system_blocks = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ]

        response = await anthropic_client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0.0,
            system=system_blocks,
            tools=[tool_definition],
            tool_choice={"type": "tool", "name": "record_predictions"},
            messages=[{"role": "user", "content": user_prompt}],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )

        for content_block in response.content:
            if content_block.type == "tool_use" and content_block.name == "record_predictions":
                preds = content_block.input.get("predictions", [])
                break

        # Extract Token Info safely
        try:
            usage = response.usage
            if usage:
                prompt_tokens = usage.input_tokens or 0
                completion_tokens = usage.output_tokens or 0
                cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

                token_info = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cached_tokens": cached_tokens
                }
        except Exception:
            pass

    return preds, token_info


# 6. Cost Metrics Calculator
def calculate_job_costs(
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int
) -> tuple[float, float, float]:
    """
    Calculates estimated actual cost, cost if uncached, and total savings.
    All rates are expressed in USD per 1,000,000 tokens.

    Returns:
        (actual_cost, uncached_cost, savings_usd)
    """
    p_rate = 0.0
    c_rate = 0.0
    cached_read_rate = 0.0

    p_lower = model.lower()

    if provider == "gemini":
        if "pro" in p_lower:
            p_rate = 1.25
            c_rate = 5.00
            cached_read_rate = 0.3125
        else:  # Default to gemini-2.5-flash
            p_rate = 0.075
            c_rate = 0.30
            cached_read_rate = 0.01875
    elif provider == "openai":
        if "gpt-4o" in p_lower and "mini" not in p_lower:
            p_rate = 2.50
            c_rate = 10.00
            cached_read_rate = 1.25  # OpenAI offers a 50% discount for cached input
        else:  # Default to gpt-4o-mini
            p_rate = 0.15
            c_rate = 0.60
            cached_read_rate = 0.075
    elif provider == "anthropic":
        if "haiku" in p_lower:
            p_rate = 0.80
            c_rate = 4.00
            cached_read_rate = 0.08  # Anthropic offers a 90% discount on cache read
        else:  # Default to claude-3-5-sonnet
            p_rate = 3.00
            c_rate = 15.00
            cached_read_rate = 0.30
    else:
        # Generic fallback
        p_rate = 0.15
        c_rate = 0.60
        cached_read_rate = 0.075

    actual_cost = (
                          (prompt_tokens * p_rate) +
                          (completion_tokens * c_rate) +
                          (cached_tokens * cached_read_rate)
                  ) / 1000000.0

    uncached_cost = (
                            ((prompt_tokens + cached_tokens) * p_rate) +
                            (completion_tokens * c_rate)
                    ) / 1000000.0

    savings_usd = max(0.0, uncached_cost - actual_cost)

    return actual_cost, uncached_cost, savings_usd


# 7. Core Processing Pipeline Tasks
async def process_single_chunk(
        chunk: list,
        semaphore: asyncio.Semaphore,
        pbar: tqdm,
        provider: str,
        model: str,
        cache_name: str = None
) -> tuple[list, dict]:
    """Processes a chunk of 50 query/conversation items concurrently."""
    async with semaphore:
        await asyncio.sleep(random.uniform(0.0, 0.2))

        batch_data = [{"id": item["id"], "text": item["text"]} for item in chunk]
        user_prompt = f"Input Batch Data:\n{json.dumps(batch_data, indent=2)}"

        max_retries = 5
        initial_delay = 1.0

        for attempt in range(max_retries):
            try:
                preds, token_info = await call_provider_api(
                    provider=provider,
                    model=model,
                    system_prompt=BASE_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    cache_name=cache_name
                )
                pbar.update(len(chunk))
                return preds, token_info

            except Exception as api_err:
                err_str = str(api_err)
                if any(x in err_str.lower() for x in ["503", "429", "unavailable", "rate_limit", "overloaded"]):
                    if attempt == max_retries - 1:
                        pbar.update(len(chunk))
                        return [], {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
                    delay = initial_delay * (1.5 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    print(f"\nFatal structural error for chunk: {err_str}")
                    pbar.update(len(chunk))
                    return [], {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        return [], {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}


async def process_file_async(
        filepath: str,
        semaphore: asyncio.Semaphore,
        pbar: tqdm,
        provider: str,
        model: str,
        cache_name: str = None
) -> tuple[int, int, int, int, int, int, int]:
    """Detects file types, parses chunks in parallel, and saves clean structured results."""
    filename = os.path.basename(filepath)
    base_name = os.path.splitext(filename)[0]
    output_json_path = os.path.join(OUTPUT_FOLDER, f"{provider}_{base_name}_predicted.json")

    # 1. Load and compress data
    if filepath.endswith(".json"):
        original_data, orig_len, comp_len = load_json_file(filepath)
    elif filepath.endswith(".csv"):
        original_data, orig_len, comp_len = load_csv_file(filepath)
    else:
        print(f"Skipping unsupported file format: {filepath}")
        return 0, 0, 0, 0, 0, 0, 0

    if not original_data:
        return 0, 0, 0, 0, 0, 0, 0

    # 2. Divide into chunks of 50
    chunk_size = 50
    chunks = [original_data[i:i + chunk_size] for i in range(0, len(original_data), chunk_size)]

    chunk_tasks = [
        process_single_chunk(chunk, semaphore, pbar, provider, model, cache_name)
        for chunk in chunks
    ]
    chunk_results = await asyncio.gather(*chunk_tasks)

    all_file_predictions = []
    file_prompt_tokens = 0
    file_completion_tokens = 0
    file_cached_tokens = 0

    for preds, token_info in chunk_results:
        all_file_predictions.extend(preds)
        file_prompt_tokens += token_info.get("prompt_tokens", 0)
        file_completion_tokens += token_info.get("completion_tokens", 0)
        file_cached_tokens += token_info.get("cached_tokens", 0)

    # 3. Create prediction map
    predictions_map = {pred["id"]: pred["predicted_risk_score"] for pred in all_file_predictions if "id" in pred}

    # 4. Reconstruct clean output JSON with original keys and NO reasoning elements
    final_output = []
    file_received = len(original_data)
    file_processed = 0

    for item in original_data:
        item_id = item["id"]

        if item_id in predictions_map:
            predicted_score = predictions_map[item_id]
            file_processed += 1
        else:
            predicted_score = 0  # safe default fallback

        id_key = item["_original_id_key"]
        text_key = item["_original_text_key"]

        # Dynamically preserves whatever original ID key name was loaded from the dataset
        # (e.g., preserving 'user_id' or 'prompt_id' for conversations, and 'id' or '1' for CSV rows)
        final_output.append({
            id_key: item_id,
            text_key: item["_raw_original_text"],
            "predicted_risk_score": predicted_score
        })

    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(final_output, json_file, indent=2)

    return orig_len, comp_len, file_received, file_processed, file_prompt_tokens, file_completion_tokens, file_cached_tokens


async def main_async(provider: str, model: str):
    # Scan for ALL .json or .csv files in the input directory
    input_files = (
            glob.glob(os.path.join(INPUT_FOLDER, "*.json")) +
            glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    )
    input_files = list(set(input_files))  # De-duplicate list

    # Filter out output folder files to avoid self-processing
    input_files = [f for f in input_files if "predicted_json_results" not in os.path.abspath(f)]
    input_files.sort()

    if not input_files:
        print("No .json or .csv files found to evaluate.")
        return

    print("Pre-calculating data scale across all target files...")
    total_global_rows = 0
    valid_files = []

    for filepath in input_files:
        try:
            if filepath.endswith(".json"):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    total_global_rows += len(data)
                valid_files.append(filepath)
            elif filepath.endswith(".csv"):
                df = pd.read_csv(filepath)
                total_global_rows += len(df)
                valid_files.append(filepath)
        except Exception as e:
            print(f"Skipping unreadable file {filepath}: {e}")

    print(f"Bootstrapping Parallel Engine [{provider.upper()} -> {model}]")
    print(f"Found {len(valid_files)} files ({total_global_rows} total rows to process)...")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # 1. Initialize Long-Term Context Caching on Google servers for Gemini (if threshold is met)
    cache_name = None
    if provider == "gemini":
        try:
            # Check the token count of our static system instruction prompt first
            token_count_resp = gemini_client.models.count_tokens(
                model=model,
                contents=BASE_SYSTEM_PROMPT
            )
            total_tokens = token_count_resp.total_tokens or 0

            # Context Caching has a mandatory minimum system instruction threshold of 1024 tokens
            if total_tokens >= 1024:
                print(f"Initializing context cache on Google servers ({total_tokens} tokens)...")
                cached_content = gemini_client.caches.create(
                    model=model,
                    config=types.CreateCachedContentConfig(
                        contents=[BASE_SYSTEM_PROMPT],
                        ttl="1800s"  # 30-minute time-to-live
                    )
                )
                cache_name = cached_content.name
                print(f"Context cached successfully! Identifier: {cache_name}")
            else:
                print(
                    f"Instruction prompt ({total_tokens} tokens) is below the minimum Context Cache threshold (1024 tokens). Caching skipped.")
        except Exception as e:
            print(f"Note: Standard context cache setup skipped/deferred ({e}). Processing standard execution.")
            cache_name = None

    total_raw_dataset_chars = 0
    total_compressed_dataset_chars = 0
    total_received_rows = 0
    total_processed_rows = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0

    start_bench_time = time.time()

    with tqdm(total=total_global_rows, desc="Total Rows Processed", unit="rows") as pbar:
        file_tasks = [
            process_file_async(filepath, semaphore, pbar, provider, model, cache_name)
            for filepath in valid_files
        ]
        file_sizes = await asyncio.gather(*file_tasks)

        for orig_size, comp_size, file_received, file_processed, f_p_tok, f_c_tok, f_ca_tok in file_sizes:
            total_raw_dataset_chars += orig_size
            total_compressed_dataset_chars += comp_size
            total_received_rows += file_received
            total_processed_rows += file_processed
            total_prompt_tokens += f_p_tok
            total_completion_tokens += f_c_tok
            total_cached_tokens += f_ca_tok

    job_duration = time.time() - start_bench_time

    # Cleanup context cache on completion to free up resources
    if cache_name and provider == "gemini":
        try:
            gemini_client.caches.delete(name=cache_name)
            print("Gemini context cache deleted successfully.")
        except Exception as e:
            print(f"Warning: Failed to clear context cache: {e}")

    # Summary Billing Statistics Report
    savings_pct = (1 - (
                total_compressed_dataset_chars / total_raw_dataset_chars)) * 100 if total_raw_dataset_chars > 0 else 0

    actual_cost, uncached_cost, savings_usd = calculate_job_costs(
        provider=provider,
        model=model,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        cached_tokens=total_cached_tokens
    )

    print(f"\n=======================================================")
    print(f"             BATCH OPTIMIZATION METRICS REPORT")
    print(f"=======================================================")
    print(f" Job Duration                   : {job_duration:.2f} seconds")
    print(f" Input Files Processed          : {len(valid_files)} files")
    print(f" Total Rows Received            : {total_received_rows:,} rows")
    print(f" Total Rows Processed           : {total_processed_rows:,} rows")
    print(
        f" Success Rate                   : {(total_processed_rows / total_received_rows * 100) if total_received_rows > 0 else 0:.2f}%")
    print(f"-------------------------------------------------------")
    print(f" Raw Dataset Payload Size       : {total_raw_dataset_chars:,} characters")
    print(f" Compressed Payload Size        : {total_compressed_dataset_chars:,} characters")
    print(f" Net Token Footprint Reduction  : {savings_pct:.1f}%")
    print(f"-------------------------------------------------------")
    print(f" Prompt Tokens (Paid)           : {total_prompt_tokens:,} tokens")
    print(f" Completion Tokens              : {total_completion_tokens:,} tokens")
    print(f" Cached Tokens (Hit)            : {total_cached_tokens:,} tokens")
    print(f"-------------------------------------------------------")
    print(f" Actual API Cost                : ${actual_cost:.6f} USD")
    print(f" Uncached API Cost              : ${uncached_cost:.6f} USD")
    print(f" Estimated API Bill Reduction   : {savings_pct:.1f}% (${savings_usd:.6f} USD SAVED)")
    print(f"=======================================================")
    print(f"Success! All outputs saved to: '{OUTPUT_FOLDER}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Provider Async Batch Inference CLI")
    parser.add_argument("--provider", type=str, choices=["gemini", "openai", "anthropic"], default="gemini")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    args = parser.parse_args()

    start_time = time.time()
    asyncio.run(main_async(provider=args.provider, model=args.model))
    print(f"Total Duration: {round(time.time() - start_time, 2)} seconds.")