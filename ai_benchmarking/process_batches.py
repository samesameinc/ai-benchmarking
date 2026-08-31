import argparse
import asyncio
import glob
import json
import os
import random
import re
import time
from typing import Any, Iterable, cast

import pandas as pd  # type: ignore
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceToolParam,
    ToolParam,
    ToolUseBlock,
)
from dotenv import load_dotenv
from google.genai import types
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, Field
from tqdm import tqdm  # type: ignore

from ai_benchmarking.inference import (
    create_anthropic_client,
    create_google_genai_client,
    create_openai_client,
)
from ai_benchmarking.utils import BASE_SYSTEM_PROMPT, calculate_job_costs

# Load environment variables prior to any client creation
load_dotenv()

MAX_CONCURRENT_REQUESTS = 100

# Global singletons for lazy initialization
_gemini_client = None
_openai_client = None
_anthropic_client = None


def get_gemini_client() -> Any:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = create_google_genai_client()
    return _gemini_client


def get_openai_client() -> Any:
    global _openai_client
    if _openai_client is None:
        _openai_client = create_openai_client()
    return _openai_client


def get_anthropic_client() -> Any:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = create_anthropic_client()
    return _anthropic_client


# Define Structured Output Schema
class PredictionRow(BaseModel):
    id: str = Field(description="The original unique ID passed in the input data")
    predicted_risk_score: int = Field(
        description="The predicted risk score from 0 to 8 based on C-SSRS rules"
    )


class BatchPredictions(BaseModel):
    predictions: list[PredictionRow]


OPENAI_STRICT_SCHEMA = to_strict_json_schema(BatchPredictions)


def compress_history(text: str) -> tuple[str, int, int]:
    if not text or not isinstance(text, str):
        return "", 0, 0

    original_len = len(text)
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
            sentences = re.split(r"(?<=[.!?])\s+", content)
            if sentences:
                question_sentence = next(
                    (s.strip() for s in reversed(sentences) if s.strip().endswith("?")),
                    None,
                )
                selected_prompt = (
                    question_sentence if question_sentence else sentences[-1].strip()
                )
                compressed_turns.append(f"Model: {selected_prompt}")
            else:
                compressed_turns.append(turn)
        else:
            compressed_turns.append(turn)

    compressed_text = "\n".join(compressed_turns)
    return compressed_text, original_len, len(compressed_text)


def load_json_file(filepath: str) -> tuple[list, int, int]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed, total_orig_len, total_comp_len = [], 0, 0
    for idx, item in enumerate(data):
        id_val, id_key = None, "id"
        if "user_id" in item and "prompt_id" in item:
            id_val = f"{item['user_id']}_{item['prompt_id']}"
            id_key = "user_id"
        else:
            for k in ["user_id", "prompt_id", "id", "uid"]:
                if k in item and item[k] is not None:
                    id_val, id_key = str(item[k]), k
                    break
        if id_val is None:
            id_val, id_key = f"index_{idx}", "id"

        text_val: str = ""
        text_key = "history"
        for k in ["history", "user_query", "text", "query"]:
            if k in item and item[k] is not None:
                text_val = str(item[k])
                text_key = k
                break

        if not text_val:
            text_val, text_key = json.dumps(item), "history"

        compressed_text, orig_len, comp_len = compress_history(text_val)
        total_orig_len += orig_len
        total_comp_len += comp_len

        processed.append(
            {
                "id": id_val,
                "text": compressed_text,
                "_original_id_key": id_key,
                "_original_text_key": text_key,
                "_raw_original_text": text_val,
            }
        )
    return processed, total_orig_len, total_comp_len


def load_csv_file(filepath: str) -> tuple[list, int, int]:
    df = pd.read_csv(filepath)
    cols = list(df.columns)
    processed, total_orig_len, total_comp_len = [], 0, 0

    for idx, row in df.iterrows():
        id_val, id_key = None, "id"
        text_val: str = ""
        text_key = "user_query"

        if "user_id" in cols and "prompt_id" in cols:
            id_val, id_key = f"{row['user_id']}_{row['prompt_id']}", "user_id"
        elif "1" in cols and "0" in cols:
            id_val, id_key = str(row["1"]), "1"
            text_val, text_key = str(row["0"]) if pd.notna(row["0"]) else "", "0"
        else:
            for k in ["id", "user_id", "prompt_id", "uid", "1"]:
                if k in cols and pd.notna(row[k]):
                    id_val, id_key = str(row[k]), k
                    break
            for k in ["user_query", "history", "text", "query", "0"]:
                if k in cols and pd.notna(row[k]):
                    text_val, text_key = str(row[k]), k
                    break

            if id_val is None:
                id_val, id_key = (
                    (str(row.iloc[1]), cols[1]) if len(row) > 1 else (str(idx), "id")
                )
            if not text_val:
                text_val, text_key = (
                    (str(row.iloc[0]), cols[0])
                    if len(row) > 0 and pd.notna(row.iloc[0])
                    else ("", "user_query")
                )

        compressed_text, orig_len, comp_len = compress_history(text_val)
        total_orig_len += orig_len
        total_comp_len += comp_len

        processed.append(
            {
                "id": id_val,
                "text": compressed_text,
                "_original_id_key": id_key,
                "_original_text_key": text_key,
                "_raw_original_text": text_val,
            }
        )
    return processed, total_orig_len, total_comp_len


async def call_provider_api(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    cache_name: str | None = None,
) -> tuple[list, dict]:
    preds = []
    token_info = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}

    if provider == "gemini":
        client = get_gemini_client()
        config_payload = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchPredictions,
            temperature=0.0,
        )
        if cache_name:
            config_payload.cached_content = cache_name
        else:
            config_payload.system_instruction = system_prompt

        response = await client.aio.models.generate_content(
            model=model,
            contents=user_prompt,
            config=config_payload,
        )

        content = str(response.text) if response.text else "{}"
        result_json = json.loads(content)
        preds = result_json.get("predictions", [])

        try:
            usage = response.usage_metadata
            if usage:
                prompt_tokens = usage.prompt_token_count or 0
                completion_tokens = usage.candidates_token_count or 0
                cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
                token_info = {
                    "prompt_tokens": max(0, prompt_tokens - cached_tokens),
                    "completion_tokens": completion_tokens,
                    "cached_tokens": cached_tokens,
                }
        except Exception:
            pass

    elif provider == "openai":
        client = get_openai_client()
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "batch_predictions",
                    "schema": OPENAI_STRICT_SCHEMA,
                    "strict": True,
                },
            },
        )

        content = (
            str(response.choices[0].message.content)
            if response.choices[0].message.content
            else "{}"
        )
        result_json = json.loads(content)
        preds = result_json.get("predictions", [])

        try:
            usage = response.usage
            if usage:
                prompt_tokens = usage.prompt_tokens or 0
                cached_tokens = 0
                if (
                    hasattr(usage, "prompt_tokens_details")
                    and usage.prompt_tokens_details
                ):
                    cached_tokens = (
                        getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
                    )
                token_info = {
                    "prompt_tokens": max(0, prompt_tokens - cached_tokens),
                    "completion_tokens": usage.completion_tokens or 0,
                    "cached_tokens": cached_tokens,
                }
        except Exception:
            pass

    elif provider == "anthropic":
        client = get_anthropic_client()
        tool_definition = cast(
            ToolParam,
            {
                "name": "batch_predictions",
                "description": "Record the predicted batch of C-SSRS risk classifications.",
                "input_schema": BatchPredictions.model_json_schema(),
            },
        )

        system_input: str | list[TextBlockParam] = (
            [
                cast(
                    TextBlockParam,
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    },
                )
            ]
            if system_prompt
            else ""
        )

        tool_choice = cast(
            ToolChoiceToolParam, {"type": "tool", "name": "batch_predictions"}
        )
        messages = cast(
            Iterable[MessageParam], [{"role": "user", "content": user_prompt}]
        )

        response = await client.messages.create(
            model=model,
            max_tokens=8192,
            system=system_input,
            tools=[tool_definition],
            tool_choice=tool_choice,
            messages=messages,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )

        content = "{}"
        for content_block in response.content:
            if (
                isinstance(content_block, ToolUseBlock)
                and content_block.name == "batch_predictions"
            ):
                content = json.dumps(content_block.input)
                break

        result_json = json.loads(content)
        preds = result_json.get("predictions", [])

        try:
            usage = response.usage
            if usage:
                prompt_tokens = usage.input_tokens or 0
                completion_tokens = usage.output_tokens or 0
                cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
                token_info = {
                    "prompt_tokens": max(0, prompt_tokens - cached_tokens),
                    "completion_tokens": completion_tokens,
                    "cached_tokens": cached_tokens,
                }
        except Exception:
            pass

    return preds, token_info


async def process_single_chunk(
    chunk: list,
    semaphore: asyncio.Semaphore,
    pbar: Any,
    provider: str,
    model: str,
    system_prompt: str,
    cache_name: str | None = None,
) -> tuple[list, dict]:
    async with semaphore:
        await asyncio.sleep(random.uniform(0.0, 0.2))

        batch_data = [{"id": item["id"], "text": item["text"]} for item in chunk]
        user_prompt = f"Input Batch Data:\n{json.dumps(batch_data, indent=2)}"

        max_retries = 5
        for attempt in range(max_retries):
            try:
                preds, token_info = await call_provider_api(
                    provider, model, system_prompt, user_prompt, cache_name
                )
                pbar.update(len(chunk))
                return preds, token_info

            except Exception as api_err:
                if any(
                    x in str(api_err).lower()
                    for x in ["503", "429", "unavailable", "rate_limit"]
                ):
                    if attempt == max_retries - 1:
                        pbar.update(len(chunk))
                        return [], {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "cached_tokens": 0,
                        }
                    await asyncio.sleep(1.0 * (1.5**attempt))
                else:
                    print(f"\nFatal structural error for chunk: {api_err}")
                    pbar.update(len(chunk))
                    return [], {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cached_tokens": 0,
                    }
        return [], {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}


async def process_file_async(
    filepath: str,
    output_path: str,
    semaphore: asyncio.Semaphore,
    pbar: Any,
    provider: str,
    model: str,
    system_prompt: str,
    cache_name: str | None = None,
) -> tuple[int, int, int, int, int, int, int]:
    original_data, orig_len, comp_len = (
        (
            load_json_file(filepath)
            if filepath.endswith(".json")
            else load_csv_file(filepath)
        )
        if filepath.endswith((".json", ".csv"))
        else (None, 0, 0)
    )

    if not original_data:
        return 0, 0, 0, 0, 0, 0, 0

    chunk_size = 50
    chunks = [
        original_data[i : i + chunk_size]
        for i in range(0, len(original_data), chunk_size)
    ]

    chunk_results = await asyncio.gather(
        *[
            process_single_chunk(
                chunk, semaphore, pbar, provider, model, system_prompt, cache_name
            )
            for chunk in chunks
        ]
    )

    all_file_predictions = []
    f_p_tok, f_c_tok, f_ca_tok = 0, 0, 0

    for preds, token_info in chunk_results:
        all_file_predictions.extend(preds)
        f_p_tok += token_info.get("prompt_tokens", 0)
        f_c_tok += token_info.get("completion_tokens", 0)
        f_ca_tok += token_info.get("cached_tokens", 0)

    predictions_map = {
        pred["id"]: pred["predicted_risk_score"]
        for pred in all_file_predictions
        if "id" in pred
    }

    final_output = []
    file_processed = 0

    for item in original_data:
        item_id = item["id"]
        predicted_score = predictions_map.get(item_id, 0)
        if item_id in predictions_map:
            file_processed += 1

        final_output.append(
            {
                item["_original_id_key"]: item_id,
                item["_original_text_key"]: item["_raw_original_text"],
                "predicted_risk_score": predicted_score,
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(final_output, json_file, indent=2)

    return (
        orig_len,
        comp_len,
        len(original_data),
        file_processed,
        f_p_tok,
        f_c_tok,
        f_ca_tok,
    )


async def main_async(
    data_path: str,
    provider: str,
    model: str,
    output_path: str | None = None,
    system_prompt: str = BASE_SYSTEM_PROMPT,
):
    if os.path.isdir(data_path):
        input_files = list(
            set(
                glob.glob(os.path.join(data_path, "*.json"))
                + glob.glob(os.path.join(data_path, "*.csv"))
            )
        )
    elif os.path.isfile(data_path):
        input_files = [data_path]
    else:
        raise FileNotFoundError(f"Target data path not found: {data_path}")

    input_files = sorted(input_files)

    if not input_files:
        print("No .json or .csv files found to evaluate.")
        return

    print(f"Bootstrapping Parallel Engine [{provider.upper()} -> {model}]")
    total_global_rows = sum(
        len(
            json.load(open(f, "r", encoding="utf-8"))
            if f.endswith(".json")
            else pd.read_csv(f)
        )
        for f in input_files
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    cache_name: str | None = None

    if provider == "gemini":
        client = get_gemini_client()
        try:
            total_tokens = (
                client.models.count_tokens(
                    model=model, contents=system_prompt
                ).total_tokens
                or 0
            )
            if total_tokens >= 1024:
                cached_content = client.caches.create(
                    model=model,
                    config=types.CreateCachedContentConfig(
                        contents=[system_prompt], ttl="1800s"
                    ),
                )
                cache_name = str(cached_content.name)
                print(f"Context cached successfully! Identifier: {cache_name}")
        except Exception as e:
            print(
                f"Note: Standard context cache setup skipped ({e}). Processing standard execution."
            )

    start_bench_time = time.time()
    metrics = {
        "orig_size": 0,
        "comp_size": 0,
        "recv": 0,
        "proc": 0,
        "p_tok": 0,
        "c_tok": 0,
        "ca_tok": 0,
    }

    with tqdm(
        total=total_global_rows, desc="Total Rows Processed", unit="rows"
    ) as pbar:
        file_tasks = []
        for f in input_files:
            if output_path and os.path.isdir(output_path):
                out_file = os.path.join(
                    output_path, f"{provider}_{os.path.basename(f)}_predicted.json"
                )
            elif output_path:
                out_file = output_path
            else:
                out_file = f"./predicted_json_results/{provider}_{os.path.basename(f)}_predicted.json"

            file_tasks.append(
                process_file_async(
                    f,
                    out_file,
                    semaphore,
                    pbar,
                    provider,
                    model,
                    system_prompt,
                    cache_name,
                )
            )

        results = await asyncio.gather(*file_tasks)
        for r in results:
            metrics["orig_size"] += r[0]
            metrics["comp_size"] += r[1]
            metrics["recv"] += r[2]
            metrics["proc"] += r[3]
            metrics["p_tok"] += r[4]
            metrics["c_tok"] += r[5]
            metrics["ca_tok"] += r[6]

    if cache_name and provider == "gemini":
        try:
            get_gemini_client().caches.delete(name=cache_name)
        except Exception:
            pass

    actual_cost, uncached_cost, savings_usd = calculate_job_costs(
        provider, model, metrics["p_tok"], metrics["c_tok"], metrics["ca_tok"]
    )

    print("\n=======================================================")
    print("             BATCH OPTIMIZATION METRICS REPORT")
    print("=======================================================")
    print(
        f" Job Duration                   : {time.time() - start_bench_time:.2f} seconds"
    )
    print(f" Input Files Processed          : {len(input_files)} files")
    print(
        f" Total Rows Processed           : {metrics['proc']:,} / {metrics['recv']:,} rows"
    )
    print(f" Actual API Cost                : ${actual_cost:.6f} USD")
    print(f" Estimated API Bill Reduction   : {savings_usd:.6f} USD SAVED")
    print("=======================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate First-Line Judge baseline predictions using batch processing."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input .json/.csv file or directory.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["gemini", "openai", "anthropic"],
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to output result JSON file or directory.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional path to a .txt file containing a custom system prompt.",
    )

    args = parser.parse_args()

    system_prompt_to_use = BASE_SYSTEM_PROMPT
    if args.prompt and os.path.exists(args.prompt):
        with open(args.prompt, "r", encoding="utf-8") as f:
            system_prompt_to_use = f.read().strip()

    asyncio.run(
        main_async(
            data_path=args.data,
            provider=args.provider,
            model=args.model,
            output_path=args.output,
            system_prompt=system_prompt_to_use,
        )
    )
