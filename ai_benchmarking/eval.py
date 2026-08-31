# scripts/run_benchmarks_once.py
import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd  # type: ignore
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio  # type: ignore

from ai_benchmarking.inference import (
    create_google_genai_client,
    generate_ai_response_async,
)
from ai_benchmarking.utils import (
    BASE_SYSTEM_PROMPT,
    calculate_cost,
    compute_metrics,
    get_severity_metrics,
    save_metrics,
)

load_dotenv()

MAX_CONCURRENT_REQUESTS = 250

CODE_TO_LABEL = {
    0: "No ideation",
    1: "Emotional distress",
    2: "Ambiguous risk",
    3: "Passive wish to be dead",
    4: "Active ideation (No method)",
    5: "Active ideation (Method, no intent)",
    6: "Active ideation (Method and intent)",
    7: "Active ideation (Plan and intent)",
    8: "Attempt or imminent risk",
}


async def process_single_item(
    item: dict[str, Any],
    provider: str,
    model: str,
    semaphore: asyncio.Semaphore,
    cache_name: str | None = None,
    fallback_prompt: str = "",
    client: Any | None = None,
) -> dict[str, Any]:
    """Processes one row concurrently inside the shared worker pool."""
    async with semaphore:
        await asyncio.sleep(random.uniform(0.0, 0.2))

        try:
            inf = await generate_ai_response_async(
                query=str(item.get("user_query", "")),
                provider=provider,
                model=model,
                cache_name=cache_name,
                fallback_prompt=fallback_prompt,
                client=client,
            )

            if not inf or "error" in inf:
                error_msg = (
                    inf.get("error", "Empty response from inference model")
                    if inf
                    else "Null response"
                )
                print(
                    f"!!! Row {item.get('id', 'Unknown')} failed inference: {error_msg}"
                )
                return {
                    "id": item.get("id"),
                    "query": item.get("user_query"),
                    "error": error_msg,
                }

            input_score = item.get("expected_risk_level", 0)
            ai_score = inf.get("risk_level", 0)

            input_sev_num, input_resp, input_cat = get_severity_metrics(input_score)
            ai_sev_num, ai_resp, ai_cat = get_severity_metrics(ai_score)

            is_exact_match = int(input_score) == int(ai_score)
            is_actionable_match = input_resp == ai_resp

            cost = calculate_cost(
                prompt_tokens=inf.get("prompt_tokens", 0),
                completion_tokens=inf.get("completion_tokens", 0),
                cached_tokens=inf.get("cached_tokens", 0),
                provider=provider,
                model=model,
            )

            latency = float(inf.get("latency", 0.0))

            return {
                "id": item.get("id"),
                "query": item.get("user_query"),
                "metrics": {
                    "predicted_risk_level": input_score,
                    "actual_risk_level": ai_score,
                    "predicted_severity_level": input_sev_num,
                    "actual_severity_level": ai_sev_num,
                    "predicted_severity_response": input_resp,
                    "actual_severity_response": ai_resp,
                    "is_exact_match": is_exact_match,
                    "is_actionable_match": is_actionable_match,
                },
                "latency": round(latency, 2),
                "inference_cost_usd": cost,
                "cached_tokens": inf.get("cached_tokens", 0),
            }

        except Exception as e:
            print(f"!!! Error evaluating ID {item.get('id', 'Unknown')}: {str(e)}")
            return {
                "id": item.get("id"),
                "query": item.get("user_query"),
                "error": str(e),
            }


async def run_benchmark_async(
    data_path: str,
    kb_path: str | None,
    output_path: str | None,
    provider: str,
    model: str,
) -> dict[str, Any]:
    if data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_dataset = json.load(f)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
        raw_dataset = df.to_dict(orient="records")
    else:
        raise ValueError("Unsupported format. Please provide a .csv or .json dataset.")

    knowledge_base: list[dict[str, Any]] = []
    raw_kb = ""

    target_kb_path = kb_path if kb_path else "data/knowledge_base.json"
    if os.path.exists(target_kb_path):
        with open(target_kb_path, "r", encoding="utf-8") as f:
            raw_kb = f.read().strip()
        if raw_kb:
            knowledge_base = json.loads(raw_kb)
    elif kb_path:
        raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")

    print(
        f"Starting async benchmark execution loop for {len(raw_dataset)} dataset entries..."
    )
    start_bench_time = time.time()

    base_system_prompt = BASE_SYSTEM_PROMPT

    if knowledge_base:
        base_system_prompt += "\n\n### FEW-SHOT SEED SAMPLES ###\n"
        for sample in knowledge_base:
            base_system_prompt += f'Query: "{sample.get("user_query", "")}" -> Expected Risk Level: {sample.get("Risk_level", 0)}\n'

    prompt_hash = hashlib.sha256(base_system_prompt.encode("utf-8")).hexdigest()
    kb_hash = hashlib.sha256(raw_kb.encode("utf-8")).hexdigest() if raw_kb else None

    cache_name: str | None = None
    client: Any | None = None

    if provider == "gemini":
        from google.genai import types
        from google.genai.errors import ClientError

        client = create_google_genai_client()

        try:
            print(
                "Initializing long-term Context Cache on Google servers (TTL: 24 Hours)..."
            )
            cached_content = client.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(
                    contents=[base_system_prompt],
                    ttl="86400s",
                ),
            )
            cache_name = str(cached_content.name)
            print(
                f"Context cached successfully! Handle reference identifier: {cache_name}"
            )
        except ClientError as e:
            if (
                getattr(e, "code", None) != 400
                or "minimum token count" not in str(e).lower()
            ):
                raise
            cache_name = None
            print(
                "Skipping Gemini explicit cache: prompt is below the 4096-token minimum. Using system_instruction instead."
            )

    elif provider == "openai":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    elif provider == "anthropic":
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    dataset: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_dataset):
        item_id: str | None = None
        if "user_id" in item and "prompt_id" in item:
            item_id = f"{item['user_id']}_{item['prompt_id']}"
        else:
            for k in ["id", "user_id", "prompt_id", "uid", "1"]:
                if k in item:
                    item_id = str(item[k])
                    break
            if item_id is None:
                item_id = f"index_{idx}"

        text_val: str = ""
        for k in ["history", "user_query", "text", "query", "0"]:
            if k in item and pd.notna(item[k]):
                text_val = str(item[k])
                break

        expected_score: int | float = 0
        for k in [
            "expected_risk_level",
            "predicted_risk_score",
            "Risk_level",
            "risk_level",
            "predicted_risk_level",
        ]:
            if k in item and pd.notna(item[k]):
                expected_score = item[k]
                break

        dataset.append(
            {
                "id": item_id,
                "user_query": text_val,
                "expected_risk_level": expected_score,
            }
        )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    tasks = [
        process_single_item(
            item=item,
            provider=provider,
            model=model,
            semaphore=semaphore,
            cache_name=cache_name,
            fallback_prompt=base_system_prompt,
            client=client,
        )
        for item in dataset
    ]

    results: list[dict[str, Any]] = []
    total_cached_tokens_accumulated = 0
    cache_hit_count = 0
    completed_count = 0
    total_tasks = len(tasks)

    pbar = tqdm_asyncio(total=total_tasks, desc="Evaluating Queries [Cache Lookups]")

    for next_task in asyncio.as_completed(tasks):
        row_result = await next_task
        results.append(row_result)
        completed_count += 1

        if "error" not in row_result:
            cached_tokens_returned = row_result.get("cached_tokens", 0)
            total_cached_tokens_accumulated += cached_tokens_returned
            if cached_tokens_returned > 0:
                cache_hit_count += 1

        completion_pct = (completed_count / total_tasks) * 100
        cache_hit_rate = (
            (cache_hit_count / completed_count) * 100 if completed_count > 0 else 0.0
        )

        pbar.set_postfix(
            {
                "done_pct": f"{completion_pct:.1f}%",
                "cache_hit_rate": f"{cache_hit_rate:.1f}%",
                "total_cached": f"{total_cached_tokens_accumulated:,}",
                "last_cached": (
                    "Yes" if row_result.get("cached_tokens", 0) > 0 else "No"
                ),
            }
        )
        pbar.update(1)

    pbar.close()

    clean_results = [r for r in results if "error" not in r]
    total_duration = time.time() - start_bench_time
    print(
        f"Completed {len(clean_results)} loop tasks in {round(total_duration, 2)} seconds."
    )

    benchmark_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "data_path": data_path,
        "kb_path": target_kb_path,
        "prompt_hash": prompt_hash,
        "kb_hash": kb_hash,
    }

    metrics = compute_metrics(clean_results, provider=provider, model=model)

    final_output = {
        "benchmark_metadata": benchmark_metadata,
        **metrics,
    }

    if output_path:
        save_metrics(final_output, output_path)
        print(f"Metrics mapped successfully. Results saved out to: {output_path}")

    return final_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["gemini", "openai", "anthropic"],
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument(
        "--kb",
        type=str,
        default=None,
        help="Optional few-shot knowledge base JSON. Omit to evaluate the --data labels without the examples as context.",
    )
    parser.add_argument("--output", type=str, required=False)

    args = parser.parse_args()

    asyncio.run(
        run_benchmark_async(
            data_path=args.data,
            kb_path=args.kb,
            output_path=args.output,
            provider=args.provider,
            model=args.model,
        )
    )
