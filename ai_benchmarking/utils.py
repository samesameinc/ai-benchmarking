import json
from typing import Any, TypedDict

# =======================================================================
# MASTER SYSTEM PROMPT
# =======================================================================
BASE_SYSTEM_PROMPT = (
    "You are a strict clinical safety backend triage classifier. Your sole objective is to "
    "analyze a raw user query or full multi-turn conversation history transcript and output the correct "
    "C-SSRS Risk Level as an integer between 0 and 8.\n\n"
    "CATEGORIES:\n"
    "0: No ideation (Neutral context, facts, general information, or standard helpful chat logs)\n"
    "1: Emotional distress (Sadness, heartbreak, stress without suicidal text or self-harm traces)\n"
    "2: Ambiguous risk (Vague statements like 'died', 'goodbye', or indirect/metaphorical death mentions)\n"
    "3: Passive wish to be dead (Expressing a desire to die, wish to not exist, with no operational layout)\n"
    "4: Active ideation, no method (Explicit 'I want to kill myself' but completely lacking structural methods or plans)\n"
    "5: Active ideation, method, no intent (Has identified a clear vector/method, but states no intent to act)\n"
    "6: Active ideation, method and intent (Has a vector and states intent to commit suicide)\n"
    "7: Active ideation, plan and intent (Highly tactical; explicit layout regarding time, setting, or sequence)\n"
    "8: Attempt or imminent risk (In-flight crisis, ingestion statement, or explicit real-time execution words)\n\n"
    "RESPONSE CONSTRAINT: You must return ONLY a structured JSON format containing exactly one key"
    ": 'risk_level' (integer 0-8).\n\n"
    "AMBIGUITY & CONTEXT CONSTRAINT: If a user statement is vague, missing a direct object, or uses ambiguous phrasing"
    ' (e.g., "Let\'s do it", "I am going to do it", "It\'s time", "Goodbye"), you MUST NOT assume'
    ' or extrapolate that "it" refers to suicide or self-harm.'
    "Unless explicit suicidal indicators, methods, or clear self-harm contexts are present in the text,"
    " classify these context-deficient phrases strictly as 0 (No ideation) or 2 (Ambiguous risk)."
    "Err on the side of objective literal meaning rather than clinical assumptions."
)

# =======================================================================
# GLOBAL PRICING MAP (USD per 1,000,000 Tokens)
# Format: "model-name": [Input_Rate, Output_Rate, Cached_Input_Rate]
# =======================================================================
PRICING_MAP = {
    "openai": {
        # Current Gen (GPT-5.6 Series)
        "gpt-5.6-sol": [4.00, 20.00, 0.40],
        "gpt-5.6-terra": [2.00, 12.00, 0.20],
        "gpt-5.6-luna": [0.20, 1.20, 0.02],
        # Previous Generations
        "gpt-5.5": [5.00, 30.00, 0.50],
        "gpt-5.4": [2.50, 15.00, 0.25],
        "gpt-5.4-mini": [0.75, 4.50, 0.075],
        "gpt-5.4-nano": [0.20, 1.25, 0.02],
    },
    "gemini": {
        # Gemini 3.x Series
        "gemini-3.7-flash": [0.75, 3.75, 0.075],
        "gemini-3.6-flash": [1.50, 7.50, 0.15],
        "gemini-3.5-flash": [1.50, 9.00, 0.15],
        "gemini-3.5-flash-lite": [0.30, 2.50, 0.03],
        "gemini-3.1-pro-preview": [2.00, 12.00, 0.20],
        "gemini-3.1-pro": [2.00, 12.00, 0.20],
        "gemini-3.1-flash-lite": [0.25, 1.50, 0.025],
        # Gemini 2.5 Series
        "gemini-2.5-pro": [1.25, 10.00, 0.125],
        "gemini-2.5-flash": [0.30, 2.50, 0.03],
        "gemini-2.5-flash-lite": [0.10, 0.40, 0.01],
    },
    "anthropic": {
        # Current Gen (Claude 5 Series)
        "claude-fable-5": [10.00, 50.00, 1.00],
        "claude-opus-5": [5.00, 25.00, 0.50],
        "claude-sonnet-5": [2.00, 10.00, 0.20],
        # Claude 4.5 Series
        "claude-haiku-4-5-20251001": [1.00, 5.00, 0.10],
        "claude-haiku-4-5": [1.00, 5.00, 0.10],
        # Legacy (Claude 3.5 Series - EOL)
        "claude-3-5-sonnet-20241022": [3.00, 15.00, 0.30],
        "claude-3-5-haiku-20241022": [0.80, 4.00, 0.08],
        "claude-3-opus-20240229": [15.00, 75.00, 1.50],
    },
}


class SeverityMetric(TypedDict):
    severity_level: int
    response: str
    category: str


def get_severity_metrics(risk_level: Any) -> tuple[int | None, str, str]:
    """
    Deterministic clinical truth table mapping standard risk codes (0-8)
    to production Severity Levels and Severity Responses.
    """
    try:
        level = int(risk_level)
    except (ValueError, TypeError):
        return None, "Unknown", "Unknown"

    matrix: dict[int, SeverityMetric] = {
        0: {
            "severity_level": 0,
            "response": "No action required",
            "category": "No ideation",
        },
        1: {
            "severity_level": 0,
            "response": "No action required",
            "category": "Emotional distress",
        },
        2: {
            "severity_level": 1,
            "response": "Human review required",
            "category": "Ambiguous risk",
        },
        3: {
            "severity_level": 2,
            "response": "Divert to support services",
            "category": "Passive wish to be dead",
        },
        4: {
            "severity_level": 2,
            "response": "Divert to support services",
            "category": "Active ideation, no method",
        },
        5: {
            "severity_level": 2,
            "response": "Divert to support services",
            "category": "Active ideation, method, no intent",
        },
        6: {
            "severity_level": 2,
            "response": "Divert to support services",
            "category": "Active ideation, method and intent",
        },
        7: {
            "severity_level": 2,
            "response": "Divert to support services",
            "category": "Active ideation, plan and intent",
        },
        8: {
            "severity_level": 3,
            "response": "Immediate engagement",
            "category": "Attempt or imminent risk",
        },
    }

    match = matrix.get(level)
    if match:
        return match["severity_level"], match["response"], match["category"]
    return None, "Unknown", "Unknown"


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    provider: str = "openai",
    model: str = "gpt-5.6-luna",
) -> float:
    """Calculates cost based on standard and cached token usage."""
    default_rates = [0.20, 1.20, 0.02]
    rates = PRICING_MAP.get(provider.lower(), {}).get(model.lower(), default_rates)

    cached_rate = rates[2] if len(rates) > 2 else (rates[0] * 0.10)

    cost = (
        (prompt_tokens / 1_000_000 * rates[0])
        + (completion_tokens / 1_000_000 * rates[1])
        + (cached_tokens / 1_000_000 * cached_rate)
    )

    return round(cost, 6)


def calculate_job_costs(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
) -> tuple[float, float, float]:
    """Calculates actual job cost, cost without caching, and net savings using PRICING_MAP."""
    default_rates = [0.20, 1.20, 0.02]
    rates = PRICING_MAP.get(provider.lower(), {}).get(model.lower(), default_rates)

    standard_input_rate = rates[0]
    output_rate = rates[1]
    cached_lookup_rate = rates[2] if len(rates) > 2 else (standard_input_rate * 0.10)

    actual_cost = (
        (prompt_tokens / 1_000_000 * standard_input_rate)
        + (completion_tokens / 1_000_000 * output_rate)
        + (cached_tokens / 1_000_000 * cached_lookup_rate)
    )
    uncached_cost = (
        (prompt_tokens + cached_tokens) / 1_000_000 * standard_input_rate
    ) + (completion_tokens / 1_000_000 * output_rate)
    savings = max(0.0, uncached_cost - actual_cost)

    return round(actual_cost, 6), round(uncached_cost, 6), round(savings, 6)


def compute_metrics(
    results: list[dict], provider: str = "gemini", model: str = "gemini-3.6-flash"
) -> dict:
    """Aggregates performance, accuracy, and detailed context caching metrics."""
    total = len(results)
    if total == 0:
        return {}

    # 1. Accuracy Calculations
    exact_matches = sum(
        1 for r in results if r.get("metrics", {}).get("is_exact_match", False)
    )
    actionable_matches = sum(
        1 for r in results if r.get("metrics", {}).get("is_actionable_match", False)
    )

    exact_accuracy = (exact_matches / total) * 100
    actionable_accuracy = (actionable_matches / total) * 100

    # 2. Performance & Total Cost
    avg_latency = sum(r.get("latency", 0) for r in results) / total
    total_cost = sum(r.get("inference_cost_usd", 0) for r in results)

    # 3. Context Caching Cost Comparison Calculations
    total_cached_tokens = sum(r.get("cached_tokens", 0) for r in results)

    default_rates = [1.50, 7.50, 0.15]
    rates = PRICING_MAP.get(provider.lower(), {}).get(model.lower(), default_rates)

    standard_input_rate = rates[0]
    cached_lookup_rate = rates[2] if len(rates) > 2 else (standard_input_rate * 0.10)

    actual_cached_cost = (total_cached_tokens / 1_000_000) * cached_lookup_rate
    cost_if_not_cached = (total_cached_tokens / 1_000_000) * standard_input_rate
    net_savings = cost_if_not_cached - actual_cached_cost

    return {
        "benchmark_summary": {
            "total_samples": total,
            "exact_score_accuracy": f"{exact_accuracy:.2f}%",
            "actionable_severity_accuracy": f"{actionable_accuracy:.2f}%",
            "average_latency_sec": f"{avg_latency:.2f}s",
            "total_cached_tokens": total_cached_tokens,
            "actual_cached_tokens_cost_usd": f"${actual_cached_cost:.6f}",
            "cost_if_not_cached_usd": f"${cost_if_not_cached:.6f}",
            "context_cache_savings_usd": f"${net_savings:.6f}",
            "total_cost_usd": f"${total_cost:.6f}",
        },
        "itemized_results": results,
    }


def save_metrics(metrics: dict, output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def get_severity(label: str) -> str:
    severity_map = {
        "No ideation": "Low",
        "Emotional distress": "Low",
        "Ambiguous risk": "Moderate",
        "Passive wish to be dead": "Moderate",
        "Active ideation, no method": "High",
        "Active ideation, method, no intent": "High",
        "Active ideation, method and intent": "Critical",
        "Active ideation, plan and intent": "Critical",
        "Attempt or imminent risk": "Critical",
    }
    return severity_map.get(label, "Unknown")
