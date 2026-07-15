import json


def get_severity_metrics(risk_level):
    """
    Deterministic clinical truth table mapping standard risk codes (0-8)
    to production Severity Levels and Severity Responses.
    """
    try:
        level = int(risk_level)
    except (ValueError, TypeError):
        return None, "Unknown", "Unknown"

    matrix = {
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


def calculate_cost(prompt_tokens, completion_tokens, cached_tokens=0, provider="openai", model="gpt-4o"):
    """
    Calculates cost based on standard and cached token usage.
    """
    pricing_map = {
        "openai": {
            "gpt-4o": [2.50, 10.00, 1.25],  # [Input, Output, Cached]
            "gpt-4o-mini": [0.15, 0.60, 0.075],
        },
        "gemini": {
            "gemini-3.1-flash-lite": [0.075, 0.30, 0.0075],  # $0.0075 per Million cached lookups!
            "gemini-3.1-flash": [0.10, 0.40, 0.01],
            "gemini-3.1-pro": [1.25, 3.75, 0.125],
        },
        "anthropic": {
            "claude-3-5-sonnet-20240620": [3.00, 15.00, 0.30],
        },
    }

    rates = pricing_map.get(provider.lower(), {}).get(model.lower(), [2.50, 10.00, 1.25])

    # If the model doesn't explicitly have a 3rd cache index in our map, default it to half price
    cached_rate = rates[2] if len(rates) > 2 else (rates[0] * 0.5)

    # Core Math Equation: Standard Inputs + Output Generation + Cheap Cached Hits
    cost = (prompt_tokens / 1_000_000 * rates[0]) + \
           (completion_tokens / 1_000_000 * rates[1]) + \
           (cached_tokens / 1_000_000 * cached_rate)

    return round(cost, 6)


def compute_metrics(results, provider="gemini", model="gemini-3.1-flash-lite"):
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

    pricing_map = {
        "openai": {
            "gpt-4o": [2.50, 10.00, 1.25],
            "gpt-4o-mini": [0.15, 0.60, 0.075],
        },
        "gemini": {
            "gemini-3.1-flash-lite": [0.075, 0.30, 0.0075],  # [Input, Output, Cached Lookup]
            "gemini-3.1-flash": [0.10, 0.40, 0.01],
            "gemini-3.1-pro": [1.25, 3.75, 0.125],
        },
        "anthropic": {
            "claude-3-5-sonnet-20240620": [3.00, 15.00, 0.30],
        },
    }

    rates = pricing_map.get(provider.lower(), {}).get(model.lower(), [0.075, 0.30, 0.0075])
    standard_input_rate = rates[0]
    cached_lookup_rate = rates[2]

    # Calculate what they did cost vs what they would have costed regularly
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


def save_metrics(metrics, output_path):
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def get_severity(label):
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