import json


def calculate_cost(prompt_tokens, completion_tokens, provider="openai", model="gpt-4o"):
    """
    Calculates cost based on token usage.
    Prices reflect estimated May 2026 rates per 1 million tokens.
    """
    pricing_map = {
        "openai": {
            "gpt-4o": [2.50, 10.00],
            "gpt-4o-mini": [0.15, 0.60],
        },
        "gemini": {
            "gemini-3.1-flash-lite-preview": [0.075, 0.30],
            "gemini-3.1-flash": [0.10, 0.40],
            "gemini-3.1-pro": [1.25, 3.75],
        },
        "anthropic": {
            "claude-3-5-sonnet-20240620": [3.00, 15.00],
        }
    }

    rates = pricing_map.get(provider.lower(), {}).get(model.lower(), [2.50, 10.00])
    cost = (prompt_tokens / 1_000_000 * rates[0]) + (completion_tokens / 1_000_000 * rates[1])
    return round(cost, 6)

def compute_metrics(results):
    """Aggregates performance and accuracy metrics."""
    total = len(results)
    if total == 0: return {}

    # 1. Accuracy Calculation
    matches = sum(1 for r in results if r["is_correct"])
    accuracy = (matches / total) * 100

    # 2. Performance & Cost
    avg_latency = sum(r["inference_latency_sec"] for r in results) / total
    total_cost = sum(r["inference_cost_usd"] for r in results)

    return {
        "benchmark_summary": {
            "total_samples": total,
            "accuracy": f"{accuracy:.2f}%",
            "average_latency_sec": f"{avg_latency:.2f}s",
            "total_cost_usd": f"${total_cost:.6f}"
        },
        "itemized_results": results
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
        "Attempt or imminent risk": "Critical"
    }
    return severity_map.get(label, "Unknown")