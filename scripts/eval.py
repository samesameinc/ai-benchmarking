import argparse
import json
import os
from inference import generate_ai_response
from judges import judge_ground_truth
from utils import calculate_cost, compute_metrics, save_metrics, get_severity


CODE_TO_LABEL = {
    0: "No ideation",
    1: "Emotional distress",
    2: "Ambiguous risk",
    3: "Passive wish to be dead",
    4: "Active ideation, no method",
    5: "Active ideation, method, no intent",
    6: "Active ideation, method and intent",
    7: "Active ideation, plan and intent",
    8: "Attempt or imminent risk"
}


def run_benchmark(data_path, output_path, provider, model, judge_model):
    with open(data_path, "r") as f:
        dataset = json.load(f)

    results = []

    for item in dataset:
        print(f"Evaluating ID: {item['id']}...")

        # 1. Inference (Captured: Text, Risk_Score, Latency, Tokens)
        inf = generate_ai_response(item["user_query"], provider=provider, model=model)

        # 2. Audit (Qualitative Safety)
        scores = judge_ground_truth(
            query=item["user_query"],
            correct_score=item["correct_risk_score"],
            prediction_text=inf["explanation"],
            judge_model=judge_model
        )

        # 3. Accuracy Check
        predicted_label = CODE_TO_LABEL.get(inf["risk_score"], str(inf["risk_score"]))

        is_correct = str(predicted_label).strip().lower() == str(item["correct_risk_score"]).strip().lower()

        # 4. Cost Calculation
        cost = calculate_cost(
            inf["prompt_tokens"],
            inf["completion_tokens"],
            provider=provider,
            model=model
        )

        results.append({
            "id": item["id"],
            "predicted_risk_score": predicted_label,
            "actual_risk_score": item["correct_risk_score"],
            "is_correct": is_correct,
            "scores": scores,
            "inference_latency_sec": round(inf["latency"], 2),
            "inference_cost_usd": cost
        })

    metrics = compute_metrics(results)
    save_metrics(metrics, output_path)
    print(f"Benchmark Complete. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--judge-model", type=str, default="gemini-2.5-flash")

    args = parser.parse_args()

    # Ensure you pass args.judge_model here:
    run_benchmark(args.data, args.output, args.provider, args.model, args.judge_model)