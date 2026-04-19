import json
import os
import sys
from pathlib import Path
from typing import Any
from typing import Dict


DEFAULT_EVAL_PATH = "/tmp/evaluation_metrics.json"
DEFAULT_PROMOTION_PATH = "/tmp/promotion_decision.json"


def load_metrics() -> Dict[str, Any]:
    eval_path = Path(os.getenv("EVALUATION_METRICS_PATH", DEFAULT_EVAL_PATH))
    if eval_path.exists():
        return json.loads(eval_path.read_text())

    return {
        "candidate_f1": float(os.getenv("CANDIDATE_F1", "0.0")),
        "baseline_f1": float(os.getenv("BASELINE_F1", "0.0")),
        "candidate_latency_p95_ms": float(os.getenv("CANDIDATE_LATENCY_P95_MS", "0.0")),
        "max_latency_p95_ms": float(os.getenv("MAX_LATENCY_P95_MS", "500")),
    }


def evaluate_promotion(metrics: Dict[str, Any]) -> Dict[str, Any]:
    candidate_f1 = float(metrics.get("candidate_f1", 0.0))
    baseline_f1 = float(metrics.get("baseline_f1", 0.0))
    candidate_latency = float(metrics.get("candidate_latency_p95_ms", 0.0))
    max_latency = float(metrics.get("max_latency_p95_ms", os.getenv("MAX_LATENCY_P95_MS", "500")))

    promote = candidate_f1 >= baseline_f1 and candidate_latency <= max_latency
    return {
        "promote": promote,
        "candidate_f1": candidate_f1,
        "baseline_f1": baseline_f1,
        "candidate_latency_p95_ms": candidate_latency,
        "max_latency_p95_ms": max_latency,
        "reason": "candidate_meets_quality_and_latency_gates" if promote else "candidate_failed_quality_or_latency_gate",
    }


def main() -> int:
    decision = evaluate_promotion(load_metrics())
    output_path = Path(os.getenv("PROMOTION_DECISION_PATH", DEFAULT_PROMOTION_PATH))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))
    return 0 if decision["promote"] else 1


if __name__ == "__main__":
    sys.exit(main())
