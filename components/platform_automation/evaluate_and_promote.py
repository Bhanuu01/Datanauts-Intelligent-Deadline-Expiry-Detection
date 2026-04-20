import json
import os
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List


DEFAULT_EVAL_PATH = "/tmp/evaluation_metrics.json"
DEFAULT_PROMOTION_PATH = "/tmp/promotion_decision.json"
DEFAULT_SERVING_METRICS_PATH = "/tmp/serving_metrics.json"


def load_metrics() -> Dict[str, Any]:
    eval_path = Path(os.getenv("EVALUATION_METRICS_PATH", DEFAULT_EVAL_PATH))
    if eval_path.exists():
        return json.loads(eval_path.read_text())

    return {
        "ner_f1": float(os.getenv("NER_F1", "0.0")),
        "clf_macro_f1": float(os.getenv("CLF_MACRO_F1", "0.0")),
        "e2e_coverage": float(os.getenv("E2E_COVERAGE", "0.0")),
        "false_alarm_count": int(os.getenv("FALSE_ALARM_COUNT", "999999")),
        "candidate_latency_p95_ms": float(os.getenv("CANDIDATE_LATENCY_P95_MS", "0.0")),
        "max_latency_p95_ms": float(os.getenv("MAX_LATENCY_P95_MS", "500")),
    }


def load_serving_metrics() -> Dict[str, Any]:
    metrics_path = Path(os.getenv("SERVING_METRICS_PATH", DEFAULT_SERVING_METRICS_PATH))
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())

    return {
        "live_error_rate": float(os.getenv("LIVE_ERROR_RATE", "0.0")),
        "live_latency_p95_ms": float(os.getenv("LIVE_LATENCY_P95_MS", "0.0")),
        "live_correction_rate_7d": float(os.getenv("LIVE_CORRECTION_RATE_7D", "0.0")),
    }


def evaluate_promotion(metrics: Dict[str, Any], serving_metrics: Dict[str, Any]) -> Dict[str, Any]:
    ner_f1 = float(metrics.get("ner_f1", metrics.get("ner", {}).get("f1", 0.0)))
    clf_macro_f1 = float(metrics.get("clf_macro_f1", metrics.get("classifier", {}).get("macro_f1", 0.0)))
    e2e_coverage = float(metrics.get("e2e_coverage", metrics.get("e2e", {}).get("coverage", 0.0)))
    false_alarm_count = int(metrics.get("false_alarm_count", metrics.get("e2e", {}).get("false_alarm_count", 999999)))
    candidate_latency = float(metrics.get("candidate_latency_p95_ms", 0.0))
    max_latency = float(metrics.get("max_latency_p95_ms", os.getenv("MAX_LATENCY_P95_MS", "500")))
    min_ner_f1 = float(os.getenv("MIN_NER_F1", "0.65"))
    min_clf_macro_f1 = float(os.getenv("MIN_CLF_MACRO_F1", "0.75"))
    min_e2e_coverage = float(os.getenv("MIN_E2E_COVERAGE", "0.60"))
    max_false_alarm_count = int(os.getenv("MAX_FALSE_ALARM_COUNT", "10"))
    max_live_error_rate = float(os.getenv("MAX_LIVE_ERROR_RATE", "0.05"))
    max_live_latency_p95_ms = float(os.getenv("MAX_LIVE_LATENCY_P95_MS", str(max_latency)))
    max_live_correction_rate_7d = float(os.getenv("MAX_LIVE_CORRECTION_RATE_7D", "0.15"))

    failures: List[str] = []
    if ner_f1 < min_ner_f1:
        failures.append("ner_f1_below_threshold")
    if clf_macro_f1 < min_clf_macro_f1:
        failures.append("clf_macro_f1_below_threshold")
    if e2e_coverage < min_e2e_coverage:
        failures.append("e2e_coverage_below_threshold")
    if false_alarm_count > max_false_alarm_count:
        failures.append("false_alarm_count_above_threshold")
    if candidate_latency and candidate_latency > max_latency:
        failures.append("latency_p95_above_threshold")

    live_error_rate = float(serving_metrics.get("live_error_rate", 0.0))
    live_latency_p95_ms = float(serving_metrics.get("live_latency_p95_ms", 0.0))
    live_correction_rate_7d = float(serving_metrics.get("live_correction_rate_7d", 0.0))
    rollback_reasons: List[str] = []
    if live_error_rate > max_live_error_rate:
        rollback_reasons.append("live_error_rate_above_threshold")
    if live_latency_p95_ms and live_latency_p95_ms > max_live_latency_p95_ms:
        rollback_reasons.append("live_latency_p95_above_threshold")
    if live_correction_rate_7d > max_live_correction_rate_7d:
        rollback_reasons.append("live_correction_rate_above_threshold")

    promote = not failures
    rollback = bool(rollback_reasons)
    return {
        "promote": promote,
        "rollback": rollback,
        "ner_f1": ner_f1,
        "clf_macro_f1": clf_macro_f1,
        "e2e_coverage": e2e_coverage,
        "false_alarm_count": false_alarm_count,
        "candidate_latency_p95_ms": candidate_latency,
        "max_latency_p95_ms": max_latency,
        "thresholds": {
            "min_ner_f1": min_ner_f1,
            "min_clf_macro_f1": min_clf_macro_f1,
            "min_e2e_coverage": min_e2e_coverage,
            "max_false_alarm_count": max_false_alarm_count,
            "max_latency_p95_ms": max_latency,
            "max_live_error_rate": max_live_error_rate,
            "max_live_latency_p95_ms": max_live_latency_p95_ms,
            "max_live_correction_rate_7d": max_live_correction_rate_7d,
        },
        "failed_gates": failures,
        "rollback_reasons": rollback_reasons,
        "candidate_version": metrics.get("candidate_version"),
        "candidate_paths": metrics.get("candidate_paths", {}),
        "reason": "candidate_meets_all_training_and_latency_gates" if promote else "candidate_failed_one_or_more_gates",
        "live_metrics": {
            "live_error_rate": live_error_rate,
            "live_latency_p95_ms": live_latency_p95_ms,
            "live_correction_rate_7d": live_correction_rate_7d,
        },
    }


def main() -> int:
    decision = evaluate_promotion(load_metrics(), load_serving_metrics())
    output_path = Path(os.getenv("PROMOTION_DECISION_PATH", DEFAULT_PROMOTION_PATH))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))
    return 0 if decision["promote"] else 1


if __name__ == "__main__":
    sys.exit(main())
