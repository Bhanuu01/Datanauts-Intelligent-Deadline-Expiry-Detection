import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List


DEFAULT_METRICS_PATH = "/tmp/feedback_metrics.json"
DEFAULT_OUTPUT_PATH = "/tmp/retrain_decision.json"


def load_feedback_metrics() -> Dict[str, Any]:
    metrics_path = Path(os.getenv("FEEDBACK_METRICS_PATH", DEFAULT_METRICS_PATH))
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())

    return {
        "new_feedback_events": int(os.getenv("NEW_FEEDBACK_EVENTS", "0")),
        "correction_rate_7d": float(os.getenv("CORRECTION_RATE_7D", "0.0")),
    }


def should_retrain(metrics: Dict[str, Any]) -> Dict[str, Any]:
    min_feedback = int(os.getenv("MIN_FEEDBACK_EVENTS", "500"))
    max_correction_rate = float(os.getenv("MAX_CORRECTION_RATE", "0.15"))

    feedback_count = int(metrics.get("new_feedback_events", 0))
    correction_rate = float(metrics.get("correction_rate_7d", 0.0))

    reasons: List[str] = []
    if feedback_count >= min_feedback:
        reasons.append("feedback_volume")
    if correction_rate >= max_correction_rate:
        reasons.append("correction_rate")

    return {
        "trigger_retrain": bool(reasons),
        "reasons": reasons,
        "metrics": {
            "new_feedback_events": feedback_count,
            "correction_rate_7d": correction_rate,
        },
        "thresholds": {
            "min_feedback_events": min_feedback,
            "max_correction_rate": max_correction_rate,
        },
    }


def run_command(command: List[str]) -> int:
    process = subprocess.run(command, check=False)
    return process.returncode


def orchestrate_training() -> Dict[str, Any]:
    training_root = Path(os.getenv("TRAINING_ROOT", "/app/components/training"))
    config = {
        "ner_model": os.getenv("NER_MODEL_NAME", "bert_ner_v5"),
        "classifier_model": os.getenv("CLASSIFIER_MODEL_NAME", "roberta_clf_v5"),
    }

    commands = [
        ["python", str(training_root / "src" / "train_ner.py"), "--model", config["ner_model"]],
        ["python", str(training_root / "src" / "train_classifier.py"), "--model", config["classifier_model"]],
    ]

    results = []
    for command in commands:
        exit_code = run_command(command)
        results.append({"command": command, "exit_code": exit_code})
        if exit_code != 0:
            return {"ok": False, "steps": results}

    return {"ok": True, "steps": results}


def main() -> int:
    decision = should_retrain(load_feedback_metrics())
    output_path = Path(os.getenv("RETRAIN_DECISION_PATH", DEFAULT_OUTPUT_PATH))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if os.getenv("DRY_RUN", "false").lower() == "true":
        output_path.write_text(json.dumps(decision, indent=2))
        print(json.dumps(decision, indent=2))
        return 0

    if not decision["trigger_retrain"]:
        decision["status"] = "skipped"
        output_path.write_text(json.dumps(decision, indent=2))
        print(json.dumps(decision, indent=2))
        return 0

    training_result = orchestrate_training()
    decision["status"] = "completed" if training_result["ok"] else "failed"
    decision["training"] = training_result
    output_path.write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))
    return 0 if training_result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
