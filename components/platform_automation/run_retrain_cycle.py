import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

from mlflow.tracking import MlflowClient


DEFAULT_METRICS_PATH = "/tmp/feedback_metrics.json"
DEFAULT_OUTPUT_PATH = "/tmp/retrain_decision.json"
DEFAULT_EVAL_PATH = "/tmp/evaluation_metrics.json"
DEFAULT_RELEASE_ROOT = "/data/model-releases"


def load_feedback_metrics() -> Dict[str, Any]:
    metrics_path = Path(os.getenv("FEEDBACK_METRICS_PATH", DEFAULT_METRICS_PATH))
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())

    feedback_log_path = Path(os.getenv("FEEDBACK_LOG_PATH", "/data/feedback_events.jsonl"))
    if feedback_log_path.exists():
        lines = [json.loads(line) for line in feedback_log_path.read_text().splitlines() if line.strip()]
        feedback_events = len(lines)
        correction_events = sum(
            1 for line in lines if line.get("event") in {"dismiss", "edit", "manual_add"}
        )
        correction_rate = (correction_events / feedback_events) if feedback_events else 0.0
        return {
            "new_feedback_events": feedback_events,
            "correction_rate_7d": correction_rate,
        }

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


def run_command(command: List[str], env: Dict[str, str] | None = None) -> int:
    process = subprocess.run(command, check=False, env=env)
    return process.returncode


def locate_model_dir(prefix: str, model_name: str) -> Path:
    candidate = Path(f"/tmp/{prefix}-{model_name}")
    if not candidate.exists():
        raise FileNotFoundError(f"Expected trained model directory at {candidate}")
    return candidate


def package_candidate(ner_model_name: str, classifier_model_name: str) -> Dict[str, Any]:
    release_root = Path(os.getenv("MODEL_RELEASES_ROOT", DEFAULT_RELEASE_ROOT))
    release_root.mkdir(parents=True, exist_ok=True)

    candidate_version = os.getenv(
        "CANDIDATE_VERSION",
        f"candidate-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
    )
    candidate_root = release_root / candidate_version
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    candidate_root.mkdir(parents=True, exist_ok=True)

    ner_src = locate_model_dir("deadline-ner", ner_model_name)
    classifier_src = locate_model_dir("deadline-clf", classifier_model_name)
    ner_dst = candidate_root / "ner"
    classifier_dst = candidate_root / "classifier"
    shutil.copytree(ner_src, ner_dst)
    shutil.copytree(classifier_src, classifier_dst)

    package_manifest = {
        "candidate_version": candidate_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ner_model_name": ner_model_name,
        "classifier_model_name": classifier_model_name,
        "ner_path": str(ner_dst),
        "classifier_path": str(classifier_dst),
    }
    (candidate_root / "package_manifest.json").write_text(json.dumps(package_manifest, indent=2))
    return package_manifest


def latest_experiment_metrics(tracking_uri: str, experiment_name: str) -> Dict[str, Any]:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No MLflow runs found for experiment: {experiment_name}")

    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
    }


def build_evaluation_metrics(package_manifest: Dict[str, Any]) -> Dict[str, Any]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.platform.svc.cluster.local:5000")
    training_root = Path(os.getenv("TRAINING_ROOT", "/app/components/training"))
    evaluation_path = Path(os.getenv("EVALUATION_METRICS_PATH", DEFAULT_EVAL_PATH))
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)

    eval_env = os.environ.copy()
    eval_env["MLFLOW_TRACKING_URI"] = tracking_uri
    eval_env["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
        "MLFLOW_S3_ENDPOINT_URL", "http://minio.platform.svc.cluster.local:9000"
    )

    eval_command = [
        "python",
        str(training_root / "src" / "evaluate.py"),
        "--clf_model",
        package_manifest["classifier_path"],
        "--ner_model",
        package_manifest["ner_path"],
        "--threshold",
        os.getenv("EVAL_THRESHOLD", "0.7"),
        "--output_json",
        str(evaluation_path),
    ]

    cross_domain_samples = os.getenv("CROSS_DOMAIN_SAMPLES_PATH")
    if cross_domain_samples:
        eval_command.extend(["--cross_domain_samples", cross_domain_samples])

    if run_command(eval_command, env=eval_env) != 0:
        raise RuntimeError("End-to-end evaluation failed")

    evaluation_metrics = json.loads(evaluation_path.read_text())
    ner_metrics = latest_experiment_metrics(tracking_uri, os.getenv("NER_EXPERIMENT_NAME", "deadline-detection-ner"))
    clf_metrics = latest_experiment_metrics(
        tracking_uri, os.getenv("CLASSIFIER_EXPERIMENT_NAME", "deadline-detection-classifier")
    )

    evaluation_metrics.update(
        {
            "candidate_version": package_manifest["candidate_version"],
            "candidate_paths": {
                "ner": package_manifest["ner_path"],
                "classifier": package_manifest["classifier_path"],
            },
            "ner_f1": float(ner_metrics["metrics"].get("test_f1", 0.0)),
            "clf_macro_f1": float(clf_metrics["metrics"].get("test_f1", 0.0)),
            "mlflow_runs": {
                "ner": ner_metrics["run_id"],
                "classifier": clf_metrics["run_id"],
            },
        }
    )
    evaluation_path.write_text(json.dumps(evaluation_metrics, indent=2))
    return evaluation_metrics


def orchestrate_training() -> Dict[str, Any]:
    training_root = Path(os.getenv("TRAINING_ROOT", "/app/components/training"))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.platform.svc.cluster.local:5000")
    shared_env = os.environ.copy()
    shared_env["MLFLOW_TRACKING_URI"] = tracking_uri
    shared_env["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
        "MLFLOW_S3_ENDPOINT_URL", "http://minio.platform.svc.cluster.local:9000"
    )

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
        exit_code = run_command(command, env=shared_env)
        results.append({"command": command, "exit_code": exit_code})
        if exit_code != 0:
            return {"ok": False, "steps": results, "config": config}

    package_manifest = package_candidate(config["ner_model"], config["classifier_model"])
    evaluation_metrics = build_evaluation_metrics(package_manifest)
    return {
        "ok": True,
        "steps": results,
        "config": config,
        "package_manifest": package_manifest,
        "evaluation_metrics": evaluation_metrics,
    }


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
    if training_result.get("package_manifest"):
        decision["candidate_version"] = training_result["package_manifest"]["candidate_version"]
        decision["candidate_paths"] = {
            "ner": training_result["package_manifest"]["ner_path"],
            "classifier": training_result["package_manifest"]["classifier_path"],
        }
    output_path.write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))
    return 0 if training_result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
